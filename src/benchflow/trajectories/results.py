"""Emit Verifiers-shaped ``results.jsonl`` artifacts for BenchFlow rollouts.

This is the canonical trainer-facing rollout surface. It intentionally lives
beside, not inside, raw traces:

- ``trajectory/llm_trajectory.jsonl`` remains the provider HTTP audit log.
- ``trajectory/acp_trajectory.jsonl`` remains the ACP event audit log.
- ``results.jsonl`` is a Verifiers/Prime-RL-shaped rollout record.

The writer is fail-closed for training readiness but not for artifact
emission: even errored or unstructured rollouts get one JSONL row with an
``error``. The trainer-shaped trajectory is produced only from healthy
``llm_trajectory.jsonl`` exchanges; ACP is never used as a training fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, cast, overload

from benchflow._utils.json_safe import scrub_non_finite
from benchflow._utils.scoring import finite_reward
from benchflow.diagnostics import RolloutDiagnostics
from benchflow.trajectories.export_prime_sft import (
    PrimeSftTrajectoryJsonlError,
    _assistant_from_chat_response,
    _messages_from_chat_request,
    _messages_from_responses_request,
    normalize_prime_sft_exchange,
    prime_sft_last_user_training_window,
    validate_prime_sft_row,
)
from benchflow.trajectories.types import redact_trajectory_obj
from benchflow.usage_tracking import (
    USAGE_SOURCE_AGENT_NATIVE_ACP,
    USAGE_SOURCE_AGENT_REPORTED,
    USAGE_SOURCE_NONE,
)

if TYPE_CHECKING:
    from benchflow.review.outcome import ScoringResult

ROLLOUT_RESULTS_FILENAME = "results.jsonl"
JOB_RESULTS_FILENAME = "results.jsonl"
#: ``info.schema_version`` of a BenchFlow row. Rows without it are version 1,
#: where an unscored rollout had ``reward`` and ``score`` 0.0 (its ``metrics``
#: then has no ``reward`` key). From version 2 both are null when unscored.
RESULTS_ROW_VERSION = 2
JOB_RESULTS_ERRORS_FILENAME = "results.errors.json"

logger = logging.getLogger(__name__)


def _record_to_redacted_json_line(record: dict[str, Any]) -> str:
    redacted = redact_trajectory_obj(scrub_non_finite(record))
    return json.dumps(redacted, default=str, allow_nan=False)


def _structured_diagnostics_info(
    *,
    error_category: str | None,
    verifier_error_category: str | None,
    diagnostics: RolloutDiagnostics | None,
) -> dict[str, Any]:
    """Build the optional, redacted trainer-facing diagnostics envelope.

    ``result.json`` remains the full rollout-status artifact. This compact
    envelope carries the same typed failure evidence into the canonical
    ``results.jsonl`` row without changing its existing error or stop fields.
    Callers that do not provide structured diagnostics retain the historical
    row shape.
    """
    block = (diagnostics or RolloutDiagnostics()).to_results_jsonl_block(
        error_category=error_category,
        verifier_error_category=verifier_error_category,
    )
    if block is None:
        return {}
    safe_block = redact_trajectory_obj(scrub_non_finite(block))
    return {"diagnostics": safe_block}


def _reward_value(rewards: dict[str, Any] | None) -> float | None:
    """Scalar row reward: null when absent or not finite.

    A missing verdict is unscored, never 0: an infrastructure failure written
    as 0.0 reads as a real failed attempt to anything counting this file.
    """
    if not isinstance(rewards, dict):
        return None
    reward = rewards.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        return finite_reward(reward)
    return None


def _metrics_from_rewards(
    rewards: dict[str, Any] | None,
    *,
    n_tool_calls: int,
    n_prompts: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n_tool_calls": n_tool_calls,
        "n_prompts": n_prompts,
    }
    if not isinstance(rewards, dict):
        return metrics
    for key, value in rewards.items():
        if key == "rubric":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[str(key)] = finite_reward(value)
    nested_metrics = rewards.get("metrics")
    if isinstance(nested_metrics, dict):
        for key, value in nested_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[str(key)] = finite_reward(value)
    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for idx, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            item = cast(dict[str, Any], item)
            score = item.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                metrics[str(item.get("name") or f"rubric_{idx}")] = finite_reward(score)
    return metrics


def _token_usage_from_agent_result(agent_result: dict[str, Any]) -> dict[str, Any]:
    input_tokens = _nonnegative_number(agent_result.get("n_input_tokens"))
    output_tokens = _nonnegative_number(agent_result.get("n_output_tokens"))
    total_tokens = _nonnegative_number(agent_result.get("total_tokens"))
    if input_tokens + output_tokens == 0 and total_tokens > 0:
        # Some harness/provider paths only expose a provider total. Prime-RL's
        # token-batch path needs final_input_tokens + final_output_tokens to be
        # non-zero; keep the total visible without inventing an output split.
        input_tokens = total_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "final_input_tokens": input_tokens,
        "final_output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _nonnegative_number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return 0.0


def _error_payload(
    *,
    error: str | None,
    verifier_error: str | None,
    export_error: str | None,
) -> dict[str, str] | None:
    for key, value in (
        ("agent_error", error),
        ("verifier_error", verifier_error),
        ("export_error", export_error),
    ):
        if value:
            return {
                "error": key,
                "error_chain_str": value,
                "error_chain_repr": value,
            }
    return None


def _stop_condition(
    *,
    error: str | None,
    verifier_error: str | None,
    export_error: str | None,
    partial_trajectory: bool,
) -> str:
    if partial_trajectory:
        return "partial_trajectory"
    if error:
        return "agent_error"
    if verifier_error:
        return "verifier_error"
    if export_error:
        return "export_error"
    return "agent_completed"


def _llm_steps_from_trajectory(
    rollout_dir: Path,
    *,
    reward: float | None,
    is_truncated: bool,
    trajectory_id_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    steps: list[dict[str, Any]] = []
    tool_defs: list[dict[str, Any]] = []
    if not path.exists():
        return steps, tool_defs, None
    try:
        with path.open("rb") as handle:
            exchanges = _IndexedExchanges(handle, path)
            return _llm_steps_from_exchanges(
                rollout_dir,
                exchanges,
                path,
                reward=reward,
                is_truncated=is_truncated,
                trajectory_id_prefix=trajectory_id_prefix,
            )
    except (PrimeSftTrajectoryJsonlError, OSError) as exc:
        return [], [], f"Invalid LLM trajectory JSONL: {exc}"


class _IndexedExchanges(Sequence[dict[str, Any]]):
    """An offset index over a validated log, retaining no request/response bodies.

    Training selection needs earlier requests and later tool replays. Seek for
    those records instead of holding the entire HTTP log and normalized histories.
    The caller owns the open handle for the duration of the export.
    """

    def __init__(self, handle: BinaryIO, path: Path):
        self._handle = handle
        self._offsets: list[int] = []
        line_num = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_num += 1
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise PrimeSftTrajectoryJsonlError(
                    f"{path}: line {line_num}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise PrimeSftTrajectoryJsonlError(
                    f"{path}: line {line_num}: top-level record must be an object"
                )
            self._offsets.append(offset)

    def __len__(self) -> int:
        return len(self._offsets)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        self._handle.seek(self._offsets[index])
        return json.loads(self._handle.readline())


def _llm_steps_from_exchanges(
    rollout_dir: Path,
    exchanges: Sequence[dict[str, Any]],
    path: Path,
    *,
    reward: float | None,
    is_truncated: bool,
    trajectory_id_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    steps: list[dict[str, Any]] = []
    tool_defs: list[dict[str, Any]] = []
    training_success_indices = _training_success_exchange_indices(exchanges)
    owners = _step_owners(rollout_dir, exchanges, path)
    skipped_successful: list[str] = []
    for exchange_idx, exchange in enumerate(exchanges):
        response = exchange.get("response")
        if exchange_idx not in training_success_indices:
            continue
        if not isinstance(response, dict):
            skipped_successful.append(
                f"exchange {exchange_idx}: selected response is not an object"
            )
            continue
        normalized, skip_reason = normalize_prime_sft_exchange(exchange)
        if normalized is None:
            skipped_successful.append(
                f"exchange {exchange_idx}: {skip_reason or 'normalization failed'}"
            )
            continue
        prompt = normalized.messages[:-1]
        completion = normalized.messages[-1:]
        if not completion:
            skipped_successful.append(f"exchange {exchange_idx}: no completion")
            continue
        if normalized.tool_defs:
            known_names = {
                tool.get("function", {}).get("name")
                for tool in tool_defs
                if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
            }
            tool_defs.extend(
                tool
                for tool in normalized.tool_defs
                if isinstance(tool, dict)
                and isinstance(tool.get("function"), dict)
                and tool["function"].get("name") not in known_names
            )
        response_body = (
            cast(dict[str, Any], response.get("body"))
            if isinstance(response.get("body"), dict)
            else {}
        )
        extras = {
            "source": "llm_trajectory",
            "tracking_source": "litellm_callback",
            "exchange_index": exchange_idx,
            **owners.get(exchange_idx, {}),
        }
        exchange_metadata = exchange.get("metadata")
        if isinstance(exchange_metadata, dict):
            extras.update(
                {
                    str(key): value
                    for key, value in exchange_metadata.items()
                    if key not in extras
                }
            )
        step = {
            "prompt": prompt,
            "completion": completion,
            "response": response_body,
            "tokens": None,
            "reward": reward,
            "advantage": 0.0,
            "is_truncated": _response_is_truncated(response_body) or is_truncated,
            "trajectory_id": f"{trajectory_id_prefix}__llm_{exchange_idx}",
            "extras": extras,
        }
        steps.append(step)
    if skipped_successful:
        return (
            steps,
            tool_defs,
            "Successful LLM exchanges were omitted from results.jsonl: "
            + "; ".join(skipped_successful),
        )
    return steps, tool_defs, None


def _step_owners(
    rollout_dir: Path,
    exchanges: Sequence[dict[str, Any]],
    path: Path,
) -> dict[int, dict[str, Any]]:
    """Which agent made each successful call, recorded while the capture is whole.

    A results.jsonl row keeps only calls a later request consumed, so a
    subagent's spawning call can be missing from its steps (a capture that
    ends inside the subagent) and the row alone cannot attribute them. The
    tags use the same attribution as converting the rollout directory and are
    written only when the capture shows subagent activity: ``agent_role`` is
    ``parent``, ``subagent`` (with ``parent_tool_call_id``),
    ``helper`` or, when attribution fails, ``unattributed``.
    """
    from benchflow.trajectories.sft_subagents import (
        SubagentAttributionError,
        attribute_rollout_exchanges,
        load_acp_parent_links,
        subagent_row_tags,
    )

    # The same candidates bench train convert attributes from the rollout
    # directory: every call the provider answered, consumed later or not.
    successful = [
        index
        for index, exchange in enumerate(exchanges)
        if isinstance(response := exchange.get("response"), dict)
        and response.get("status_code") == 200
    ]
    try:
        attribution = attribute_rollout_exchanges(
            exchanges,
            successful,
            acp_parent_links=load_acp_parent_links(rollout_dir),
            source=str(path),
        )
    except SubagentAttributionError:
        return {index: {"agent_role": "unattributed"} for index in successful}
    if attribution is None:
        # No spawn call and no ACP linkage: every call is the parent's, and
        # the rows stay as they were before attribution existed.
        return {}
    owners: dict[int, dict[str, Any]] = {}
    for index in successful:
        owner = attribution.by_exchange.get(index)
        if owner is None:
            owners[index] = {"agent_role": "unattributed"}
        elif owner.role == "subagent" and owner.spawn is not None:
            owners[index] = subagent_row_tags(owner.spawn)
        else:
            owners[index] = {"agent_role": owner.role}
    return owners


def _response_is_truncated(response_body: dict[str, Any]) -> bool:
    return bool(
        response_body.get("incomplete_details")
        or response_body.get("status") == "incomplete"
    )


def _response_is_training_success(response: Any) -> bool:
    if not isinstance(response, dict) or response.get("status_code") != 200:
        return False
    body = response.get("body")
    if not isinstance(body, dict):
        return False
    status = body.get("status")
    if status is not None and status != "completed":
        return False
    # Legacy chat and native Anthropic use finish/stop reasons instead of a
    # Responses status. Only terminal, usable completions enter training.
    if "stop_reason" in body and body["stop_reason"] not in (
        "end_turn",
        "tool_use",
        "stop_sequence",
    ):
        return False
    choices = body.get("choices")
    if isinstance(choices, list) and any(
        isinstance(choice, dict)
        and "finish_reason" in choice
        and choice["finish_reason"] not in ("stop", "tool_calls", "function_call")
        for choice in choices
    ):
        return False
    return not bool(body.get("incomplete_details"))


def _training_success_exchange_indices(
    exchanges: Sequence[dict[str, Any]],
) -> set[int]:
    candidates_by_request: dict[bytes, list[int]] = {}
    for exchange_idx, exchange in enumerate(exchanges):
        if not _response_is_training_success(exchange.get("response")):
            continue
        request_body = json.dumps(
            (
                request.get("body")
                if isinstance(request := exchange.get("request"), dict)
                and isinstance(request.get("body"), dict)
                else {}
            ),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        request_key = hashlib.sha256(request_body.encode()).digest()
        candidates_by_request.setdefault(request_key, []).append(exchange_idx)

    selected: set[int] = set()
    for candidates in candidates_by_request.values():
        consumed = [
            exchange_idx
            for exchange_idx in candidates
            if _response_consumed_by_later_request(exchanges, exchange_idx)
        ]
        if consumed:
            # Identical prompts can belong to independently consumed branches
            # (for example two subagents). Only repeated captures of the same
            # protocol calls are duplicates; distinct consumed calls all count.
            by_calls: dict[tuple[tuple[str, tuple[str, str] | None], ...], int] = {}
            for exchange_idx in consumed:
                calls = _response_tool_calls(
                    exchanges[exchange_idx]["response"]["body"]
                )
                identity = tuple(
                    (call["id"], _call_signature(call))
                    for call in sorted(calls, key=lambda call: call["id"])
                )
                by_calls[identity] = exchange_idx
            selected.update(by_calls.values())
            continue
        safe_unconsumed = [
            exchange_idx
            for exchange_idx in candidates
            if _response_safe_without_later_consumption(exchanges, exchange_idx)
        ]
        if safe_unconsumed:
            selected.add(max(safe_unconsumed))
    return selected


def _response_consumed_by_later_request(
    exchanges: Sequence[dict[str, Any]],
    exchange_idx: int,
) -> bool:
    response = exchanges[exchange_idx].get("response")
    body = response.get("body") if isinstance(response, dict) else {}
    body = body if isinstance(body, dict) else {}
    calls = _response_tool_calls(body)
    if not calls:
        return False
    # Only legacy function_call has no protocol ID. Never compare a generated
    # row ID with request data; missing IDs in other protocols fail closed.
    message = body.get("message")
    choices = body.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
    legacy = (
        isinstance(message, dict)
        and message.get("tool_calls") is None
        and isinstance(message.get("function_call"), dict)
    )
    idless = any(not call["id"] for call in calls)
    if not legacy and idless:
        return False
    prefix = _request_training_history(exchanges[exchange_idx])
    for later_idx in range(exchange_idx + 1, len(exchanges)):
        history = _request_training_history(exchanges[later_idx])
        # Without a protocol ID the replay must extend this exact prompt. An
        # exact ID remains useful after the runtime compacts older history.
        if idless and history[: len(prefix)] != prefix:
            continue
        offsets = [len(prefix)] if idless else range(len(history))
        if any(
            _history_consumes_calls(calls, history, offset, legacy=legacy)
            for offset in offsets
        ):
            return True
    return False


def _history_consumes_calls(
    calls: list[dict[str, Any]],
    history: list[dict[str, Any]],
    offset: int,
    *,
    legacy: bool,
) -> bool:
    if (
        offset >= len(history)
        or history[offset].get("role") != "assistant"
        or (offset > 0 and history[offset - 1].get("role") == "assistant")
    ):
        return False
    # Responses replays each parallel call as its own assistant item; chat
    # history may group the same calls. Match the entire consecutive assistant
    # turn, never a prefix/suffix or calls on opposite sides of another role.
    replay_calls: list[dict[str, Any]] = []
    result_offset = offset
    while (
        result_offset < len(history)
        and history[result_offset].get("role") == "assistant"
    ):
        replay_calls.extend(history[result_offset].get("tool_calls") or [])
        result_offset += 1
    # Parallel calls may be replayed in another order. Their protocol IDs, not
    # list positions, bind each call to its signature and result.
    ordered_calls = sorted(calls, key=lambda call: call["id"])
    ordered_replay = sorted(replay_calls, key=lambda call: call["id"])
    if len(replay_calls) != len(calls) or any(
        (call["id"] and call["id"] != replay_call["id"])
        or _call_signature(call) is None
        or _call_signature(call) != _call_signature(replay_call)
        for call, replay_call in zip(ordered_calls, ordered_replay, strict=True)
    ):
        return False
    pending = {call["id"] for call in replay_calls}
    if len(pending) != len(replay_calls):
        return False
    previous_result_id = None
    for result in history[result_offset:]:
        if result.get("role") != "tool":
            break
        call_id = result.get("tool_call_id")
        # The export normalizer merges adjacent chunks of one tool result.
        if previous_result_id is not None and call_id == previous_result_id:
            continue
        if call_id not in pending:
            break
        if not call_id and (
            not legacy or result.get("name") != calls[0]["function"]["name"]
        ):
            break
        pending.remove(call_id)
        previous_result_id = call_id
        if not pending:
            return True
    return False


def _call_signature(call: dict[str, Any]) -> tuple[str, str] | None:
    function = call["function"]
    try:
        arguments = json.loads(function["arguments"])
    except (TypeError, ValueError):
        return None
    if not isinstance(arguments, dict):
        return None
    return function["name"], json.dumps(
        arguments, sort_keys=True, separators=(",", ":")
    )


def _request_training_history(exchange: dict[str, Any]) -> list[dict[str, Any]]:
    request = exchange.get("request")
    body = request.get("body") if isinstance(request, dict) else {}
    if not isinstance(body, dict):
        return []
    normalizer = (
        _messages_from_chat_request
        if "messages" in body
        else _messages_from_responses_request
    )
    history = normalizer(body, redact=False, synthetic_ids=False)
    for message in history:
        for call in message.get("tool_calls") or []:
            signature = _call_signature(call)
            if signature is not None:
                call["function"]["arguments"] = signature[1]
    return history


def _response_safe_without_later_consumption(
    exchanges: Sequence[dict[str, Any]], exchange_idx: int
) -> bool:
    exchange = exchanges[exchange_idx]
    response = exchange.get("response")
    body = response.get("body") if isinstance(response, dict) else {}
    calls = _response_tool_calls(body if isinstance(body, dict) else {})
    if not calls or all(call["function"]["name"] == "finish" for call in calls):
        return True
    return not any(
        isinstance(request := later.get("request"), dict)
        and isinstance(request.get("body"), dict)
        for later in (exchanges[i] for i in range(exchange_idx + 1, len(exchanges)))
    )


def _response_tool_calls(body: dict[str, Any]) -> list[dict[str, Any]]:
    # Eligibility must see the same calls that enter the training row. Sharing
    # normalization covers native Anthropic blocks, legacy function_call,
    # Chat Completions and Responses without a second protocol parser drifting.
    message = _assistant_from_chat_response(body, redact=False, synthetic_ids=False)
    if message is None:
        return []
    return message.get("tool_calls") or []


def _top_level_prompt_completion(
    steps: list[dict[str, Any]],
    prompts: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    if not steps:
        prompt = [{"role": "user", "content": prompt_text} for prompt_text in prompts]
        return prompt, None
    prompt = list(steps[0].get("prompt") or [])
    final_step = steps[-1]
    full_messages = list(final_step.get("prompt") or []) + list(
        final_step.get("completion") or []
    )
    typed_full_messages = [
        message for message in full_messages if isinstance(message, dict)
    ]
    window = prime_sft_last_user_training_window(typed_full_messages)
    if window is not None:
        return window
    if prompt and len(full_messages) >= len(prompt):
        return prompt, full_messages[len(prompt) :]
    return prompt, list(final_step.get("completion") or [])


def build_rollout_results_record(
    rollout_dir: str | Path,
    *,
    task_name: str,
    rollout_name: str,
    agent: str,
    agent_name: str,
    model: str | None,
    n_tool_calls: int,
    prompts: list[str],
    trajectory: list[dict[str, Any]],
    partial_trajectory: bool,
    rewards: dict[str, Any] | None,
    error: str | None,
    verifier_error: str | None,
    error_category: str | None = None,
    verifier_error_category: str | None = None,
    diagnostics: RolloutDiagnostics | None = None,
    export_error: str | None = None,
    timing: dict[str, Any] | None = None,
    agent_result: dict[str, Any] | None = None,
    example_id: int = 0,
    scoring: ScoringResult | None = None,
    purpose: Literal["task", "reviewer"] = "task",
    parent_rollout: str | None = None,
    branches: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rollout_path = Path(rollout_dir)
    if scoring is not None and scoring.status == "error":
        verifier_error = verifier_error or scoring.error
        rewards = None
    reward = _reward_value(rewards)
    trajectory_id_prefix = (
        rollout_name
        if rollout_name.startswith(f"{task_name}__")
        else f"{task_name}__{rollout_name}"
    )
    is_truncated = bool(partial_trajectory)
    steps, tool_defs, llm_export_error = _llm_steps_from_trajectory(
        rollout_path,
        reward=reward,
        is_truncated=is_truncated,
        trajectory_id_prefix=trajectory_id_prefix,
    )
    prompt, completion = _top_level_prompt_completion(steps, prompts)
    validation_error = _prime_sft_validation_error(
        prompt=prompt,
        completion=completion,
        tool_defs=tool_defs,
    )
    effective_export_error = export_error or llm_export_error or validation_error
    error_obj = _error_payload(
        error=error,
        verifier_error=verifier_error,
        export_error=effective_export_error,
    )
    terminal_health_error = bool(error or verifier_error or partial_trajectory)
    # No proxy runs for a native login, or for an agent that routes its own
    # model calls (agent-reported or none): no LLM trajectory is
    # expected, so its absence is not an export error.
    native_subscription_without_llm = bool(
        (agent_result or {}).get("usage_source")
        in (
            USAGE_SOURCE_AGENT_NATIVE_ACP,
            USAGE_SOURCE_AGENT_REPORTED,
            USAGE_SOURCE_NONE,
        )
        and not (rollout_path / "trajectory" / "llm_trajectory.jsonl").exists()
        and effective_export_error is None
        and not terminal_health_error
    )
    training_ready = bool(
        steps
        and completion
        and effective_export_error is None
        and not terminal_health_error
    )
    training_ready_reason = None
    if not training_ready:
        if llm_export_error:
            training_ready_reason = (
                "invalid_llm_trajectory_jsonl"
                if llm_export_error.startswith("Invalid LLM trajectory JSONL:")
                else "export_error"
            )
        elif validation_error:
            training_ready_reason = "invalid_prime_sft_row"
        elif effective_export_error:
            training_ready_reason = "export_error"
        elif not steps or not completion:
            training_ready_reason = "missing_healthy_structured_llm_trajectory"
        elif partial_trajectory:
            training_ready_reason = "partial_trajectory"
        elif error:
            training_ready_reason = "agent_error"
        elif verifier_error:
            training_ready_reason = "verifier_error"
        else:
            training_ready_reason = "missing_healthy_structured_llm_trajectory"
        if error_obj is None and not native_subscription_without_llm:
            error_name = (
                training_ready_reason
                if training_ready_reason
                in {"agent_error", "verifier_error", "partial_trajectory"}
                else "missing_llm_trajectory"
            )
            error_obj = {
                "error": error_name,
                "error_chain_str": (
                    "Rollout is not training-ready: "
                    f"{training_ready_reason or 'unknown'}"
                ),
                "error_chain_repr": (
                    "Rollout is not training-ready: "
                    f"{training_ready_reason or 'unknown'}"
                ),
            }
    completed = error_obj is None and not partial_trajectory
    token_usage = _token_usage_from_agent_result(agent_result or {})
    metrics = _metrics_from_rewards(
        rewards,
        n_tool_calls=n_tool_calls,
        n_prompts=len(prompts),
    )
    record = {
        "example_id": example_id,
        "prompt": prompt,
        "completion": completion,
        "info": {
            "schema_version": RESULTS_ROW_VERSION,
            "task_id": task_name,
            "task_name": task_name,
            "rollout_name": rollout_name,
            "environment": task_name,
            "agent": agent,
            "agent_name": agent_name,
            "model": model,
            "source": "benchflow",
            "rollout_dir": str(rollout_path),
            "training_ready": training_ready,
            "training_ready_reason": training_ready_reason,
            **_structured_diagnostics_info(
                error_category=error_category,
                verifier_error_category=verifier_error_category,
                diagnostics=diagnostics,
            ),
            **(
                {"reward_details": rewards.get("details")}
                if isinstance(rewards, dict)
                and isinstance(rewards.get("details"), dict)
                else {}
            ),
            # Branch lineage (result.json's `branches` block): which forks
            # this rollout made and where each child's archive lives.
            **({"branches": branches} if branches is not None else {}),
        },
        "reward": reward,
        "error": error_obj,
        "timing": timing or {},
        "is_completed": completed,
        "is_truncated": is_truncated,
        "stop_condition": _stop_condition(
            error=error,
            verifier_error=verifier_error,
            export_error=effective_export_error,
            partial_trajectory=partial_trajectory,
        ),
        "metrics": metrics,
        "tool_defs": tool_defs,
        "token_usage": token_usage,
        "score": reward,
        "total_tool_calls": float(n_tool_calls),
        "trajectory": steps,
    }
    if scoring is not None:
        record["scoring"] = scoring.to_dict()
        record["passed"] = scoring.passed
    if purpose != "task":
        record["purpose"] = purpose
        record["parent_rollout"] = parent_rollout
    return record


def _prime_sft_validation_error(
    *,
    prompt: list[dict[str, Any]],
    completion: list[dict[str, Any]] | None,
    tool_defs: list[dict[str, Any]],
) -> str | None:
    if not completion:
        return None
    try:
        validate_prime_sft_row(
            {
                "prompt": prompt,
                "completion": completion,
                "tool_defs": tool_defs,
            },
            1,
        )
    except ValueError as exc:
        return f"Prime-SFT results row validation failed: {exc}"
    return None


def write_rollout_results_jsonl(
    rollout_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write one Verifiers-shaped row to ``rollout_dir/results.jsonl``."""
    record = build_rollout_results_record(rollout_dir, **kwargs)
    out = Path(rollout_dir) / ROLLOUT_RESULTS_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_record_to_redacted_json_line(record) + "\n")
    return record


def write_job_results_jsonl(job_dir: str | Path) -> Path | None:
    """Aggregate per-rollout ``results.jsonl`` files into ``job_dir/results.jsonl``."""
    job_path = Path(job_dir)
    if not job_path.is_dir():
        return None
    # A legacy resume preserves the old job aggregate here, not a trial export.
    # Other children may contain only exports, including repeated attempts.
    rollout_files = sorted(
        path
        for path in job_path.glob(f"*/{ROLLOUT_RESULTS_FILENAME}")
        if path.parent.name != ".pre-0.8"
    )
    if not rollout_files:
        return None

    example_ids: dict[str, int] = {}
    skipped_errors: list[dict[str, Any]] = []
    out = job_path / JOB_RESULTS_FILENAME
    with out.open("w", encoding="utf-8") as handle:
        for src in rollout_files:
            try:
                lines = src.read_text().splitlines()
            except OSError as exc:
                logger.warning("Skipping unreadable results artifact %s: %s", src, exc)
                skipped_errors.append(
                    {
                        "path": str(src),
                        "error": "unreadable_results_artifact",
                        "message": str(exc),
                    }
                )
                continue
            for line_num, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping invalid results artifact row %s:%s: %s",
                        src,
                        line_num,
                        exc,
                    )
                    skipped_errors.append(
                        {
                            "path": str(src),
                            "line": line_num,
                            "error": "invalid_results_artifact_json",
                            "message": str(exc),
                        }
                    )
                    continue
                if not isinstance(row, dict):
                    logger.warning(
                        "Skipping non-object results artifact row %s:%s",
                        src,
                        line_num,
                    )
                    skipped_errors.append(
                        {
                            "path": str(src),
                            "line": line_num,
                            "error": "non_object_results_artifact_row",
                        }
                    )
                    continue
                if row.get("purpose") == "reviewer":
                    continue
                key = _job_example_key(row, src.parent)
                row["example_id"] = example_ids.setdefault(key, len(example_ids))
                handle.write(_record_to_redacted_json_line(row) + "\n")
    _write_job_results_errors(job_path, skipped_errors)
    return out


def _write_job_results_errors(
    job_path: Path,
    skipped_errors: list[dict[str, Any]],
) -> None:
    out = job_path / JOB_RESULTS_ERRORS_FILENAME
    if not skipped_errors:
        with suppress(FileNotFoundError):
            out.unlink()
        return
    payload = {
        "schema_version": 1,
        "error": "skipped_results_artifact_rows",
        "skipped_count": len(skipped_errors),
        "skipped": skipped_errors,
    }
    out.write_text(_record_to_redacted_json_line(payload) + "\n")


def _job_example_key(row: dict[str, Any], rollout_dir: Path) -> str:
    info = row.get("info")
    if isinstance(info, dict):
        task_id = info.get("task_id") or info.get("task_name")
        if task_id:
            return str(task_id)
    task_id = row.get("task_id") or row.get("task_name")
    if task_id:
        return str(task_id)
    return rollout_dir.name
