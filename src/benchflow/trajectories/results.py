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

import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from benchflow._utils.json_safe import scrub_non_finite
from benchflow.trajectories.export_prime_sft import (
    PrimeSftTrajectoryJsonlError,
    load_llm_trajectory_jsonl,
    normalize_prime_sft_exchange,
    prime_sft_last_user_training_window,
    validate_prime_sft_row,
)
from benchflow.trajectories.types import redact_trajectory_obj

ROLLOUT_RESULTS_FILENAME = "results.jsonl"
JOB_RESULTS_FILENAME = "results.jsonl"
JOB_RESULTS_ERRORS_FILENAME = "results.errors.json"

logger = logging.getLogger(__name__)


def _record_to_redacted_json_line(record: dict[str, Any]) -> str:
    redacted = redact_trajectory_obj(scrub_non_finite(record))
    return json.dumps(redacted, default=str, allow_nan=False)


def _reward_value(rewards: dict[str, Any] | None) -> float:
    if not isinstance(rewards, dict):
        return 0.0
    reward = rewards.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        return float(reward)
    return 0.0


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
            metrics[str(key)] = float(value)
    nested_metrics = rewards.get("metrics")
    if isinstance(nested_metrics, dict):
        for key, value in nested_metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[str(key)] = float(value)
    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for idx, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            item = cast(dict[str, Any], item)
            score = item.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                metrics[str(item.get("name") or f"rubric_{idx}")] = float(score)
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
    reward: float,
    is_truncated: bool,
    trajectory_id_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    steps: list[dict[str, Any]] = []
    tool_defs: list[dict[str, Any]] = []
    if not path.exists():
        return steps, tool_defs, None
    try:
        exchanges = load_llm_trajectory_jsonl(path, strict=True)
    except PrimeSftTrajectoryJsonlError as exc:
        return [], [], f"Invalid LLM trajectory JSONL: {exc}"
    skipped_successful: list[str] = []
    for exchange_idx, exchange in enumerate(exchanges):
        response = exchange.get("response")
        if not isinstance(response, dict) or response.get("status_code") != 200:
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
        }
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


def _response_is_truncated(response_body: dict[str, Any]) -> bool:
    if response_body.get("incomplete_details"):
        return True
    truncation = response_body.get("truncation")
    if isinstance(truncation, str):
        return truncation.strip().lower() not in {"", "disabled", "false", "none"}
    return bool(truncation)


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
    export_error: str | None = None,
    timing: dict[str, Any] | None = None,
    agent_result: dict[str, Any] | None = None,
    example_id: int = 0,
) -> dict[str, Any]:
    rollout_path = Path(rollout_dir)
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
        if error_obj is None:
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
            **(
                {"reward_details": rewards.get("details")}
                if isinstance(rewards, dict)
                and isinstance(rewards.get("details"), dict)
                else {}
            ),
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
    rollout_files = sorted(job_path.glob(f"*/{ROLLOUT_RESULTS_FILENAME}"))
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
