#!/usr/bin/env python3
"""Fail-closed BenchFlow run-artifact health validator.

This script is intentionally self-contained so it can travel with the
benchflow-experiment-review skill. It checks completed rollout folders for the
minimum artifact contract required before a trial can be treated as a healthy
BenchFlow model result:

- result.json parses.
- trajectory/acp_trajectory.jsonl exists, is non-empty, parses as JSONL, and
  contains agent-side events.
- trajectory/llm_trajectory.jsonl exists, is non-empty, parses as JSONL, and
  contains real provider request and response records.
- results.jsonl exists, is non-empty, parses as JSONL, and contains a
  Prime-RL/Verifiers-shaped training-readiness row backed by the LLM trajectory.
- token usage, timing, and tool usage metadata are present.

Any violation makes that rollout unhealthy. The script exits 1 if any checked
rollout is unhealthy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TOKEN_KEYS = {
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "promptTokenCount",
    "candidatesTokenCount",
    "totalTokenCount",
}

ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}
BANNED_TRAINING_ROW_KEYS = {
    "gold",
    "gold_solution",
    "verify_source",
    "tools_py",
    "initial_db",
    "db_json",
    "target_constants",
    "private_reasoning",
    "reasoning_content",
    "thinking_blocks",
}
BANNED_MESSAGE_KEYS = {
    "reasoning_content",
    "thinking_blocks",
    "private_reasoning",
    "provider_specific_fields",
    "function_call",
}

INFRA_ERROR_MARKERS = (
    "missing required",
    "api key",
    "credential",
    "provider",
    "sandbox",
    "docker",
    "daytona",
    "no space left",
    "transport",
    "pipe closed",
    "connection",
)


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(errors="replace"))
    except OSError as exc:
        return None, f"{path}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"{path}: JSON parse error at line {exc.lineno}: {exc.msg}"
    if not isinstance(value, dict):
        return None, f"{path}: expected JSON object"
    return value, None


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    try:
        raw = path.read_text(errors="replace")
    except OSError as exc:
        return [], [f"{path}: {exc}"]
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{path}:{lineno}: JSON parse error: {exc.msg}")
            continue
        if not isinstance(row, dict):
            issues.append(f"{path}:{lineno}: expected JSON object")
            continue
        rows.append(row)
    if not rows and not issues:
        issues.append(f"{path}: empty JSONL")
    return rows, issues


def iter_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        out = [value]
        for child in value.values():
            out.extend(iter_dicts(child))
        return out
    if isinstance(value, list):
        out: list[dict[str, Any]] = []
        for child in value:
            out.extend(iter_dicts(child))
        return out
    return []


def has_token_usage(value: Any) -> bool:
    for obj in iter_dicts(value):
        for key in TOKEN_KEYS:
            token_value = obj.get(key)
            if (
                isinstance(token_value, (int, float))
                and not isinstance(token_value, bool)
                and token_value > 0
            ):
                return True
    return False


def positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def numeric_token_total(result: dict[str, Any]) -> int | None:
    sources: list[dict[str, Any]] = []
    agent_result = result.get("agent_result")
    if isinstance(agent_result, dict):
        sources.append(agent_result)
    token_usage = result.get("token_usage")
    if isinstance(token_usage, dict):
        sources.append(token_usage)
    sources.append(result)

    for source in sources:
        total = source.get("total_tokens") or source.get("n_total_tokens")
        if isinstance(total, int) and not isinstance(total, bool):
            return total
        input_tokens = (
            source.get("n_input_tokens")
            if source.get("n_input_tokens") is not None
            else source.get("input_tokens")
        )
        output_tokens = (
            source.get("n_output_tokens")
            if source.get("n_output_tokens") is not None
            else source.get("output_tokens")
        )
        if (
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
        ):
            return input_tokens + output_tokens
        prompt_tokens = source.get("prompt_tokens")
        completion_tokens = source.get("completion_tokens")
        if (
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
        ):
            return prompt_tokens + completion_tokens
    return None


def reward_present(result: dict[str, Any]) -> bool:
    rewards = result.get("rewards")
    if isinstance(rewards, dict):
        return rewards.get("reward") is not None
    return result.get("reward") is not None


def timing_present(result: dict[str, Any]) -> bool:
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    return bool(
        result.get("started_at")
        and (result.get("finished_at") or timing.get("total") is not None)
    ) or bool(
        timing.get("started_at")
        and (timing.get("ended_at") or timing.get("duration_seconds") is not None)
    )


def tool_usage_count(result: dict[str, Any]) -> int | None:
    n_tool_calls = result.get("n_tool_calls")
    if isinstance(n_tool_calls, int) and not isinstance(n_tool_calls, bool):
        return n_tool_calls
    tool_usage = result.get("tool_usage")
    if isinstance(tool_usage, dict):
        total = 0
        found = False
        for value in tool_usage.values():
            if isinstance(value, int) and not isinstance(value, bool):
                total += value
                found = True
        if found:
            return total
    return None


def is_oracle_result(result: dict[str, Any], run_config: dict[str, Any] | None) -> bool:
    values = [result.get("agent"), result.get("agent_name")]
    if run_config:
        values.extend([run_config.get("agent"), run_config.get("harness")])
    return any(str(value).strip().lower() == "oracle" for value in values if value)


def validate_acp(rows: list[dict[str, Any]], path: Path) -> list[str]:
    issues: list[str] = []
    agent_events = 0
    for index, row in enumerate(rows, start=1):
        event_type = row.get("type")
        if not isinstance(event_type, str) or not event_type:
            issues.append(f"{path}:{index}: missing string 'type'")
        phase = str(row.get("phase") or "").lower()
        if phase != "verifier":
            agent_events += 1
    if agent_events == 0:
        issues.append(f"{path}: no agent-side events")
    return issues


def validate_llm(
    rows: list[dict[str, Any]], path: Path
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    request_count = 0
    response_count = 0
    completed_candidates: list[tuple[int, str, bool]] = []
    error_count = 0
    usage_count = 0
    for index, row in enumerate(rows, start=1):
        request = row.get("request")
        response = row.get("response")
        error = row.get("error")
        if isinstance(request, dict):
            body = request.get("body")
            if isinstance(body, dict):
                request_count += 1
            else:
                issues.append(f"{path}:{index}: request missing object body")
        if isinstance(response, dict):
            body = response.get("body")
            if isinstance(body, dict):
                response_count += 1
                response_status = body.get("status")
                completed = (
                    response.get("status_code") == 200
                    and response_status in {None, "completed"}
                    and not body.get("incomplete_details")
                )
                if completed:
                    signature = json.dumps(
                        (
                            request.get("body")
                            if isinstance(request, dict)
                            and isinstance(request.get("body"), dict)
                            else {}
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    completed_candidates.append(
                        (index - 1, signature, has_token_usage(body))
                    )
                if has_token_usage(body):
                    usage_count += 1
            else:
                issues.append(f"{path}:{index}: response missing object body")
        if isinstance(error, dict):
            error_count += 1
    if request_count == 0:
        issues.append(f"{path}: no provider request bodies")
    if response_count == 0:
        issues.append(f"{path}: no provider response bodies")
    if request_count and response_count and response_count < request_count:
        issues.append(
            f"{path}: provider response count {response_count} < request count {request_count}"
        )
    if usage_count == 0:
        issues.append(f"{path}: no provider token usage in response bodies")
    candidates_by_request: dict[str, list[tuple[int, bool]]] = {}
    for index, signature, has_usage in completed_candidates:
        candidates_by_request.setdefault(signature, []).append((index, has_usage))
    request_bodies = [
        json.dumps(
            (
                request.get("body")
                if isinstance(request := row.get("request"), dict)
                and isinstance(request.get("body"), dict)
                else {}
            ),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for row in rows
    ]
    selected: list[tuple[int, bool]] = []
    for candidates in candidates_by_request.values():
        consumed = [
            candidate
            for candidate in candidates
            if response_consumed_by_later_request(rows, request_bodies, candidate[0])
        ]
        if consumed:
            selected.append(max(consumed))
            continue
        safe_unconsumed = [
            candidate
            for candidate in candidates
            if response_safe_without_later_consumption(rows, candidate[0])
        ]
        if safe_unconsumed:
            selected.append(max(safe_unconsumed))
    selected.sort()
    successful_exchange_indices = [index for index, _ in selected]
    successful_response_count = len(selected)
    successful_usage_count = sum(1 for _, has_usage in selected if has_usage)
    if successful_response_count and successful_usage_count < successful_response_count:
        issues.append(
            f"{path}: completed responses with usage {successful_usage_count} < "
            f"completed responses {successful_response_count}"
        )
    return issues, {
        "requests": request_count,
        "responses": response_count,
        "successful_responses": successful_response_count,
        "successful_exchange_indices": successful_exchange_indices,
        "successful_responses_with_usage": successful_usage_count,
        "deduplicated_completed_responses": (
            len(completed_candidates) - successful_response_count
        ),
        "errors": error_count,
        "responses_with_usage": usage_count,
    }


def response_consumed_by_later_request(
    rows: list[dict[str, Any]],
    request_bodies: list[str],
    exchange_idx: int,
) -> bool:
    response = rows[exchange_idx].get("response")
    body = response.get("body") if isinstance(response, dict) else {}
    call_ids = response_call_ids(body if isinstance(body, dict) else {})
    return bool(
        call_ids
        and any(
            any(call_id in request_body for call_id in call_ids)
            for request_body in request_bodies[exchange_idx + 1 :]
        )
    )


def response_call_ids(body: dict[str, Any]) -> set[str]:
    return {call_id for call_id, _ in response_tool_calls(body)}


def response_safe_without_later_consumption(
    rows: list[dict[str, Any]], exchange_idx: int
) -> bool:
    row = rows[exchange_idx]
    response = row.get("response")
    body = response.get("body") if isinstance(response, dict) else {}
    calls = response_tool_calls(body if isinstance(body, dict) else {})
    if not calls or all(name == "finish" for _, name in calls):
        return True
    return not any(
        isinstance(request := later.get("request"), dict)
        and isinstance(request.get("body"), dict)
        for later in rows[exchange_idx + 1 :]
    )


def response_tool_calls(body: dict[str, Any]) -> list[tuple[str, str | None]]:
    calls = [
        (
            str(item.get("call_id") or item.get("id")),
            str(item.get("name")) if item.get("name") else None,
        )
        for item in body.get("output") or []
        if isinstance(item, dict)
        and item.get("type") in {"function_call", "tool_call"}
        and (item.get("call_id") or item.get("id"))
    ]
    choices = body.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                continue
            calls.extend(
                (
                    str(call.get("id") or call.get("tool_call_id")),
                    (
                        str(function.get("name"))
                        if isinstance(function := call.get("function"), dict)
                        and function.get("name")
                        else str(call.get("name"))
                        if call.get("name")
                        else None
                    ),
                )
                for call in message.get("tool_calls") or []
                if isinstance(call, dict)
                and (call.get("id") or call.get("tool_call_id"))
            )
    message = body.get("message")
    if isinstance(message, dict):
        calls.extend(
            (
                str(call.get("id") or call.get("tool_call_id")),
                (
                    str(function.get("name"))
                    if isinstance(function := call.get("function"), dict)
                    and function.get("name")
                    else str(call.get("name"))
                    if call.get("name")
                    else None
                ),
            )
            for call in message.get("tool_calls") or []
            if isinstance(call, dict) and (call.get("id") or call.get("tool_call_id"))
        )
    return calls


def result_has_terminal_error(result: dict[str, Any]) -> bool:
    if result.get("error") or result.get("verifier_error"):
        return True
    return result.get("partial_trajectory") is True


def normalize_training_messages(
    row: dict[str, Any], row_path: str, issues: list[str]
) -> list[dict[str, Any]]:
    messages = row.get("messages")
    if messages is None:
        prompt = row.get("prompt")
        completion = row.get("completion")
        if isinstance(prompt, list) and isinstance(completion, list):
            messages = prompt + completion
    if not isinstance(messages, list) or not messages:
        issues.append(f"{row_path}: missing non-empty messages or prompt+completion")
        return []
    typed: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            issues.append(f"{row_path}: messages[{idx}] must be an object")
            continue
        typed.append(message)
    return typed


def validate_training_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[Any] | None,
    row_path: str,
) -> list[str]:
    issues: list[str] = []
    for idx, message in enumerate(messages):
        leaked = sorted(BANNED_MESSAGE_KEYS.intersection(message))
        if leaked:
            issues.append(
                f"{row_path}: messages[{idx}] has banned keys: {', '.join(leaked)}"
            )
        role = message.get("role")
        if role not in ALLOWED_MESSAGE_ROLES:
            issues.append(f"{row_path}: messages[{idx}].role invalid: {role!r}")
        if role == "system" and idx != 0:
            issues.append(
                f"{row_path}: system message must be at index 0, got index {idx}"
            )
        if "content" not in message and "tool_calls" not in message:
            issues.append(f"{row_path}: messages[{idx}] needs content or tool_calls")
        if message.get("tool_calls") and role != "assistant":
            issues.append(f"{row_path}: only assistant messages may contain tool_calls")
        if role == "tool" and not message.get("tool_call_id"):
            issues.append(f"{row_path}: tool message requires tool_call_id")

    if not any(message.get("role") == "assistant" for message in messages):
        issues.append(f"{row_path}: no assistant message")
    has_tool_calls = any(bool(message.get("tool_calls")) for message in messages)
    if has_tool_calls and not tools:
        issues.append(
            f"{row_path}: assistant tool_calls require non-empty tool_defs/tools"
        )
    if tools is not None:
        for idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                issues.append(f"{row_path}: tool_defs[{idx}] must be an object")
    return issues


def normalize_tool_defs(
    row: dict[str, Any], row_path: str
) -> tuple[list[Any] | None, str | None]:
    tools = row.get("tool_defs", row.get("tools"))
    if tools is None:
        return None, None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            return None, f"{row_path}: tool_defs/tools is not valid JSON: {exc.msg}"
    if not isinstance(tools, list):
        return None, f"{row_path}: tool_defs/tools must be a list"
    return tools, None


def validate_results_row(
    row: dict[str, Any],
    *,
    row_path: str,
    result: dict[str, Any],
    llm_summary: dict[str, Any] | None,
) -> list[str]:
    issues: list[str] = []
    leaked = sorted(BANNED_TRAINING_ROW_KEYS.intersection(row))
    if leaked:
        issues.append(f"{row_path}: banned leakage keys present: {', '.join(leaked)}")

    info = row.get("info")
    if not isinstance(info, dict):
        issues.append(f"{row_path}: missing object info")
        info = {}
    task_name = result.get("task_name")
    row_task = info.get("task_id") or info.get("task_name") or row.get("task_name")
    if task_name and row_task and str(task_name) != str(row_task):
        issues.append(
            f"{row_path}: task mismatch result={task_name!r} row={row_task!r}"
        )

    training_ready = info.get("training_ready")
    if not isinstance(training_ready, bool):
        issues.append(f"{row_path}: info.training_ready must be boolean")
    if result_has_terminal_error(result) and training_ready:
        issues.append(
            f"{row_path}: terminal errored/partial rollout marked training_ready=true"
        )
    if not result_has_terminal_error(result) and training_ready is False:
        issues.append(f"{row_path}: healthy rollout marked training_ready=false")

    row_error = row.get("error")
    if training_ready:
        if row_error is not None:
            issues.append(f"{row_path}: training_ready row carries non-null error")
        if info.get("training_ready_reason") is not None:
            issues.append(
                f"{row_path}: training_ready row has non-null training_ready_reason"
            )
        if row.get("is_completed") is not True:
            issues.append(f"{row_path}: training_ready row is not completed")
        if row.get("is_truncated") is True:
            issues.append(f"{row_path}: training_ready row is truncated")
        if row.get("stop_condition") not in {None, "agent_completed"}:
            issues.append(
                f"{row_path}: unexpected stop_condition for training-ready row"
            )
    else:
        if not info.get("training_ready_reason"):
            issues.append(
                f"{row_path}: non-training-ready row lacks training_ready_reason"
            )
        if row_error is None:
            issues.append(f"{row_path}: non-training-ready row lacks error payload")

    messages = normalize_training_messages(row, row_path, issues)
    tools, tool_error = normalize_tool_defs(row, row_path)
    if tool_error:
        issues.append(tool_error)
    issues.extend(validate_training_messages(messages, tools=tools, row_path=row_path))

    if training_ready:
        for field in ("prompt", "completion", "trajectory"):
            value = row.get(field)
            if not isinstance(value, list) or not value:
                issues.append(
                    f"{row_path}: training_ready row missing non-empty {field}"
                )
        trajectory = row.get("trajectory")
        if isinstance(trajectory, list):
            successful_responses = (
                int(llm_summary.get("successful_responses", 0)) if llm_summary else 0
            )
            if successful_responses and len(trajectory) != successful_responses:
                issues.append(
                    f"{row_path}: results trajectory steps {len(trajectory)} != "
                    f"successful LLM responses {successful_responses}"
                )
            expected_indices = (
                set(llm_summary.get("successful_exchange_indices", []))
                if llm_summary
                else set()
            )
            observed_indices: set[int] = set()
            for idx, step in enumerate(trajectory):
                if not isinstance(step, dict):
                    issues.append(f"{row_path}: trajectory[{idx}] must be an object")
                    continue
                if not isinstance(step.get("prompt"), list) or not isinstance(
                    step.get("completion"), list
                ):
                    issues.append(
                        f"{row_path}: trajectory[{idx}] missing prompt/completion lists"
                    )
                extras = step.get("extras")
                if (
                    isinstance(extras, dict)
                    and extras.get("source") != "llm_trajectory"
                ):
                    issues.append(
                        f"{row_path}: trajectory[{idx}] source is not llm_trajectory"
                    )
                if isinstance(extras, dict) and isinstance(
                    extras.get("exchange_index"), int
                ):
                    observed_indices.add(extras["exchange_index"])
                if step.get("is_truncated") is True:
                    issues.append(
                        f"{row_path}: trajectory[{idx}] is incorrectly truncated"
                    )
                step_messages = []
                for field in ("prompt", "completion"):
                    value = step.get(field)
                    if isinstance(value, list):
                        step_messages.extend(
                            message for message in value if isinstance(message, dict)
                        )
                issues.extend(
                    validate_training_messages(
                        step_messages,
                        tools=tools,
                        row_path=f"{row_path}: trajectory[{idx}]",
                    )
                )
            if expected_indices and observed_indices != expected_indices:
                issues.append(
                    f"{row_path}: results exchange indices "
                    f"{sorted(observed_indices)} != successful LLM exchange indices "
                    f"{sorted(expected_indices)}"
                )
        token_usage = row.get("token_usage")
        if not isinstance(token_usage, dict):
            issues.append(f"{row_path}: missing object token_usage")
        else:
            has_total = positive_number(token_usage.get("total_tokens"))
            has_input = positive_number(
                token_usage.get("final_input_tokens")
            ) or positive_number(token_usage.get("input_tokens"))
            has_output = positive_number(
                token_usage.get("final_output_tokens")
            ) or positive_number(token_usage.get("output_tokens"))
            if not (has_total or has_input):
                issues.append(f"{row_path}: missing positive input/total token usage")
            if not (has_total or has_output):
                issues.append(f"{row_path}: missing positive output/total token usage")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            issues.append(f"{row_path}: missing object metrics")
        elif not positive_number(metrics.get("n_tool_calls")) and tool_usage_count(
            result
        ):
            issues.append(f"{row_path}: metrics missing positive n_tool_calls")
        if row.get("reward") is None and row.get("score") is None:
            issues.append(f"{row_path}: missing reward/score")

    return issues


def validate_results_jsonl(
    root: Path,
    *,
    result: dict[str, Any],
    llm_summary: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    path = root / "results.jsonl"
    if not path.is_file():
        return [f"missing required artifact: {path}"], {}
    rows, row_issues = read_jsonl(path)
    issues = list(row_issues)
    if len(rows) != 1:
        issues.append(
            f"{path}: expected exactly one rollout results row, got {len(rows)}"
        )
    for idx, row in enumerate(rows, start=1):
        issues.extend(
            validate_results_row(
                row,
                row_path=f"{path}:{idx}",
                result=result,
                llm_summary=llm_summary,
            )
        )
    summary = {
        "rows": len(rows),
        "training_ready": sum(
            1
            for row in rows
            if isinstance(row.get("info"), dict)
            and row["info"].get("training_ready") is True
        ),
        "rows_with_tool_calls": sum(
            1
            for row in rows
            if any(
                isinstance(message, dict) and bool(message.get("tool_calls"))
                for message in normalize_training_messages(row, str(path), [])
            )
        ),
    }
    return issues, summary


def load_run_config(root: Path) -> dict[str, Any] | None:
    for name in ("run_config.json", "config.json", "metadata.json"):
        path = root / name
        if not path.is_file():
            continue
        data, _ = read_json(path)
        if data is not None:
            return data
    return None


def validate_rollout(
    root: Path, *, allow_oracle_without_llm: bool = False
) -> dict[str, Any]:
    result_path = root / "result.json"
    result, result_error = read_json(result_path)
    issues: list[str] = []
    warnings: list[str] = []
    if result_error:
        issues.append(result_error)
        result = {}

    run_config = load_run_config(root)
    oracle = is_oracle_result(result, run_config)

    acp_path = root / "trajectory" / "acp_trajectory.jsonl"
    llm_path = root / "trajectory" / "llm_trajectory.jsonl"
    artifact_summary: dict[str, Any] = {}
    llm_summary: dict[str, Any] | None = None

    if not acp_path.is_file():
        issues.append(f"missing required artifact: {acp_path}")
    else:
        acp_rows, acp_issues = read_jsonl(acp_path)
        issues.extend(acp_issues)
        issues.extend(validate_acp(acp_rows, acp_path))
        artifact_summary["acp_events"] = len(acp_rows)

    if oracle and allow_oracle_without_llm:
        warnings.append("oracle rollout: llm_trajectory requirement bypassed by flag")
    elif not llm_path.is_file():
        issues.append(f"missing required artifact: {llm_path}")
    else:
        llm_rows, llm_issues = read_jsonl(llm_path)
        issues.extend(llm_issues)
        llm_health_issues, llm_summary = validate_llm(llm_rows, llm_path)
        issues.extend(llm_health_issues)
        artifact_summary["llm_exchanges"] = len(llm_rows)
        artifact_summary["llm"] = llm_summary

    results_issues, results_summary = validate_results_jsonl(
        root,
        result=result,
        llm_summary=llm_summary,
    )
    issues.extend(results_issues)
    if results_summary:
        artifact_summary["results"] = results_summary

    tokens = numeric_token_total(result)
    if not tokens or tokens <= 0:
        issues.append("missing or zero token usage in result metadata")
    if not timing_present(result):
        issues.append("missing timing metadata")
    tools = tool_usage_count(result)
    if tools is None or tools <= 0:
        issues.append("missing or zero tool usage metadata")
    if not reward_present(result):
        issues.append("missing verifier reward/score")

    error_text = " ".join(
        str(result.get(key) or "")
        for key in (
            "error",
            "verifier_error",
            "error_category",
            "verifier_error_category",
        )
    ).lower()
    if any(marker in error_text for marker in INFRA_ERROR_MARKERS):
        issues.append("result carries infra/provider error markers")

    return {
        "root": str(root),
        "status": "healthy" if not issues else "unhealthy",
        "healthy": not issues,
        "issues": issues,
        "warnings": warnings,
        "result": {
            "task_name": result.get("task_name"),
            "agent": result.get("agent"),
            "model": result.get("model"),
            "tokens": tokens,
            "tool_calls": tools,
            "reward_present": reward_present(result),
            "oracle": oracle,
        },
        "artifacts": artifact_summary,
    }


def discover_rollouts(path: Path) -> list[Path]:
    if path.is_file() and path.name == "result.json":
        return [path.parent]
    if (path / "result.json").is_file():
        return [path]
    return sorted({candidate.parent for candidate in path.rglob("result.json")})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Rollout dir, result.json, or jobs root to validate.",
    )
    parser.add_argument(
        "--allow-oracle-without-llm",
        action="store_true",
        help="Treat oracle reward-only runs as out of scope for LLM trajectory capture.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a concise text report.",
    )
    args = parser.parse_args(argv)

    rollouts: list[Path] = []
    for path in args.paths:
        rollouts.extend(discover_rollouts(path))
    deduped = sorted({path.resolve(): path for path in rollouts}.values())

    reports = [
        validate_rollout(
            rollout,
            allow_oracle_without_llm=args.allow_oracle_without_llm,
        )
        for rollout in deduped
    ]
    summary = {
        "checked": len(reports),
        "healthy": sum(1 for report in reports if report["healthy"]),
        "unhealthy": sum(1 for report in reports if not report["healthy"]),
    }
    payload = {
        "healthy": summary["unhealthy"] == 0 and summary["checked"] > 0,
        "summary": summary,
        "rollouts": reports,
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"checked={summary['checked']} healthy={summary['healthy']} "
            f"unhealthy={summary['unhealthy']}"
        )
        for report in reports:
            marker = "OK" if report["healthy"] else "UNHEALTHY"
            print(f"{marker} {report['root']}")
            for issue in report["issues"]:
                print(f"  - {issue}")
            for warning in report["warnings"]:
                print(f"  warning: {warning}")

    return 0 if payload["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
