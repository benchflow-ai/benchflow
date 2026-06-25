"""Convert BenchFlow LLM trajectories into Prime-RL SFT JSONL.

Prime-RL's local SFT path consumes a Hugging Face-style dataset whose rows carry
OpenAI-compatible ``messages`` plus optional ``tool_defs`` / ``tools``. BenchFlow
already emits lower-level provider traffic as
``trajectory/llm_trajectory.jsonl``; this module reconstructs trainer rows from
those request/response exchanges without touching the existing Verifiers/ADP
exporters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from benchflow._utils.json_safe import dumps_finite, scrub_non_finite
from benchflow.trajectories.types import redact_trajectory_text

PrimeSftRowMode = Literal["rollout", "exchange"]

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
BANNED_ROW_KEYS = {
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


@dataclass
class PrimeSftExportStats:
    rollouts_seen: int = 0
    exchanges_seen: int = 0
    rows_written: int = 0
    rows_with_tool_calls: int = 0
    skipped_no_result: int = 0
    skipped_no_trajectory: int = 0
    skipped_reward: int = 0
    skipped_provider_error: int = 0
    skipped_no_assistant: int = 0
    skipped_missing_tool_defs: int = 0
    skipped_invalid: int = 0
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rollouts_seen": self.rollouts_seen,
            "exchanges_seen": self.exchanges_seen,
            "rows_written": self.rows_written,
            "rows_with_tool_calls": self.rows_with_tool_calls,
            "skipped_no_result": self.skipped_no_result,
            "skipped_no_trajectory": self.skipped_no_trajectory,
            "skipped_reward": self.skipped_reward,
            "skipped_provider_error": self.skipped_provider_error,
            "skipped_no_assistant": self.skipped_no_assistant,
            "skipped_missing_tool_defs": self.skipped_missing_tool_defs,
            "skipped_invalid": self.skipped_invalid,
            "sources": self.sources,
        }


def _json_line(record: dict[str, Any]) -> str:
    raw = dumps_finite(scrub_non_finite(record), default=str)
    return redact_trajectory_text(raw)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _iter_rollout_dirs(root: str | Path) -> list[Path]:
    path = Path(root)
    if (path / "result.json").is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted({p.parent for p in path.rglob("result.json")})


def _reward_from_result(result: dict[str, Any] | None) -> float | None:
    if not isinstance(result, dict):
        return None
    rewards = result.get("rewards")
    if isinstance(rewards, dict):
        reward = rewards.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            return float(reward)
    reward = result.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        return float(reward)
    return None


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _normalize_role(role: Any) -> str:
    if role == "developer":
        return "system"
    if role == "model":
        return "assistant"
    return str(role or "user")


def _normalize_tool_call(call: dict[str, Any], index: int = 0) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict):
        function = {}
    name = function.get("name") or call.get("name") or "tool"
    arguments = function.get("arguments", call.get("arguments", {}))
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, sort_keys=True)
    return {
        "id": str(call.get("id") or call.get("tool_call_id") or f"call_{index:06d}"),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": arguments,
        },
    }


def _normalize_message(message: dict[str, Any], index: int) -> dict[str, Any]:
    message_type = message.get("type")
    if message_type == "function_call":
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _normalize_tool_call(
                    {
                        "id": message.get("call_id") or message.get("id"),
                        "type": "function",
                        "function": {
                            "name": message.get("name"),
                            "arguments": message.get("arguments", {}),
                        },
                    },
                    index,
                )
            ],
        }
    if message_type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": str(message.get("call_id") or message.get("id") or ""),
            "content": _content_to_text(message.get("output")),
        }
    role = _normalize_role(message.get("role"))
    out: dict[str, Any] = {"role": role}
    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        if tool_call_id is not None:
            out["tool_call_id"] = str(tool_call_id)
    content = message.get("content")
    out["content"] = _content_to_text(content)
    tool_calls = message.get("tool_calls")
    if tool_calls is None and isinstance(message.get("function_call"), dict):
        tool_calls = [message["function_call"]]
    if isinstance(tool_calls, list) and tool_calls:
        out["tool_calls"] = [
            _normalize_tool_call(call, i)
            for i, call in enumerate(tool_calls)
            if isinstance(call, dict)
        ]
    return out


def _normalize_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_non_system = False
    for message in messages:
        out = dict(message)
        role = out.get("role")
        if role == "system" and seen_non_system:
            out["role"] = "user"
        elif role != "system":
            seen_non_system = True
        normalized.append(out)
    return normalized


def _messages_from_chat_request(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        message = cast(dict[str, Any], message)
        if message.get("type") == "reasoning":
            continue
        normalized.append(_normalize_message(message, idx))
    return normalized


def _messages_from_responses_request(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": _content_to_text(instructions)})
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for idx, item in enumerate(raw_input):
            if not isinstance(item, dict):
                continue
            item = cast(dict[str, Any], item)
            if item.get("type") != "reasoning":
                messages.append(_normalize_message(item, idx))
    return messages


def _tool_defs_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tools = body.get("tools") or body.get("tool_defs") or []
    if not isinstance(raw_tools, list):
        return []
    tools: list[dict[str, Any]] = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("function"), dict):
            function = dict(item["function"])
        else:
            function = {
                "name": item.get("name"),
                "description": item.get("description", ""),
                "parameters": item.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
        if not function.get("name"):
            continue
        function.setdefault("description", "")
        function.setdefault("parameters", {"type": "object", "properties": {}})
        tools.append({"type": "function", "function": function})
    return tools


def _assistant_from_chat_response(body: dict[str, Any]) -> dict[str, Any] | None:
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and isinstance(first.get("message"), dict):
            return _normalize_message(first["message"], 0)
    message = body.get("message")
    if isinstance(message, dict):
        return _normalize_message(message, 0)
    content = body.get("content")
    if content:
        return {"role": "assistant", "content": _content_to_text(content)}
    assistant = _assistant_from_responses_response(body)
    if assistant is not None:
        return assistant
    return None


def _assistant_from_responses_response(body: dict[str, Any]) -> dict[str, Any] | None:
    output = body.get("output")
    if not isinstance(output, list):
        return None
    texts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            texts.append(_content_to_text(item.get("content")))
        elif item_type in {"function_call", "tool_call"}:
            tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments", {}),
                    },
                }
            )
    if not texts and not tool_calls:
        return None
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(t for t in texts if t),
    }
    if tool_calls:
        message["tool_calls"] = [
            _normalize_tool_call(call, i) for i, call in enumerate(tool_calls)
        ]
    return message


def _exchange_to_messages_and_tools(
    exchange: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    request = (
        cast(dict[str, Any], exchange.get("request"))
        if isinstance(exchange.get("request"), dict)
        else {}
    )
    response = (
        cast(dict[str, Any], exchange.get("response"))
        if isinstance(exchange.get("response"), dict)
        else {}
    )
    request_body = (
        cast(dict[str, Any], request.get("body"))
        if isinstance(request.get("body"), dict)
        else {}
    )
    response_body = (
        cast(dict[str, Any], response.get("body"))
        if isinstance(response.get("body"), dict)
        else {}
    )

    if "messages" in request_body:
        messages = _messages_from_chat_request(request_body)
        assistant = _assistant_from_chat_response(response_body)
    else:
        messages = _messages_from_responses_request(request_body)
        assistant = _assistant_from_responses_response(response_body)

    if assistant is None:
        return [], [], "no_assistant"
    messages.append(assistant)
    return (
        _normalize_system_messages(messages),
        _tool_defs_from_body(request_body),
        None,
    )


def _has_tool_calls(messages: list[dict[str, Any]]) -> bool:
    return any(bool(message.get("tool_calls")) for message in messages)


def _row_messages(row: dict[str, Any], row_num: int) -> list[Any]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    prompt = row.get("prompt")
    completion = row.get("completion")
    if isinstance(prompt, list) and isinstance(completion, list):
        combined = prompt + completion
        if combined:
            return combined
    raise ValueError(
        f"row {row_num}: expected non-empty messages or prompt+completion lists"
    )


def validate_prime_sft_row(row: dict[str, Any], row_num: int = 1) -> None:
    leaked = sorted(BANNED_ROW_KEYS.intersection(row))
    if leaked:
        raise ValueError(
            f"row {row_num}: banned leakage keys present: {', '.join(leaked)}"
        )

    messages = _row_messages(row, row_num)

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"row {row_num}: messages[{idx}] must be object")
        message = cast(dict[str, Any], message)
        leaked_message = sorted(BANNED_MESSAGE_KEYS.intersection(message))
        if leaked_message:
            raise ValueError(
                f"row {row_num}: messages[{idx}] has banned keys: {', '.join(leaked_message)}"
            )
        role = message.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"row {row_num}: messages[{idx}].role invalid: {role!r}")
        if role == "system" and idx != 0:
            raise ValueError(
                f"row {row_num}: system message must be at index 0, got index {idx}"
            )
        if "content" not in message and "tool_calls" not in message:
            raise ValueError(
                f"row {row_num}: messages[{idx}] needs content or tool_calls"
            )
        if (
            "tool_calls" in message
            and message.get("tool_calls")
            and role != "assistant"
        ):
            raise ValueError(
                f"row {row_num}: only assistant messages may contain tool_calls"
            )
        if role == "tool" and not message.get("tool_call_id"):
            raise ValueError(f"row {row_num}: tool message requires tool_call_id")

    if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
        raise ValueError(f"row {row_num}: no assistant message")

    tools = row.get("tool_defs", row.get("tools"))
    if tools is not None:
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"row {row_num}: tool_defs/tools is not valid JSON: {exc}"
                ) from exc
        if not isinstance(tools, list):
            raise ValueError(f"row {row_num}: tool_defs/tools must be a list")
        for tool_idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f"row {row_num}: tool_defs[{tool_idx}] must be object")


def validate_prime_sft_jsonl(
    jsonl: str | Path,
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    path = Path(jsonl)
    rows = 0
    rows_with_tool_calls = 0
    with path.open("r", encoding="utf-8") as handle:
        for row_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"row {row_num}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {row_num}: top-level row must be object")
            validate_prime_sft_row(row, row_num)
            row_messages = _row_messages(row, row_num)
            typed_messages = [m for m in row_messages if isinstance(m, dict)]
            if _has_tool_calls(typed_messages):
                rows_with_tool_calls += 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"row count {rows} != expected {expected_rows}")
    return {"ok": True, "rows": rows, "rows_with_tool_calls": rows_with_tool_calls}


def _row_from_exchange(
    *,
    exchange: dict[str, Any],
    rollout_dir: Path,
    result: dict[str, Any] | None,
    reward: float | None,
    exchange_idx: int,
) -> tuple[dict[str, Any] | None, str | None]:
    messages, tool_defs, skip_reason = _exchange_to_messages_and_tools(exchange)
    if skip_reason:
        return None, skip_reason
    if _has_tool_calls(messages) and not tool_defs:
        return None, "missing_tool_defs"

    agent_result = result.get("agent_result") if isinstance(result, dict) else None
    row = {
        "messages": messages,
        "tool_defs": tool_defs,
        "task_name": (result or {}).get("task_name") or rollout_dir.name,
        "source": "benchflow-llm-trajectory",
        "source_path": str(rollout_dir / "trajectory" / "llm_trajectory.jsonl"),
        "exchange_index": exchange_idx,
        "reward": reward,
        "score": reward,
        "model": ((exchange.get("request") or {}).get("body") or {}).get("model"),
        "agent": (result or {}).get("agent"),
        "token_usage": agent_result if isinstance(agent_result, dict) else None,
    }
    return {key: value for key, value in row.items() if value is not None}, None


def convert_benchflow_rollouts_to_prime_sft_rows(
    jobs_dir: str | Path,
    *,
    min_reward: float | None = None,
    row_mode: PrimeSftRowMode = "rollout",
) -> tuple[list[dict[str, Any]], PrimeSftExportStats]:
    stats = PrimeSftExportStats()
    rows: list[dict[str, Any]] = []

    for rollout_dir in _iter_rollout_dirs(jobs_dir):
        stats.rollouts_seen += 1
        result = _load_json(rollout_dir / "result.json")
        if result is None:
            stats.skipped_no_result += 1
            continue
        reward = _reward_from_result(result)
        if min_reward is not None and (reward is None or reward < min_reward):
            stats.skipped_reward += 1
            continue

        trajectory_path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
        exchanges = _load_jsonl(trajectory_path)
        if not exchanges:
            stats.skipped_no_trajectory += 1
            continue
        stats.exchanges_seen += len(exchanges)
        successful = [
            (idx, ex)
            for idx, ex in enumerate(exchanges)
            if ((ex.get("response") or {}).get("status_code") == 200)
        ]
        if not successful:
            stats.skipped_provider_error += len(exchanges)
            continue

        candidates = successful if row_mode == "exchange" else [successful[-1]]
        for idx, exchange in candidates:
            row, skip_reason = _row_from_exchange(
                exchange=exchange,
                rollout_dir=rollout_dir,
                result=result,
                reward=reward,
                exchange_idx=idx,
            )
            if skip_reason == "no_assistant":
                stats.skipped_no_assistant += 1
                continue
            if skip_reason == "missing_tool_defs":
                stats.skipped_missing_tool_defs += 1
                continue
            if row is None:
                stats.skipped_invalid += 1
                continue
            try:
                validate_prime_sft_row(row, len(rows) + 1)
            except ValueError:
                stats.skipped_invalid += 1
                continue
            rows.append(row)
            stats.rows_written += 1
            if _has_tool_calls(row["messages"]):
                stats.rows_with_tool_calls += 1
            stats.sources.append(str(trajectory_path))

    return rows, stats


def export_prime_sft_jsonl(
    jobs_dir: str | Path,
    out: str | Path,
    *,
    min_reward: float | None = None,
    row_mode: PrimeSftRowMode = "rollout",
    expected_rows: int | None = None,
    manifest: str | Path | None = None,
) -> PrimeSftExportStats:
    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        jobs_dir,
        min_reward=min_reward,
        row_mode=row_mode,
    )
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"row count {len(rows)} != expected {expected_rows}")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_line(row) + "\n")

    manifest_path = Path(manifest) if manifest is not None else None
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n"
        )

    return stats
