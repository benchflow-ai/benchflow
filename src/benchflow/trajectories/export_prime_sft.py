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
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from benchflow._utils.json_safe import dumps_finite, scrub_non_finite
from benchflow.trajectories.types import redact_trajectory_obj

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
    # Rollout-level: number of rollouts skipped because *every* captured exchange
    # was a provider error (non-200). Counts rollouts (+= 1), like the other
    # skipped_* fields, so the manifest stays comparable across granularities.
    skipped_provider_error: int = 0
    # Exchange-level companion: total non-200 exchanges across those skipped
    # rollouts. A single rollout with 20 failed calls adds 20 here but 1 above.
    skipped_exchanges_provider_error: int = 0
    skipped_no_assistant: int = 0
    skipped_missing_tool_defs: int = 0
    skipped_terminal_error: int = 0
    skipped_invalid: int = 0
    tool_call_ids_rewritten: int = 0
    tool_messages_merged: int = 0
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
            "skipped_exchanges_provider_error": self.skipped_exchanges_provider_error,
            "skipped_no_assistant": self.skipped_no_assistant,
            "skipped_missing_tool_defs": self.skipped_missing_tool_defs,
            "skipped_terminal_error": self.skipped_terminal_error,
            "skipped_invalid": self.skipped_invalid,
            "tool_call_ids_rewritten": self.tool_call_ids_rewritten,
            "tool_messages_merged": self.tool_messages_merged,
            "sources": self.sources,
        }


@dataclass(frozen=True)
class PrimeSftExchangeData:
    messages: list[dict[str, Any]]
    tool_defs: list[dict[str, Any]]


class PrimeSftTrajectoryJsonlError(ValueError):
    """Raised when an LLM trajectory JSONL file is not parseable."""


def _json_line(record: dict[str, Any], *, redact: bool = True) -> str:
    # Redact secrets in the record's string values BEFORE serializing so the
    # emitted SFT row is always valid JSON; redacting the serialized text could
    # split a backslash escape next to a secret and corrupt the line.
    clean = scrub_non_finite(record)
    if redact:
        clean = redact_trajectory_obj(clean)
    clean = _sanitize_prime_sft_row_tool_call_arguments(clean, redact=redact)
    return dumps_finite(clean, default=str)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_llm_trajectory_jsonl(
    path: Path,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        if strict:
            raise PrimeSftTrajectoryJsonlError(
                f"{path}: cannot read LLM trajectory JSONL: {exc}"
            ) from exc
        return records
    for line_num, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if strict:
                raise PrimeSftTrajectoryJsonlError(
                    f"{path}: line {line_num}: invalid JSON: {exc}"
                ) from exc
            continue
        if isinstance(record, dict):
            records.append(record)
        elif strict:
            raise PrimeSftTrajectoryJsonlError(
                f"{path}: line {line_num}: top-level record must be an object"
            )
    return records


def _iter_rollout_dirs(root: str | Path) -> list[Path]:
    path = Path(root)
    if (path / "result.json").is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted({p.parent for p in path.rglob("result.json")})


def _iter_selected_rollout_dirs(selection_path: str | Path) -> list[Path]:
    path = Path(selection_path)
    try:
        selection = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid canonical selection JSON: {path}: {exc}") from exc
    if not isinstance(selection, dict):
        raise ValueError(f"canonical selection must be an object: {path}")
    job_dir = Path(str(selection.get("job_dir") or ""))
    selected = selection.get("selected", selection.get("selection"))
    if not isinstance(selected, list):
        raise ValueError(
            f"canonical selection {path} must contain selected or selection list"
        )
    rollout_dirs = []
    for idx, row in enumerate(selected, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"canonical selection row {idx} must be an object")
        row = cast(dict[str, Any], row)
        raw_dir = row.get("rollout_dir")
        if not isinstance(raw_dir, str) or not raw_dir:
            raise ValueError(f"canonical selection row {idx} missing rollout_dir")
        rollout_dir = Path(raw_dir)
        if (
            not (rollout_dir / "result.json").is_file()
            and not rollout_dir.is_absolute()
        ):
            rollout_dir = job_dir / rollout_dir
        if not (rollout_dir / "result.json").is_file() and isinstance(
            row.get("result_json"), str
        ):
            result_json = Path(row["result_json"])
            if result_json.is_absolute() and not result_json.is_file():
                marker = f"/{path.parent.name}/"
                _, sep, suffix = str(result_json).partition(marker)
                if sep:
                    result_json = path.parent / suffix
            rollout_dir = result_json.parent
        if not (rollout_dir / "result.json").is_file():
            raise ValueError(f"selected rollout has no result.json: {rollout_dir}")
        rollout_dirs.append(rollout_dir)
    return rollout_dirs


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


def _json_tool_call_arguments(arguments: Any, *, redact: bool = True) -> str:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {"_malformed_json_arguments": arguments}
        else:
            if not isinstance(parsed, dict):
                parsed = {"_non_object_json_arguments": parsed}
            else:
                clean = redact_trajectory_obj(parsed) if redact else parsed
                if clean == parsed:
                    return arguments
                return dumps_finite(clean, sort_keys=False, default=str)
    elif isinstance(arguments, dict):
        parsed = arguments
    elif arguments is None:
        parsed = {}
    else:
        parsed = {"_non_object_arguments": arguments}
    clean = redact_trajectory_obj(parsed) if redact else parsed
    return dumps_finite(clean, sort_keys=False, default=str)


def _normalize_tool_call(
    call: dict[str, Any], index: int = 0, *, redact: bool = True
) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict):
        function = {}
    name = function.get("name") or call.get("name") or "tool"
    arguments = function.get("arguments", call.get("arguments", {}))
    return {
        "id": str(call.get("id") or call.get("tool_call_id") or f"call_{index:06d}"),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": _json_tool_call_arguments(arguments, redact=redact),
        },
    }


def _normalize_message(
    message: dict[str, Any], index: int, *, redact: bool = True
) -> dict[str, Any]:
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
                    redact=redact,
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
            _normalize_tool_call(call, i, redact=redact)
            for i, call in enumerate(tool_calls)
            if isinstance(call, dict)
        ]
    return out


def _normalize_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Prime-RL SFT allows a system message only at index 0 (see
    # validate_prime_sft_row). Any system message after the first position —
    # including a *second consecutive* leading system message — is remapped to
    # "user" so the whole row isn't silently dropped into skipped_invalid.
    normalized: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        out = dict(message)
        if out.get("role") == "system" and idx != 0:
            out["role"] = "user"
        normalized.append(out)
    return normalized


def prime_sft_last_user_training_window(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return a compact prompt/completion window anchored at the last user turn.

    OpenHands-style system prompts are large enough that a full conversation
    prefix can push the first trainable assistant token beyond an 8k SFT
    sequence. Prime-RL can then skip the row even though the JSONL is valid.
    Keeping the latest user instruction plus the following assistant/tool turns
    preserves the supervised action trace while moving trainable tokens into the
    loaded context window.
    """
    for idx in range(len(messages) - 2, -1, -1):
        message = messages[idx]
        if message.get("role") != "user":
            continue
        completion = messages[idx + 1 :]
        if any(item.get("role") == "assistant" for item in completion):
            return [message], completion
    return None


def _messages_from_chat_request(
    body: dict[str, Any], *, redact: bool = True
) -> list[dict[str, Any]]:
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
        normalized.append(_normalize_message(message, idx, redact=redact))
    return normalized


def _messages_from_responses_request(
    body: dict[str, Any], *, redact: bool = True
) -> list[dict[str, Any]]:
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
                messages.append(_normalize_message(item, idx, redact=redact))
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


def _assistant_from_anthropic_content(
    content: Any, *, redact: bool = True
) -> dict[str, Any] | None:
    """Build an assistant row from Anthropic ``/v1/messages`` content blocks.

    Anthropic responses carry a list of typed blocks: ``text`` blocks hold the
    visible reply and ``tool_use`` blocks hold tool calls. The previous fallback
    flattened the whole list to text, silently dropping the tool calls and
    turning a tool-using assistant turn into corrupted SFT data. Preserve
    ``tool_use`` blocks as OpenAI-shaped ``tool_calls`` instead. Returns ``None``
    when ``content`` is not a block list, so the caller can fall back to text.
    """
    if not isinstance(content, list):
        return None
    raw_tool_calls = [
        {
            "id": item.get("id"),
            "type": "function",
            "function": {
                "name": item.get("name"),
                "arguments": item.get("input", {}),
            },
        }
        for item in content
        if isinstance(item, dict) and item.get("type") == "tool_use"
    ]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": _content_to_text(content),
    }
    if raw_tool_calls:
        message["tool_calls"] = [
            _normalize_tool_call(call, i, redact=redact)
            for i, call in enumerate(raw_tool_calls)
        ]
    return message


def _assistant_from_chat_response(
    body: dict[str, Any], *, redact: bool = True
) -> dict[str, Any] | None:
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and isinstance(first.get("message"), dict):
            return _normalize_message(first["message"], 0, redact=redact)
    message = body.get("message")
    if isinstance(message, dict):
        return _normalize_message(message, 0, redact=redact)
    content = body.get("content")
    if content:
        assistant = _assistant_from_anthropic_content(content, redact=redact)
        if assistant is not None:
            return assistant
        return {"role": "assistant", "content": _content_to_text(content)}
    assistant = _assistant_from_responses_response(body, redact=redact)
    if assistant is not None:
        return assistant
    return None


def _assistant_from_responses_response(
    body: dict[str, Any], *, redact: bool = True
) -> dict[str, Any] | None:
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
            _normalize_tool_call(call, i, redact=redact)
            for i, call in enumerate(tool_calls)
        ]
    return message


def _exchange_to_messages_and_tools(
    exchange: dict[str, Any],
    *,
    redact: bool = True,
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
        messages = _messages_from_chat_request(request_body, redact=redact)
        assistant = _assistant_from_chat_response(response_body, redact=redact)
    else:
        messages = _messages_from_responses_request(request_body, redact=redact)
        assistant = _assistant_from_responses_response(response_body, redact=redact)

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


def _normalize_tools_for_validation(
    row: dict[str, Any], row_num: int
) -> list[Any] | None:
    tools = row.get("tool_defs", row.get("tools"))
    if tools is None:
        return None
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"row {row_num}: tool_defs/tools is not valid JSON: {exc}"
            ) from exc
    if not isinstance(tools, list):
        raise ValueError(f"row {row_num}: tool_defs/tools must be a list")
    return tools


def _tool_names_for_validation(tools: list[Any] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


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


def _sanitize_message_tool_call_arguments(
    messages: list[Any], *, redact: bool = True
) -> list[Any]:
    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        out = dict(message)
        tool_calls = out.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            out["tool_calls"] = [
                _normalize_tool_call(
                    cast(dict[str, Any], tool_call), idx, redact=redact
                )
                for idx, tool_call in enumerate(tool_calls)
                if isinstance(tool_call, dict)
            ]
        sanitized.append(out)
    return sanitized


def _sanitize_prime_sft_row_tool_call_arguments(
    row: dict[str, Any], *, redact: bool = True
) -> dict[str, Any]:
    out = dict(row)
    messages = out.get("messages")
    if isinstance(messages, list):
        out["messages"] = _sanitize_message_tool_call_arguments(messages, redact=redact)
    prompt = out.get("prompt")
    if isinstance(prompt, list):
        out["prompt"] = _sanitize_message_tool_call_arguments(prompt, redact=redact)
    completion = out.get("completion")
    if isinstance(completion, list):
        out["completion"] = _sanitize_message_tool_call_arguments(
            completion, redact=redact
        )
    return out


def _row_message_segments(
    row: dict[str, Any],
) -> list[tuple[Literal["messages", "prompt", "completion"], Any]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return [("messages", message) for message in messages]
    prompt = row.get("prompt")
    completion = row.get("completion")
    if isinstance(prompt, list) and isinstance(completion, list):
        return [("prompt", message) for message in prompt] + [
            ("completion", message) for message in completion
        ]
    return []


def _content_join(left: Any, right: Any) -> str:
    return "\n".join(
        part for part in (_content_to_text(left), _content_to_text(right)) if part
    )


def _align_legacy_tool_call_ids(
    segments: list[tuple[Literal["messages", "prompt", "completion"], Any]],
) -> tuple[
    list[tuple[Literal["messages", "prompt", "completion"], Any]], dict[str, int]
]:
    """Repair legacy BenchFlow rows whose provider call ids drifted.

    Some historical ``results.jsonl`` artifacts preserved assistant tool-call ids
    from one provider layer (``fc_*``) while the following tool messages used the
    OpenAI-compatible ``call_*`` ids that the runtime sent back on the next turn.
    Pair by message order and rewrite only when a pending assistant call exists;
    true orphan tool outputs still fail validation.
    """
    out: list[tuple[Literal["messages", "prompt", "completion"], Any]] = []
    pending: list[dict[str, Any]] = []
    stats = {"tool_call_ids_rewritten": 0, "tool_messages_merged": 0}

    for segment, raw_message in segments:
        message = deepcopy(raw_message)
        if not isinstance(message, dict):
            out.append((segment, message))
            continue

        tool_calls = message.get("tool_calls")
        if message.get("role") == "assistant" and isinstance(tool_calls, list):
            pending.extend(
                tool_call for tool_call in tool_calls if isinstance(tool_call, dict)
            )

        if message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            if (
                out
                and isinstance(out[-1][1], dict)
                and out[-1][1].get("role") == "tool"
                and out[-1][1].get("tool_call_id") == tool_call_id
            ):
                out[-1][1]["content"] = _content_join(
                    out[-1][1].get("content"),
                    message.get("content"),
                )
                stats["tool_messages_merged"] += 1
                continue

            match_index = next(
                (
                    idx
                    for idx, tool_call in enumerate(pending)
                    if tool_call.get("id") == tool_call_id
                ),
                None,
            )
            if match_index is not None:
                pending.pop(match_index)
            elif pending and tool_call_id:
                tool_call = pending.pop(0)
                if tool_call.get("id") != tool_call_id:
                    tool_call["id"] = tool_call_id
                    stats["tool_call_ids_rewritten"] += 1

        out.append((segment, message))

    return out, stats


def _canonicalize_existing_prime_sft_row(
    row: dict[str, Any],
    row_num: int,
    *,
    sanitize_tool_call_arguments: bool = False,
    redact: bool = True,
) -> tuple[dict[str, Any], dict[str, int]]:
    out = (
        _sanitize_prime_sft_row_tool_call_arguments(row, redact=redact)
        if sanitize_tool_call_arguments
        else dict(row)
    )
    segments = _row_message_segments(out)
    if not segments:
        validate_prime_sft_row(out, row_num)
        return out, {"tool_call_ids_rewritten": 0, "tool_messages_merged": 0}

    repaired, stats = _align_legacy_tool_call_ids(segments)
    segment_names = {segment for segment, _ in repaired}
    if segment_names == {"messages"}:
        out["messages"] = [message for _, message in repaired]
    else:
        out.pop("messages", None)
        out["prompt"] = [
            message for segment, message in repaired if segment == "prompt"
        ]
        out["completion"] = [
            message for segment, message in repaired if segment == "completion"
        ]
    validate_prime_sft_row(out, row_num)
    return out, stats


def validate_prime_sft_row(row: dict[str, Any], row_num: int = 1) -> None:
    leaked = sorted(BANNED_ROW_KEYS.intersection(row))
    if leaked:
        raise ValueError(
            f"row {row_num}: banned leakage keys present: {', '.join(leaked)}"
        )

    messages = _row_messages(row, row_num)
    tools = _normalize_tools_for_validation(row, row_num)
    known_tool_names = _tool_names_for_validation(tools)
    pending_tool_call_ids: set[str] = set()

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
        tool_calls = message.get("tool_calls")
        if tool_calls and role != "assistant":
            raise ValueError(
                f"row {row_num}: only assistant messages may contain tool_calls"
            )
        if role == "tool" and not message.get("tool_call_id"):
            raise ValueError(f"row {row_num}: tool message requires tool_call_id")
        if role == "tool" and message.get("tool_call_id") not in pending_tool_call_ids:
            raise ValueError(
                f"row {row_num}: tool message references unknown tool_call_id"
            )
        if role == "tool":
            pending_tool_call_ids.discard(cast(str, message.get("tool_call_id")))
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise ValueError(
                f"row {row_num}: messages[{idx}].tool_calls must be a list"
            )
        if isinstance(tool_calls, list):
            for tool_call_idx, tool_call in enumerate(tool_calls):
                prefix = f"row {row_num}: messages[{idx}].tool_calls[{tool_call_idx}]"
                if not isinstance(tool_call, dict):
                    raise ValueError(f"{prefix} must be object")
                tool_call = cast(dict[str, Any], tool_call)
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    raise ValueError(f"{prefix}.function must be object")
                tool_call_id = tool_call.get("id")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    raise ValueError(f"{prefix}.id must be a non-empty string")
                if tool_call.get("type") != "function":
                    raise ValueError(f"{prefix}.type must be 'function'")
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"{prefix}.function.name must be a non-empty string"
                    )
                if known_tool_names and name not in known_tool_names:
                    raise ValueError(
                        f"{prefix}.function.name {name!r} not found in tool_defs/tools"
                    )
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        parsed_arguments = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{prefix}.function.arguments is not valid JSON: {exc}"
                        ) from exc
                    if not isinstance(parsed_arguments, dict):
                        raise ValueError(
                            f"{prefix}.function.arguments must be a JSON object"
                        )
                elif not isinstance(arguments, dict):
                    raise ValueError(
                        f"{prefix}.function.arguments must be a JSON object or JSON-encoded object"
                    )
                pending_tool_call_ids.add(tool_call_id)

    if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
        raise ValueError(f"row {row_num}: no assistant message")

    typed_messages = [m for m in messages if isinstance(m, dict)]
    if _has_tool_calls(typed_messages) and not tools:
        raise ValueError(
            f"row {row_num}: assistant tool_calls require non-empty tool_defs/tools"
        )
    if tools is not None:
        for tool_idx, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise ValueError(f"row {row_num}: tool_defs[{tool_idx}] must be object")
            function = tool.get("function")
            name = (
                function.get("name") if isinstance(function, dict) else tool.get("name")
            )
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"row {row_num}: tool_defs[{tool_idx}] missing function name"
                )


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


def _iter_prime_sft_jsonl_rows(
    path: Path,
    *,
    sanitize_tool_call_arguments: bool = False,
    redact: bool = True,
) -> Iterator[tuple[int, dict[str, Any], dict[str, int]]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_num, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"row {row_num}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {row_num}: top-level row must be object")
            row, repair_stats = _canonicalize_existing_prime_sft_row(
                row,
                row_num,
                sanitize_tool_call_arguments=sanitize_tool_call_arguments,
                redact=redact,
            )
            yield row_num, row, repair_stats


def _row_reward(row: dict[str, Any]) -> float | None:
    reward = row.get("reward", row.get("score"))
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        return float(reward)
    return None


def _compact_existing_prime_sft_row(
    row: dict[str, Any],
    *,
    row_num: int,
    source: Path,
) -> dict[str, Any]:
    """Emit a compact trainer row from a validated existing JSONL row."""
    messages = [
        message for message in _row_messages(row, row_num) if isinstance(message, dict)
    ]
    if _is_benchflow_results_row(row):
        window = prime_sft_last_user_training_window(messages)
        if window is not None:
            prompt, completion = window
            messages = prompt + completion
    out: dict[str, Any] = {"messages": messages}

    tools = row.get("tool_defs", row.get("tools"))
    if tools is not None:
        out["tool_defs"] = tools

    reward = _row_reward(row)
    if reward is not None:
        out["reward"] = reward

    raw_info = row.get("info")
    info: dict[str, Any] = (
        cast(dict[str, Any], raw_info) if isinstance(raw_info, dict) else {}
    )
    task_id = (
        row.get("task_id")
        or row.get("task_name")
        or info.get("task_id")
        or info.get("task_name")
    )
    if task_id:
        out["task_id"] = str(task_id)
    task_name = row.get("task_name") or info.get("task_name") or task_id
    if task_name:
        out["task_name"] = str(task_name)
    if isinstance(row.get("model"), str):
        out["model"] = row["model"]
    elif isinstance(info.get("model"), str):
        out["model"] = info["model"]
    if isinstance(row.get("agent"), str):
        out["agent"] = row["agent"]
    elif isinstance(info.get("agent"), str):
        out["agent"] = info["agent"]
    if isinstance(row.get("example_id"), int):
        out["source_example_id"] = row["example_id"]
    if isinstance(info.get("rollout_dir"), str):
        out["source_rollout_dir"] = info["rollout_dir"]
    if row.get("chat_template_kwargs") is not None:
        out["chat_template_kwargs"] = row["chat_template_kwargs"]
    out["source_path"] = str(source)
    out["source_index"] = row_num - 1
    out["source_format"] = "benchflow-results-jsonl"
    return out


def _is_benchflow_results_row(row: dict[str, Any]) -> bool:
    info = row.get("info")
    if isinstance(info, dict) and info.get("source") == "benchflow":
        return True
    return any(
        key in row
        for key in (
            "trajectory",
            "stop_condition",
            "token_usage",
            "is_completed",
            "is_truncated",
        )
    )


def _benchflow_row_training_skip_reason(row: dict[str, Any]) -> str | None:
    if not _is_benchflow_results_row(row):
        return None
    info = row.get("info")
    if isinstance(info, dict) and info.get("training_ready") is False:
        return "not_training_ready"
    if row.get("error"):
        return "terminal_error"
    if row.get("is_truncated") is True:
        return "partial_trajectory"
    stop_condition = row.get("stop_condition")
    if isinstance(stop_condition, str) and stop_condition not in {
        "",
        "agent_completed",
    }:
        return stop_condition
    return None


def _existing_prime_sft_jsonl_stats(
    path: Path,
    *,
    min_reward: float | None,
    redact: bool = True,
) -> PrimeSftExportStats:
    stats = PrimeSftExportStats(sources=[str(path)])
    for row_num, row, repair_stats in _iter_prime_sft_jsonl_rows(
        path, sanitize_tool_call_arguments=True, redact=redact
    ):
        stats.tool_call_ids_rewritten += repair_stats["tool_call_ids_rewritten"]
        stats.tool_messages_merged += repair_stats["tool_messages_merged"]
        stats.rollouts_seen += 1
        if _benchflow_row_training_skip_reason(row) is not None:
            stats.skipped_terminal_error += 1
            continue
        reward = _row_reward(row)
        if min_reward is not None and (reward is None or reward < min_reward):
            stats.skipped_reward += 1
            continue
        messages = _row_messages(row, row_num)
        typed_messages = [m for m in messages if isinstance(m, dict)]
        if _has_tool_calls(typed_messages):
            stats.rows_with_tool_calls += 1
        stats.rows_written += 1
    return stats


def _copy_existing_prime_sft_jsonl(
    source: Path,
    out: Path,
    *,
    min_reward: float | None,
    redact: bool = True,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row_num, row, _ in _iter_prime_sft_jsonl_rows(
            source, sanitize_tool_call_arguments=True, redact=redact
        ):
            if _benchflow_row_training_skip_reason(row) is not None:
                continue
            if min_reward is not None:
                reward = _row_reward(row)
                if reward is None or reward < min_reward:
                    continue
            compact = _compact_existing_prime_sft_row(
                row,
                row_num=row_num,
                source=source,
            )
            validate_prime_sft_row(compact, row_num)
            handle.write(_json_line(compact, redact=redact) + "\n")


def normalize_prime_sft_exchange(
    exchange: dict[str, Any],
    *,
    redact: bool = True,
) -> tuple[PrimeSftExchangeData | None, str | None]:
    """Normalize one raw LLM exchange through the Prime-SFT validator path."""
    messages, tool_defs, skip_reason = _exchange_to_messages_and_tools(
        exchange, redact=redact
    )
    if skip_reason:
        return None, skip_reason
    repaired, _ = _align_legacy_tool_call_ids(
        [("messages", message) for message in messages]
    )
    messages = [message for _, message in repaired]
    if _has_tool_calls(messages) and not tool_defs:
        return None, "missing_tool_defs"
    try:
        validate_prime_sft_row({"messages": messages, "tool_defs": tool_defs}, 1)
    except ValueError as exc:
        return None, f"invalid_prime_sft_row: {exc}"
    return PrimeSftExchangeData(messages=messages, tool_defs=tool_defs), None


def _row_from_exchange(
    *,
    exchange: dict[str, Any],
    rollout_dir: Path,
    result: dict[str, Any] | None,
    reward: float | None,
    exchange_idx: int,
    redact: bool = True,
) -> tuple[dict[str, Any] | None, str | None]:
    normalized, skip_reason = normalize_prime_sft_exchange(exchange, redact=redact)
    if skip_reason:
        return None, skip_reason
    if normalized is None:
        return None, "invalid_prime_sft_row"
    messages = normalized.messages
    window = prime_sft_last_user_training_window(messages)
    if window is not None:
        prompt, completion = window
        messages = prompt + completion

    agent_result = result.get("agent_result") if isinstance(result, dict) else None
    row = {
        "messages": messages,
        "tool_defs": normalized.tool_defs,
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
    canonical_selection: str | Path | None = None,
    redact: bool = True,
) -> tuple[list[dict[str, Any]], PrimeSftExportStats]:
    stats = PrimeSftExportStats()
    rows: list[dict[str, Any]] = []

    rollout_dirs = (
        _iter_selected_rollout_dirs(canonical_selection)
        if canonical_selection is not None
        else _iter_rollout_dirs(jobs_dir)
    )
    for rollout_dir in rollout_dirs:
        stats.rollouts_seen += 1
        result = _load_json(rollout_dir / "result.json")
        if result is None:
            stats.skipped_no_result += 1
            continue
        if _result_training_skip_reason(result) is not None:
            stats.skipped_terminal_error += 1
            continue
        reward = _reward_from_result(result)
        if min_reward is not None and (reward is None or reward < min_reward):
            stats.skipped_reward += 1
            continue

        trajectory_path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
        exchanges = load_llm_trajectory_jsonl(trajectory_path, strict=True)
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
            stats.skipped_provider_error += 1
            stats.skipped_exchanges_provider_error += len(exchanges)
            continue

        candidates = successful if row_mode == "exchange" else [successful[-1]]
        for idx, exchange in candidates:
            row, skip_reason = _row_from_exchange(
                exchange=exchange,
                rollout_dir=rollout_dir,
                result=result,
                reward=reward,
                exchange_idx=idx,
                redact=redact,
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


def _result_training_skip_reason(result: dict[str, Any]) -> str | None:
    if result.get("error"):
        return "agent_error"
    if result.get("verifier_error"):
        return "verifier_error"
    if result.get("partial_trajectory") is True:
        return "partial_trajectory"
    return None


def export_prime_sft_jsonl(
    jobs_dir: str | Path,
    out: str | Path,
    *,
    min_reward: float | None = None,
    row_mode: PrimeSftRowMode = "rollout",
    expected_rows: int | None = None,
    manifest: str | Path | None = None,
    canonical_selection: str | Path | None = None,
    redact: bool = True,
) -> PrimeSftExportStats:
    """Export BenchFlow rollouts to a Prime-RL SFT JSONL file.

    When ``expected_rows`` is set, the row-count assertion fires *before* the
    output file (or manifest) is opened — so a mismatch raises ``ValueError`` and
    writes nothing, rather than leaving a partial file. Callers should not expect
    ``out`` to exist on failure.
    """
    source_path = Path(jobs_dir)
    out_path = Path(out)
    manifest_path = Path(manifest) if manifest is not None else None

    if source_path.is_file() and source_path.suffix == ".jsonl":
        if source_path.resolve() == out_path.resolve():
            raise ValueError("--out must differ from the source JSONL path")
        stats = _existing_prime_sft_jsonl_stats(
            source_path, min_reward=min_reward, redact=redact
        )
        if expected_rows is not None and stats.rows_written != expected_rows:
            raise ValueError(
                f"row count {stats.rows_written} != expected {expected_rows}"
            )
        _copy_existing_prime_sft_jsonl(
            source_path,
            out_path,
            min_reward=min_reward,
            redact=redact,
        )
        if manifest_path is not None:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n"
            )
        return stats

    rows, stats = convert_benchflow_rollouts_to_prime_sft_rows(
        jobs_dir,
        min_reward=min_reward,
        row_mode=row_mode,
        canonical_selection=canonical_selection,
        redact=redact,
    )
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"row count {len(rows)} != expected {expected_rows}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_line(row, redact=redact) + "\n")

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(stats.as_dict(), indent=2, sort_keys=True) + "\n"
        )

    return stats
