"""Decode captured LLM exchanges into validated message records."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

from benchflow._utils.json_safe import dumps_finite
from benchflow.trajectories.types import redact_trajectory_obj

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


@dataclass(frozen=True)
class NormalizedExchange:
    messages: list[dict[str, Any]]
    tool_defs: list[dict[str, Any]]


class TrajectoryJsonlError(ValueError):
    """Raised when an LLM trajectory JSONL file is not parseable."""


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
            raise TrajectoryJsonlError(
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
                raise TrajectoryJsonlError(
                    f"{path}: line {line_num}: invalid JSON: {exc}"
                ) from exc
            continue
        if isinstance(record, dict):
            records.append(record)
        elif strict:
            raise TrajectoryJsonlError(
                f"{path}: line {line_num}: top-level record must be an object"
            )
    return records


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
                        "name": message.get("name"),
                        "arguments": message.get("arguments", {}),
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
    # The message contract allows a system message only at index 0 (see
    # validate_message_record). Any system message after the first position —
    # including a *second consecutive* leading system message — is remapped to
    # "user" so the whole row isn't silently dropped into skipped_invalid.
    normalized: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        out = dict(message)
        if out.get("role") == "system" and idx != 0:
            out["role"] = "user"
        normalized.append(out)
    return normalized


def last_user_training_window(
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
        messages.extend(
            _messages_from_chat_request({"messages": raw_input}, redact=redact)
        )
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
            "name": item.get("name"),
            "arguments": item.get("input", {}),
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
                    "name": item.get("name"),
                    "arguments": item.get("arguments", {}),
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


def normalize_provider_exchange(exchange: dict[str, Any]) -> dict[str, Any]:
    """Decode LiteLLM's Gemini passthrough envelope without changing raw evidence."""
    metadata = exchange.get("metadata") or {}
    if (
        not isinstance(metadata, dict)
        or metadata.get("call_type") != "pass_through_endpoint"
        or metadata.get("training_input_format") == "gemini"
    ):
        return exchange
    request = exchange.get("request") or {}
    if not isinstance(request, dict):
        return exchange
    body = request.get("body") or {}
    if not isinstance(body, dict):
        return exchange

    def invalid(detail: str) -> NoReturn:
        raise ValueError(f"Unsupported Gemini passthrough: {detail}")

    try:
        envelope = body["messages"]
        if not isinstance(envelope, list) or len(envelope) != 1:
            raise ValueError("expected singleton envelope")
        content = envelope[0]["content"]
        if not isinstance(content, str):
            raise ValueError("expected JSON text")
        native = json.loads(content)
        if not isinstance(native, dict) or "contents" not in native:
            raise ValueError("expected native contents")
    except (KeyError, TypeError, ValueError):
        model = (
            metadata.get("provider_model")
            or metadata.get("request_model")
            or body.get("model")
            or ""
        )
        if str(model).rsplit("/", 1)[-1].startswith(("gemini-", "gemma-")):
            invalid("malformed native request envelope")
        return exchange

    contents = native["contents"]
    if not isinstance(contents, list):
        invalid("contents must be a list")
    messages: list[dict[str, Any]] = []
    pending_calls: list[tuple[str | None, str, str]] = []
    explicit_ids: set[str] = set()
    for turn in contents:
        parts = turn.get("parts") if isinstance(turn, dict) else None
        for part in parts if isinstance(parts, list) else []:
            call = part.get("functionCall") if isinstance(part, dict) else None
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                signature = part.get("thoughtSignature")
                explicit_ids.add(
                    call["id"]
                    + (f"__thought__{signature}" if isinstance(signature, str) else "")
                )
    generated_id = 0
    system = native.get("systemInstruction")
    if system is not None:
        if not isinstance(system, dict):
            invalid("systemInstruction must be an object")
        contents = [{**system, "role": "system"}, *contents]
    for turn in contents:
        if not isinstance(turn, dict) or turn.get("role") not in {
            "system",
            "user",
            "model",
        }:
            invalid("unknown content role")
        parts = turn.get("parts")
        if not isinstance(parts, list):
            invalid("parts must be a list")
        role = "assistant" if turn["role"] == "model" else turn["role"]
        message: dict[str, Any] = {"role": role, "content": ""}
        for part in parts:
            if not isinstance(part, dict) or set(part) - {
                "text",
                "thought",
                "thoughtSignature",
                "functionCall",
                "functionResponse",
            }:
                invalid("unsupported content part")
            if len(set(part) & {"text", "functionCall", "functionResponse"}) != 1:
                invalid("content part must have exactly one payload")
            if "text" in part:
                if not isinstance(part["text"], str):
                    invalid("text must be a string")
                if not part.get("thought"):
                    message["content"] += part["text"]
            elif "functionCall" in part:
                call = part["functionCall"]
                if (
                    role != "assistant"
                    or not isinstance(call, dict)
                    or not isinstance(call.get("name"), str)
                    or not call["name"]
                    or not isinstance(call.get("args", {}), dict)
                ):
                    invalid("function call requires model role and name")
                call_id = call.get("id")
                if call_id is not None and (
                    not isinstance(call_id, str) or not call_id
                ):
                    invalid("function call id must be a non-empty string")
                signature = part.get("thoughtSignature")
                if signature is not None and not isinstance(signature, str):
                    invalid("thought signature must be a string")
                if call_id is None:
                    while True:
                        candidate = f"gemini_call_{generated_id}"
                        generated_id += 1
                        decorated = candidate + (
                            f"__thought__{signature}" if signature else ""
                        )
                        if (
                            candidate not in explicit_ids
                            and decorated not in explicit_ids
                        ):
                            break
                    normalized_id = candidate
                else:
                    normalized_id = call_id
                decorated = normalized_id + (
                    f"__thought__{signature}" if signature else ""
                )
                pending_calls.append((call_id, call["name"], decorated))
                message.setdefault("tool_calls", []).append(
                    {
                        "id": decorated,
                        "name": call["name"],
                        "arguments": call.get("args", {}),
                    }
                )
            elif "functionResponse" in part:
                response = part["functionResponse"]
                if (
                    role != "user"
                    or not isinstance(response, dict)
                    or not isinstance(response.get("name"), str)
                    or not response["name"]
                    or not isinstance(response.get("response"), dict)
                ):
                    invalid("function response requires user role and name")
                response_id = response.get("id")
                if response_id is not None and (
                    not isinstance(response_id, str) or not response_id
                ):
                    invalid("function response id must be a non-empty string")
                if message["content"]:
                    messages.append(message)
                    message = {"role": role, "content": ""}
                matches = [
                    index
                    for index, (source_id, name, _) in enumerate(pending_calls)
                    if (
                        source_id == response_id
                        if response_id is not None
                        else name == response["name"]
                    )
                ]
                if not matches:
                    invalid(
                        "function response references unknown call id"
                        if response_id is not None
                        else "function response references unknown call name"
                    )
                if len(matches) > 1:
                    invalid("function response is ambiguous")
                _, name, normalized_id = pending_calls.pop(matches[0])
                if response_id is not None and name != response["name"]:
                    invalid("function response name does not match call")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": normalized_id,
                        "content": json.dumps(response.get("response")),
                    }
                )
        if message["content"] or message.get("tool_calls"):
            messages.append(message)
    tools = []
    declarations = native.get("tools", [])
    if not isinstance(declarations, list):
        invalid("tools must be a list")
    for tool in declarations:
        if (
            not isinstance(tool, dict)
            or set(tool) != {"functionDeclarations"}
            or not isinstance(tool["functionDeclarations"], list)
        ):
            invalid("unsupported tool declaration")
        for declaration in tool["functionDeclarations"]:
            if not isinstance(declaration, dict) or not declaration.get("name"):
                invalid("function declaration requires a name")
            parameters = declaration.get(
                "parametersJsonSchema",
                declaration.get("parameters", {"type": "object", "properties": {}}),
            )
            if not isinstance(parameters, dict):
                invalid("function declaration requires an object schema")
            tools.append({**declaration, "parameters": parameters})
    return {
        **exchange,
        "metadata": {**metadata, "training_input_format": "gemini"},
        "request": {
            **request,
            "body": {
                **body,
                "messages": messages,
                "tools": _tool_defs_from_body({"tools": tools}),
            },
        },
    }


def _exchange_to_messages_and_tools(
    exchange: dict[str, Any],
    *,
    redact: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    try:
        exchange = normalize_provider_exchange(exchange)
    except ValueError as exc:
        return [], [], str(exc)
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


def validate_message_record(row: dict[str, Any], row_num: int = 1) -> None:
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


def normalize_exchange(
    exchange: dict[str, Any],
    *,
    redact: bool = True,
) -> tuple[NormalizedExchange | None, str | None]:
    """Normalize one raw LLM exchange through the shared message contract."""
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
        validate_message_record({"messages": messages, "tool_defs": tool_defs}, 1)
    except ValueError as exc:
        return None, f"invalid_prime_sft_row: {exc}"
    return NormalizedExchange(messages=messages, tool_defs=tool_defs), None
