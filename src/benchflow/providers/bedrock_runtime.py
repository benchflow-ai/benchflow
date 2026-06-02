"""Bedrock runtime and translation helpers.

These helpers intentionally stay pure and narrow:
- validate env and build a Bedrock Runtime client lazily
- translate Anthropic Messages requests to Bedrock Converse
- translate OpenAI Responses requests to Bedrock Converse
- normalize Bedrock responses and stream events back to those frontends

The supported surface is the subset BenchFlow needs for codex-acp and
claude-agent-acp: text, tool calling, and basic inference config.
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

BEDROCK_RUNTIME_SERVICE = "bedrock-runtime"
BEDROCK_OPTIONAL_DEPENDENCY_HINT = "uv sync --extra bedrock"
BEDROCK_CONNECT_TIMEOUT_SEC = 10
BEDROCK_READ_TIMEOUT_SEC = 300


class _FallbackBedrockConfig:
    def __init__(self, *, connect_timeout: int, read_timeout: int) -> None:
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout


def resolve_bedrock_region(env: dict[str, str]) -> str:
    """Return AWS region from env or raise."""
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError(
            "AWS_REGION or AWS_DEFAULT_REGION required for Bedrock runtime."
        )
    return region


def validate_bedrock_runtime_env(env: dict[str, str]) -> dict[str, str]:
    """Validate Bedrock runtime env and normalize region aliases."""
    token = env.get("AWS_BEARER_TOKEN_BEDROCK")
    if not token:
        raise ValueError("AWS_BEARER_TOKEN_BEDROCK required for Bedrock runtime.")
    region = resolve_bedrock_region(env)
    normalized = dict(env)
    normalized.setdefault("AWS_REGION", region)
    normalized.setdefault("AWS_DEFAULT_REGION", region)
    return normalized


def build_bedrock_client(
    env: dict[str, str],
    *,
    boto3_module: Any | None = None,
) -> Any:
    """Build a Bedrock Runtime client using lazy boto3 import."""
    normalized = validate_bedrock_runtime_env(env)
    if boto3_module is None:
        try:
            import boto3 as boto3_module  # ty: ignore[unresolved-import]
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise RuntimeError(
                "Bedrock support requires the optional 'bedrock' dependency group. "
                f"Install it with: {BEDROCK_OPTIONAL_DEPENDENCY_HINT}"
            ) from exc
    try:
        from botocore.config import Config
    except ImportError:  # pragma: no cover - botocore ships with boto3
        Config = _FallbackBedrockConfig
    config = Config(
        connect_timeout=BEDROCK_CONNECT_TIMEOUT_SEC,
        read_timeout=BEDROCK_READ_TIMEOUT_SEC,
    )
    return boto3_module.client(
        service_name=BEDROCK_RUNTIME_SERVICE,
        region_name=normalized["AWS_REGION"],
        config=config,
    )


def _anthropic_system_to_bedrock(system: Any) -> list[dict[str, str]]:
    if not system:
        return []
    if isinstance(system, str):
        return [{"text": system}]
    blocks = []
    for block in system:
        if block.get("type") != "text":
            raise ValueError(f"Unsupported Anthropic system block: {block!r}")
        blocks.append({"text": block["text"]})
    return blocks


_IMAGE_MEDIA_TYPE_TO_BEDROCK_FORMAT = {
    "image/gif": "gif",
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}


def _anthropic_image_block_to_bedrock(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source") or {}
    if source.get("type") != "base64":
        raise ValueError(f"Unsupported Anthropic image source: {source!r}")
    image_format = _IMAGE_MEDIA_TYPE_TO_BEDROCK_FORMAT.get(source.get("media_type"))
    if not image_format:
        raise ValueError(f"Unsupported Anthropic image media type: {source!r}")
    data = source.get("data")
    if not isinstance(data, str):
        raise ValueError(f"Unsupported Anthropic image data: {source!r}")
    return {
        "image": {
            "format": image_format,
            "source": {"bytes": base64.b64decode(data, validate=True)},
        }
    }


def _tool_result_content_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}]
    blocks = []
    for block in content:
        if isinstance(block, str):
            blocks.append({"text": block})
            continue
        block_type = block.get("type")
        if block_type == "image":
            blocks.append(_anthropic_image_block_to_bedrock(block))
            continue
        if block_type != "text":
            raise ValueError(f"Unsupported tool_result content block: {block!r}")
        blocks.append({"text": block["text"]})
    return blocks


def _anthropic_content_to_bedrock(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    blocks: list[dict[str, Any]] = []
    for block in content:
        block_type = block.get("type")
        if block_type == "text":
            blocks.append({"text": block["text"]})
        elif block_type == "image":
            blocks.append(_anthropic_image_block_to_bedrock(block))
        elif block_type == "tool_use":
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": block["id"],
                        "name": block["name"],
                        "input": block.get("input", {}),
                    }
                }
            )
        elif block_type == "tool_result":
            blocks.append(
                {
                    "toolResult": {
                        "toolUseId": block["tool_use_id"],
                        "content": _tool_result_content_blocks(block.get("content")),
                        "status": "error" if block.get("is_error") else "success",
                    }
                }
            )
        else:
            raise ValueError(f"Unsupported Anthropic content block: {block!r}")
    return blocks


def _responses_content_to_bedrock(role: str, content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    blocks: list[dict[str, Any]] = []
    for block in content:
        block_type = block.get("type")
        if block_type in {"input_text", "output_text", "text"}:
            blocks.append({"text": block["text"]})
        elif block_type == "function_call":
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": block["call_id"],
                        "name": block["name"],
                        "input": json.loads(block.get("arguments") or "{}"),
                    }
                }
            )
        elif block_type == "function_call_output":
            output = block.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output)
            blocks.append(
                {
                    "toolResult": {
                        "toolUseId": block["call_id"],
                        "content": [{"text": output}] if output else [],
                        "status": "success",
                    }
                }
            )
        else:
            raise ValueError(
                f"Unsupported OpenAI Responses content block for role {role!r}: {block!r}"
            )
    return blocks


_MODELS_WITHOUT_BEDROCK_SAMPLING_PARAMS = {"anthropic.claude-opus-4-7"}


def _bedrock_model_key(model: str | None) -> str:
    """Return the foundation model key from a Bedrock model/profile id."""
    if not model:
        return ""
    key = model.rsplit("/", 1)[-1].lower()
    for prefix in ("us.", "global."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _supports_bedrock_sampling_params(model: str | None) -> bool:
    return _bedrock_model_key(model) not in _MODELS_WITHOUT_BEDROCK_SAMPLING_PARAMS


def _inference_config_from_request(
    body: dict[str, Any], *, max_tokens_key: str
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if body.get(max_tokens_key) is not None:
        config["maxTokens"] = body[max_tokens_key]
    if _supports_bedrock_sampling_params(body.get("model")):
        if body.get("temperature") is not None:
            config["temperature"] = body["temperature"]
        if body.get("top_p") is not None:
            config["topP"] = body["top_p"]
    stop = body.get("stop_sequences")
    if stop is None:
        stop = body.get("stop")
    if stop:
        config["stopSequences"] = stop if isinstance(stop, list) else [stop]
    return config


def _tools_to_bedrock_tool_config(
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    *,
    schema_key: str,
) -> dict[str, Any] | None:
    if not tools:
        return None
    bedrock_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            function = tool["function"]
            name = function["name"]
            description = function.get("description", "")
            schema = function.get("parameters", {})
        else:
            name = tool["name"]
            description = tool.get("description", "")
            schema = tool.get(schema_key, {})
        bedrock_tools.append(
            {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": schema},
                }
            }
        )

    config: dict[str, Any] = {"tools": bedrock_tools}
    if not tool_choice:
        return config
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            config["toolChoice"] = {"auto": {}}
        elif tool_choice == "required":
            config["toolChoice"] = {"any": {}}
        elif tool_choice == "none":
            return None
        return config
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        config["toolChoice"] = {"auto": {}}
    elif choice_type == "any":
        config["toolChoice"] = {"any": {}}
    elif choice_type == "tool" or choice_type == "function":
        config["toolChoice"] = {"tool": {"name": tool_choice["name"]}}
    return config


# Bedrock Claude Opus/Sonnet/Haiku 4.8+ require the adaptive-thinking contract
# (thinking.type=adaptive + output_config.effort) and reject the legacy enabled form.
_BEDROCK_ADAPTIVE_THINKING_RE = re.compile(r"claude-(opus|sonnet|haiku)-4-(8|9|1\d)")


def _force_bedrock_adaptive_thinking_for_opus_4_8(payload: dict[str, Any]) -> None:
    """Force Bedrock Claude 4.8+ to adaptive thinking at MAX effort.

    On Docker, openhands→litellm reaches Bedrock through this host proxy via the
    Anthropic-Messages / OpenAI-Responses path, which otherwise drops the incoming
    thinking block (=> 0 extended thinking) — and Bedrock 4.8 rejects the legacy
    ``thinking.type=enabled`` regardless. Regex-gated on the resolved model id, so
    every other model's payload is byte-identical to before.
    """
    model = str(payload.get("modelId", "")).lower()
    if not _BEDROCK_ADAPTIVE_THINKING_RE.search(model):
        return
    fields = payload.setdefault("additionalModelRequestFields", {})
    fields["thinking"] = {"type": "adaptive"}
    fields["output_config"] = {"effort": "max"}


def anthropic_request_to_bedrock_converse(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request to Converse kwargs."""
    payload: dict[str, Any] = {
        "modelId": body["model"],
        "messages": [
            {
                "role": message["role"],
                "content": _anthropic_content_to_bedrock(message["content"]),
            }
            for message in body.get("messages", [])
        ],
    }
    system = _anthropic_system_to_bedrock(body.get("system"))
    if system:
        payload["system"] = system
    inference_config = _inference_config_from_request(body, max_tokens_key="max_tokens")
    if inference_config:
        payload["inferenceConfig"] = inference_config
    tool_config = _tools_to_bedrock_tool_config(
        body.get("tools"),
        body.get("tool_choice"),
        schema_key="input_schema",
    )
    if tool_config:
        payload["toolConfig"] = tool_config
    _force_bedrock_adaptive_thinking_for_opus_4_8(payload)
    return payload


def _responses_function_call_to_bedrock_block(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "toolUse": {
            "toolUseId": item["call_id"],
            "name": item["name"],
            "input": json.loads(item.get("arguments") or "{}"),
        }
    }


def _responses_function_call_output_to_bedrock_block(
    item: dict[str, Any],
) -> dict[str, Any]:
    output = item.get("output", "")
    if not isinstance(output, str):
        output = json.dumps(output)
    return {
        "toolResult": {
            "toolUseId": item["call_id"],
            "content": [{"text": output}] if output else [],
            "status": "success",
        }
    }


def openai_responses_request_to_bedrock_converse(
    body: dict[str, Any],
) -> dict[str, Any]:
    """Translate an OpenAI Responses request to Converse kwargs.

    OpenAI Responses ``input`` may contain message items with a role and content,
    plus top-level tool-flow items (``function_call``, ``function_call_output``,
    ``reasoning``) that do not carry a role. Top-level tool items are folded into
    adjacent assistant/user messages so Bedrock Converse sees a valid alternating
    transcript; reasoning items are dropped (Bedrock has no equivalent surface).
    """
    input_items = body.get("input", [])
    if isinstance(input_items, str):
        input_items = [
            {"role": "user", "content": [{"type": "input_text", "text": input_items}]}
        ]
    messages: list[dict[str, Any]] = []

    def _append(role: str, block: dict[str, Any]) -> None:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": role, "content": [block]})

    for item in input_items:
        item_type = item.get("type")
        if item_type == "function_call":
            _append("assistant", _responses_function_call_to_bedrock_block(item))
            continue
        if item_type == "function_call_output":
            _append("user", _responses_function_call_output_to_bedrock_block(item))
            continue
        if item_type == "reasoning":
            # Bedrock Converse has no input-side reasoning surface; drop it.
            continue
        if item_type is not None and item_type != "message":
            raise ValueError(
                f"Unsupported OpenAI Responses input item type: {item_type!r}"
            )
        role = item.get("role")
        if not isinstance(role, str):
            raise ValueError(
                f"OpenAI Responses message item missing string role: {item!r}"
            )
        content = item.get("content", [])
        for block in _responses_content_to_bedrock(role, content):
            _append(role, block)

    payload: dict[str, Any] = {
        "modelId": body["model"],
        "messages": messages,
    }
    instructions = body.get("instructions")
    if instructions:
        payload["system"] = [{"text": instructions}]
    inference_config = _inference_config_from_request(
        body,
        max_tokens_key="max_output_tokens",
    )
    if inference_config:
        payload["inferenceConfig"] = inference_config
    tool_config = _tools_to_bedrock_tool_config(
        body.get("tools"),
        body.get("tool_choice"),
        schema_key="parameters",
    )
    if tool_config:
        payload["toolConfig"] = tool_config
    _force_bedrock_adaptive_thinking_for_opus_4_8(payload)
    return payload


def _bedrock_content_to_anthropic(
    content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        if "text" in block:
            blocks.append({"type": "text", "text": block["text"]})
        elif "toolUse" in block:
            tool = block["toolUse"]
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool["toolUseId"],
                    "name": tool["name"],
                    "input": tool.get("input", {}),
                }
            )
        elif "reasoningContent" in block:
            continue
        else:
            raise ValueError(f"Unsupported Bedrock content block: {block!r}")
    return blocks


def bedrock_response_to_anthropic(
    response: dict[str, Any],
    *,
    model: str,
    message_id: str = "msg_bedrock",
) -> dict[str, Any]:
    """Normalize a Converse response to an Anthropic Messages response."""
    output_message = response["output"]["message"]
    usage = response.get("usage", {})
    return {
        "id": message_id,
        "type": "message",
        "role": output_message["role"],
        "model": model,
        "content": _bedrock_content_to_anthropic(output_message.get("content", [])),
        "stop_reason": response.get("stopReason"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        },
    }


def bedrock_response_to_openai_response(
    response: dict[str, Any],
    *,
    model: str,
    response_id: str = "resp_bedrock",
    created_at: int | None = None,
) -> dict[str, Any]:
    """Normalize a Converse response to an OpenAI Responses response."""
    output_message = response["output"]["message"]
    usage = response.get("usage", {})
    output_items: list[dict[str, Any]] = []
    output_text_parts: list[str] = []

    for idx, block in enumerate(output_message.get("content", [])):
        if "text" in block:
            text = block["text"]
            output_text_parts.append(text)
            output_items.append(
                {
                    "id": f"msg_{idx}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": text, "annotations": []}
                    ],
                }
            )
        elif "toolUse" in block:
            tool = block["toolUse"]
            output_items.append(
                {
                    "id": f"fc_{idx}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tool["toolUseId"],
                    "name": tool["name"],
                    "arguments": json.dumps(
                        tool.get("input", {}), separators=(",", ":")
                    ),
                }
            )
        elif "reasoningContent" in block:
            continue
        else:
            raise ValueError(f"Unsupported Bedrock content block: {block!r}")

    input_tokens = usage.get("inputTokens", 0)
    output_tokens = usage.get("outputTokens", 0)
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at or int(time.time()),
        "status": "completed",
        "model": model,
        "output": output_items,
        "output_text": "".join(output_text_parts),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.get("totalTokens", input_tokens + output_tokens),
        },
    }


def _sse(data: dict[str, Any], event: str | None = None) -> str:
    lines = []
    if event is not None:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data, separators=(',', ':'))}")
    return "\n".join(lines) + "\n\n"


def bedrock_stream_event_to_anthropic_sse(
    event: dict[str, Any],
    *,
    model: str,
    message_id: str = "msg_bedrock",
) -> list[str]:
    """Translate one ConverseStream event to Anthropic-style SSE frames."""
    if "messageStart" in event:
        return [
            _sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": event["messageStart"].get("role", "assistant"),
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
            )
        ]
    if "contentBlockStart" in event:
        start = event["contentBlockStart"]
        if "toolUse" in start.get("start", {}):
            tool = start["start"]["toolUse"]
            return [
                _sse(
                    {
                        "type": "content_block_start",
                        "index": start["contentBlockIndex"],
                        "content_block": {
                            "type": "tool_use",
                            "id": tool["toolUseId"],
                            "name": tool["name"],
                            "input": {},
                        },
                    }
                )
            ]
        return [
            _sse(
                {
                    "type": "content_block_start",
                    "index": start["contentBlockIndex"],
                    "content_block": {"type": "text", "text": ""},
                }
            )
        ]
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"]
        payload: dict[str, Any]
        if "text" in delta.get("delta", {}):
            payload = {
                "type": "content_block_delta",
                "index": delta["contentBlockIndex"],
                "delta": {
                    "type": "text_delta",
                    "text": delta["delta"]["text"],
                },
            }
        elif "reasoningContent" in delta.get("delta", {}):
            return []
        elif "toolUse" in delta.get("delta", {}):
            payload = {
                "type": "content_block_delta",
                "index": delta["contentBlockIndex"],
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": delta["delta"]["toolUse"]["input"],
                },
            }
        else:
            return []
        return [_sse(payload)]
    if "contentBlockStop" in event:
        return [
            _sse(
                {
                    "type": "content_block_stop",
                    "index": event["contentBlockStop"]["contentBlockIndex"],
                }
            )
        ]
    if "messageStop" in event:
        stop_reason = event["messageStop"].get("stopReason")
        return [
            _sse({"type": "message_delta", "delta": {"stop_reason": stop_reason}}),
            _sse({"type": "message_stop"}),
        ]
    if "metadata" in event:
        usage = event["metadata"].get("usage", {})
        return [
            _sse(
                {
                    "type": "message_delta",
                    "usage": {
                        "input_tokens": usage.get("inputTokens", 0),
                        "output_tokens": usage.get("outputTokens", 0),
                    },
                }
            )
        ]
    return []


def bedrock_stream_event_to_openai_response_sse(
    event: dict[str, Any],
    *,
    model: str,
    response_id: str = "resp_bedrock",
    block_types: dict[int, str] | None = None,
) -> list[str]:
    """Translate one ConverseStream event to OpenAI Responses SSE frames."""
    if "messageStart" in event:
        return [
            _sse(
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": int(time.time()),
                        "status": "in_progress",
                        "model": model,
                    },
                }
            )
        ]
    if "contentBlockStart" in event:
        start = event["contentBlockStart"]
        index = start["contentBlockIndex"]
        if "toolUse" in start.get("start", {}):
            if block_types is not None:
                block_types[index] = "function_call"
            tool = start["start"]["toolUse"]
            item = {
                "id": f"fc_{index}",
                "type": "function_call",
                "status": "in_progress",
                "call_id": tool["toolUseId"],
                "name": tool["name"],
                "arguments": "",
            }
        else:
            if block_types is not None:
                block_types[index] = "text"
            item = {
                "id": f"msg_{index}",
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
        return [
            _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": item,
                }
            )
        ]
    if "contentBlockDelta" in event:
        delta = event["contentBlockDelta"]
        index = delta["contentBlockIndex"]
        if "text" in delta.get("delta", {}):
            return [
                _sse(
                    {
                        "type": "response.output_text.delta",
                        "output_index": index,
                        "delta": delta["delta"]["text"],
                    }
                )
            ]
        if "reasoningContent" in delta.get("delta", {}):
            return []
        return [
            _sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": index,
                    "delta": delta["delta"]["toolUse"]["input"],
                }
            )
        ]
    if "contentBlockStop" in event:
        index = event["contentBlockStop"]["contentBlockIndex"]
        block_type = "text"
        if block_types is not None:
            block_type = block_types.pop(index, "text")
        return [
            _sse(
                {
                    "type": (
                        "response.function_call_arguments.done"
                        if block_type == "function_call"
                        else "response.output_text.done"
                    ),
                    "output_index": index,
                }
            ),
            _sse({"type": "response.output_item.done", "output_index": index}),
        ]
    if "messageStop" in event:
        return []
    if "metadata" in event:
        usage = event["metadata"].get("usage", {})
        return [
            _sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "status": "completed",
                        "model": model,
                        "usage": {
                            "input_tokens": usage.get("inputTokens", 0),
                            "output_tokens": usage.get("outputTokens", 0),
                            "total_tokens": usage.get(
                                "totalTokens",
                                usage.get("inputTokens", 0)
                                + usage.get("outputTokens", 0),
                            ),
                        },
                    },
                }
            )
        ]
    return []
