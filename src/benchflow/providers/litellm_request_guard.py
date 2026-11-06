"""Single-route HTTP boundary, installed before LiteLLM parses client input.

LiteLLM's request dictionaries mix provider parameters and gateway controls.
Only protocol data crosses this boundary; destinations, authentication, router
settings and alternate pass-through APIs belong exclusively to the controller.
This file is copied into the proxy's private runtime, independent of BenchFlow.
"""

from __future__ import annotations

import base64
import contextvars
import json
import os
import re
import time
import uuid
from contextlib import suppress
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode

MAX_REQUEST_BYTES = 64 * 1024 * 1024
_REQUEST: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "benchflow_model_request", default=None
)


def _lifecycle(event, **fields):
    """Append bounded metadata only; never URLs, headers, bodies or exceptions.

    Best effort: the record is evidence about the request, so a failed write
    (a full disk) never fails or replaces the request's own outcome."""
    path = os.environ.get("BENCHFLOW_LITELLM_LOG_PATH")
    if not path:
        return
    try:
        record = {"event": event, "time": time.time(), **fields}
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (json.dumps(record) + "\n").encode())
        finally:
            os.close(fd)
    except Exception:
        pass


class _UpstreamAttempt:
    def __init__(self, request):
        self.request = _REQUEST.get()
        self.id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.status = None
        self.bytes = 0
        self.error_body = bytearray()
        self.encoding = ""
        self.ended = False
        if self.request is not None:
            import httpx

            try:
                request_bytes = len(request.content)
            except httpx.RequestNotRead:
                request_bytes = None
            state = self.request
            if state["failed_at"] is not None:
                state["retry_wait_seconds"] += self.started - state["failed_at"]
                state["failed_at"] = None
            _lifecycle(
                "upstream_start",
                request_id=state["id"],
                attempt_id=self.id,
                request_bytes=request_bytes,
            )

    def chunk(self, data):
        self.bytes += len(data)
        if self.request is not None and self.status is not None and self.status >= 500:
            # Validate the error body the client reads; never log its content.
            self.error_body.extend(data)

    def end(self):
        if self.ended or self.request is None:
            return
        self.ended = True
        elapsed = time.monotonic() - self.started
        invalid = False
        if self.status is not None and self.status >= 500:
            import httpx

            try:
                body = self.error_body
                if self.encoding:
                    body = httpx.Response(
                        self.status,
                        headers={"content-encoding": self.encoding},
                        content=bytes(body),
                    ).content
                json.loads(body)
            except (ValueError, UnicodeError, httpx.DecodingError):
                invalid = True
        if self.status is None or self.status >= 400:
            self.request["failed_at"] = self.started
        self.request["upstream_seconds"] += elapsed
        _lifecycle(
            "upstream_end",
            request_id=self.request["id"],
            attempt_id=self.id,
            status=self.status,
            response_bytes=self.bytes,
            upstream_seconds=elapsed,
            invalid_provider_response=invalid,
        )


def _track_response(httpx, response, attempt, *, asynchronous):
    attempt.status = response.status_code
    if response.is_stream_consumed:
        attempt.chunk(response.content)
        attempt.end()
        return response
    stream = response.stream
    attempt.encoding = response.headers.get("content-encoding", "")

    class SyncStream(httpx.SyncByteStream):
        def __iter__(self):
            try:
                for chunk in stream:
                    attempt.chunk(chunk)
                    yield chunk
            finally:
                attempt.end()

        def close(self):
            try:
                stream.close()
            finally:
                attempt.end()

    class AsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            try:
                async for chunk in stream:
                    attempt.chunk(chunk)
                    yield chunk
            finally:
                attempt.end()

        async def aclose(self):
            try:
                await stream.aclose()
            finally:
                attempt.end()

    response.stream = AsyncStream() if asynchronous else SyncStream()
    return response


# These are wire fields, not LiteLLM's much broader completion kwargs. A new
# provider control cannot become client-settable when LiteLLM adds a spelling.
_COMMON = frozenset(
    {
        "model",
        "stream",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "store",
        "service_tier",
    }
)
_CHAT = _COMMON | frozenset(
    {
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "stream_options",
        "n",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "response_format",
        "reasoning_effort",
        "thinking",
        "output_config",
        "logprobs",
        "top_logprobs",
        "prediction",
        "modalities",
        "audio",
        "verbosity",
        "functions",
        "function_call",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
    }
)
_RESPONSES = _COMMON | frozenset(
    {
        "input",
        "instructions",
        "max_output_tokens",
        "previous_response_id",
        "include",
        "reasoning",
        "text",
        "truncation",
        "background",
        "conversation",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "max_tool_calls",
        "context_management",
        "stream_options",
        # Codex tracing data is not LiteLLM's privileged `metadata` namespace.
        "client_metadata",
    }
)
_MESSAGES = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "system",
        "stream",
        "stop_sequences",
        "temperature",
        "top_p",
        "top_k",
        "tools",
        "tool_choice",
        "thinking",
        "output_config",
        "cache_control",
        "container",
        "context_management",
        "inference_geo",
        "service_tier",
        "metadata",
    }
)
_POST_FIELDS = {
    "/chat/completions": _CHAT,
    "/v1/chat/completions": _CHAT,
    "/responses": _RESPONSES,
    "/v1/responses": _RESPONSES,
    "/messages": _MESSAGES,
    "/v1/messages": _MESSAGES,
    "/v1/messages/count_tokens": _MESSAGES,
}
_GET_PATHS = frozenset(
    {"/health", "/health/liveliness", "/health/readiness", "/models", "/v1/models"}
)
_GEMINI_PATH = re.compile(
    r"/gemini/v1(?:beta)?/models/([a-zA-Z0-9._-]+):(generateContent|streamGenerateContent|countTokens)"
)
_GEMINI_FIELDS = frozenset(
    {
        "contents",
        "systemInstruction",
        "tools",
        "toolConfig",
        "safetySettings",
        "generationConfig",
        "cachedContent",
        "labels",
        "generateContentRequest",
    }
)
# Discard arbitrary forwarding and LiteLLM control headers before vendor code
# sees them. The gateway credential and documented protocol headers survive.
# x-openai-subagent is the Codex client's label for a sub-agent's request
# (review, compact, ...): protocol data that names no destination,
# credential or router setting, which proxy callbacks (a sub-agent model pin)
# read.
_HEADERS = frozenset(
    {
        b"authorization",
        b"x-api-key",
        b"x-goog-api-key",
        b"content-type",
        b"accept",
        b"user-agent",
        b"anthropic-version",
        b"anthropic-beta",
        b"x-goog-api-client",
        b"x-openai-subagent",
    }
)

# Responses input is a second protocol envelope, not arbitrary message data:
# some items load tools. Admit only inspected forms under web restrictions so
# a new input item cannot silently acquire provider-hosted capabilities.
_ITEM_FIELDS = {"id", "type", "status"}
_RESPONSE_INPUT_FIELDS = {
    "message": {"role", "content", "phase"},
    "function_call": {"name", "namespace", "call_id", "arguments"},
    "function_call_output": {"call_id", "output"},
    "custom_tool_call": {"name", "namespace", "call_id", "input"},
    "custom_tool_call_output": {"call_id", "output"},
    "computer_call": {"call_id", "action", "pending_safety_checks"},
    "computer_call_output": {"call_id", "output", "acknowledged_safety_checks"},
    "reasoning": {"summary", "content", "encrypted_content"},
    "compaction": {"encrypted_content"},
    "compaction_trigger": set(),
    "additional_tools": {"role", "tools"},
    "tool_search_call": {"call_id", "execution", "arguments"},
    "tool_search_output": {"call_id", "execution", "tools"},
}
_RESPONSE_CONTENT_FIELDS = {
    "input_text": {"text"},
    "output_text": {"text", "annotations", "logprobs"},
    "input_image": {"detail", "image_url"},
    "input_file": {"file_data", "filename"},
    "computer_screenshot": {"image_url"},
    "refusal": {"refusal"},
    "reasoning_text": {"text"},
    "summary_text": {"text"},
}


def _inline_bytes(value, *, data_url=False):
    # An allowlist of self-contained encodings, never a URL scheme denylist.
    # URI-list data URLs and opaque provider file handles are not inline bytes.
    if not isinstance(value, str):
        return False
    encoded = r"[A-Za-z0-9+/]+={0,2}"
    prefix = r"data:[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+;base64,"
    return re.fullmatch(prefix + encoded, value) is not None or (
        not data_url and re.fullmatch(encoded, value) is not None
    )


def _text_fields_are_data(obj, fields):
    # Type hints in provider translators are not runtime validation: a list
    # in a nominally textual slot can become multimodal message content.
    # Keep text scalar by construction, without inspecting URL spellings.
    return all(
        obj.get(field) is None or isinstance(obj[field], str) for field in fields
    )


def _response_content_is_data(content):
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            return False
        kind = part.get("type")
        if not isinstance(kind, str) or kind not in _RESPONSE_CONTENT_FIELDS:
            return False
        if not part.keys() <= (_RESPONSE_CONTENT_FIELDS[kind] | {"type"}):
            return False
        if not _text_fields_are_data(part, {"text", "refusal", "filename"}):
            return False
        if kind in {"input_image", "computer_screenshot"} and not _inline_bytes(
            part.get("image_url"), data_url=True
        ):
            return False
        if kind == "input_file" and not _inline_bytes(part.get("file_data")):
            return False
    return True


_CHAT_CONTENT_FIELDS = {
    "text": {"text", "cache_control"},
    "refusal": {"refusal"},
    "image_url": {"image_url"},
    "video_url": {"video_url"},
    "input_audio": {"input_audio"},
    "file": {"file"},
}
_MESSAGES_CONTENT_FIELDS = {
    "text": {"text", "cache_control", "citations"},
    "image": {"source", "cache_control"},
    "document": {"source", "cache_control", "title", "context", "citations"},
    "thinking": {"thinking", "signature"},
    "redacted_thinking": {"data"},
    "tool_use": {"id", "name", "input", "cache_control"},
    "tool_result": {"tool_use_id", "content", "is_error", "cache_control"},
    "search_result": {"source", "title", "content", "citations", "cache_control"},
}


def _message_content_is_data(content, *, messages):
    if content is None or isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    fields = _MESSAGES_CONTENT_FIELDS if messages else _CHAT_CONTENT_FIELDS
    for part in content:
        if not isinstance(part, dict):
            return False
        kind = part.get("type")
        if not isinstance(kind, str) or kind not in fields:
            return False
        if not part.keys() <= (fields[kind] | {"type"}):
            return False
        if not _text_fields_are_data(
            part, {"text", "refusal", "thinking", "signature", "title", "context"}
        ):
            return False
        if messages and kind in {"image", "document"}:
            source = part.get("source")
            if not isinstance(source, dict):
                return False
            if source.get("type") == "base64":
                if not source.keys() <= {
                    "type",
                    "media_type",
                    "data",
                } or not _inline_bytes(source.get("data")):
                    return False
            elif kind == "document" and source.get("type") == "text":
                if not source.keys() <= {
                    "type",
                    "media_type",
                    "data",
                } or not isinstance(source.get("data"), str):
                    return False
            elif kind == "document" and source.get("type") == "content":
                if not source.keys() <= {
                    "type",
                    "content",
                } or not _message_content_is_data(source.get("content"), messages=True):
                    return False
            else:
                return False
        if (
            messages
            and kind in {"tool_result", "search_result"}
            and not _message_content_is_data(part.get("content"), messages=True)
        ):
            return False
        if not messages and kind in {"image_url", "video_url"}:
            source = part.get(kind)
            if not isinstance(source, dict) or not source.keys() <= {"url", "detail"}:
                return False
            if not _inline_bytes(source.get("url"), data_url=True):
                return False
        if not messages and kind in {"file", "input_audio"}:
            source = part.get(kind)
            allowed = (
                {"file_data", "filename"} if kind == "file" else {"data", "format"}
            )
            field = "file_data" if kind == "file" else "data"
            if not isinstance(source, dict) or not source.keys() <= allowed:
                return False
            if not _inline_bytes(source.get(field)):
                return False
    return True


def _local_messages(body, *, messages):
    items = body.get("messages", [])
    if not isinstance(items, list):
        return False
    # Unknown envelope fields can introduce another content carrier in a
    # translator. Function arguments and tool schemas remain ordinary data.
    fields = {
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "function_call",
        "refusal",
        "annotations",
        "reasoning_content",
    }
    if messages:
        # The pinned Messages client attaches effort to system messages too.
        # Admit that scalar setting, not another opaque content carrier.
        fields.add("output_config")
    for item in items:
        if not isinstance(item, dict) or not item.keys() <= fields:
            return False
        if "output_config" in item:
            config = item["output_config"]
            if (
                not isinstance(config, dict)
                or not config.keys() <= {"effort"}
                or config.get("effort")
                not in (None, "low", "medium", "high", "xhigh", "max")
            ):
                return False
        if not _text_fields_are_data(
            item, {"role", "name", "tool_call_id", "refusal", "reasoning_content"}
        ):
            return False
        if not _message_content_is_data(item.get("content"), messages=messages):
            return False
    return not messages or _message_content_is_data(body.get("system"), messages=True)


def _gemini_parts_are_data(parts):
    if not isinstance(parts, list):
        return False
    for part in parts:
        if not isinstance(part, dict) or not part.keys() <= {
            "text",
            "inlineData",
            "inline_data",
            "functionCall",
            "function_call",
            "functionResponse",
            "function_response",
            "thought",
            "thoughtSignature",
            "thought_signature",
            "videoMetadata",
            "video_metadata",
        }:
            return False
        if not _text_fields_are_data(
            part, {"text", "thoughtSignature", "thought_signature"}
        ):
            return False
        for key in ("inlineData", "inline_data"):
            if key in part:
                source = part[key]
                if (
                    not isinstance(source, dict)
                    or not source.keys()
                    <= {
                        "data",
                        "mimeType",
                        "mime_type",
                        "displayName",
                        "display_name",
                    }
                    or not _inline_bytes(source.get("data"))
                ):
                    return False
        for key in ("functionResponse", "function_response"):
            if key in part:
                output = part[key]
                if not isinstance(output, dict) or not output.keys() <= {
                    "id",
                    "name",
                    "response",
                    "parts",
                    "willContinue",
                    "will_continue",
                    "scheduling",
                }:
                    return False
                if "parts" in output and not _gemini_parts_are_data(output["parts"]):
                    return False
    return True


def _local_gemini_content(body):
    # Cached content is expanded at the provider and cannot be inspected here.
    if body.get("cachedContent") is not None:
        return False
    contents = body.get("contents", [])
    if not isinstance(contents, list):
        return False
    if "systemInstruction" in body:
        contents = [*contents, body["systemInstruction"]]
    return all(
        isinstance(item, dict)
        and item.keys() <= {"role", "parts"}
        and _gemini_parts_are_data(item.get("parts", []))
        for item in contents
    )


def _local_response_input(body):
    if not _text_fields_are_data(body, {"instructions"}):
        return False
    # These references expand remotely into input or prompt configuration we
    # cannot inspect. Restricted callers must send explicit history instead.
    if any(
        body.get(key) is not None
        for key in ("previous_response_id", "conversation", "prompt")
    ):
        return False
    items = body.get("input", [])
    if isinstance(items, str):
        return True
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        kind = item.get("type", "message" if "role" in item else None)
        if not isinstance(kind, str) or kind not in _RESPONSE_INPUT_FIELDS:
            return False
        if not item.keys() <= (_RESPONSE_INPUT_FIELDS[kind] | _ITEM_FIELDS):
            return False
        if not _text_fields_are_data(
            item,
            {
                "id",
                "status",
                "role",
                "phase",
                "name",
                "namespace",
                "call_id",
                "arguments",
                "input",
                "encrypted_content",
            }
            - ({"arguments"} if kind == "tool_search_call" else set()),
        ):
            return False
        if kind in {"additional_tools", "tool_search_output"} and not _local_tools(
            item, native=False, messages=False
        ):
            return False
        if (
            kind in {"tool_search_call", "tool_search_output"}
            and item.get("execution") != "client"
        ):
            return False
        if kind in {"message", "reasoning"}:
            for field in ("content", "summary"):
                if field in item and not _response_content_is_data(item[field]):
                    return False
        if kind in {
            "function_call_output",
            "custom_tool_call_output",
        } and not _response_content_is_data(item.get("output")):
            return False
        if kind == "computer_call_output" and not _response_content_is_data(
            [item.get("output")]
        ):
            return False
    return True


def _local_tools(body, *, native, messages):
    """Under web restrictions only agent-executed tool declarations survive.

    Use protocol structure, not tool names or a growing search-name denylist:
    a function named web_search is local; a provider hosted search is not.
    Unknown hosted capabilities require an explicit policy decision.
    """
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            return False
        if "type" in tool and not isinstance(tool["type"], str):
            return False
        if native:
            if not tool.keys() <= {"functionDeclarations", "function_declarations"}:
                return False
        elif messages:
            kind = tool.get("type", "custom")
            if kind != "custom" and kind not in ANTHROPIC_CLIENT_TOOLS:
                return False
        elif tool.get("type") == "namespace":
            if not _local_tools(tool, native=False, messages=False):
                return False
        elif tool.get("type") == "tool_search":
            if tool.get("execution") != "client":
                return False
        elif tool.get("type") not in {
            "function",
            "custom",
            "computer",
            "computer_use_preview",
        }:
            return False
    # Chat's legacy search option also grants provider-side browsing.
    return "web_search_options" not in body


# The outbound boundary (PY8-ACP-C13). The checks above read what a client
# sent; LiteLLM then translates it, and a translator can turn a local-looking
# envelope into a provider-hosted capability (a function tool carrying
# googleSearch, legacy functions, a Messages tool named web_search promoted to
# web_search_options). Under web restrictions every upstream request is
# re-read after translation, at the HTTP client every LiteLLM route sends
# through, and admitted only when each wire field is known and every tool is
# a plain client-executed declaration. Anything else is refused, unknown keys
# included, so a new translator spelling cannot add a capability.
_WEB_TOOLS_DISABLED = "web tools are disabled for this run"
_OUT_CHAT = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "top_k",
        "n",
        "stop",
        "max_tokens",
        "max_completion_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "user",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "functions",
        "function_call",
        "reasoning_effort",
        "thinking",
        "prediction",
        "modalities",
        "audio",
        "verbosity",
        "store",
        "service_tier",
        "metadata",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        # Token capture (litellm_logging._apply_token_capture).
        "return_token_ids",
        "return_input_ids_in_sglext",
        "return_output_ids_in_sglext",
    }
)
_OUT_RESPONSES = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "top_logprobs",
        "max_output_tokens",
        "max_tool_calls",
        "reasoning",
        "text",
        "truncation",
        "store",
        "include",
        "user",
        "service_tier",
        "metadata",
        "client_metadata",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "context_management",
        "background",
    }
)
# Anthropic Messages, also the body of Bedrock invoke and Vertex rawPredict.
# `container` and `mcp_servers` attach provider-side execution and are absent.
_OUT_MESSAGES = frozenset(
    {
        "model",
        "messages",
        "system",
        "max_tokens",
        "metadata",
        "stop_sequences",
        "stream",
        "temperature",
        "top_p",
        "top_k",
        "tools",
        "tool_choice",
        "thinking",
        "output_config",
        "output_format",
        "cache_control",
        "context_management",
        "inference_geo",
        "service_tier",
        "anthropic_version",
        "anthropic_beta",
    }
)
_OUT_GEMINI = frozenset(
    {
        "contents",
        "systemInstruction",
        "system_instruction",
        "tools",
        "toolConfig",
        "tool_config",
        "safetySettings",
        "safety_settings",
        "generationConfig",
        "generation_config",
        "labels",
        "cachedContent",
        "cached_content",
        "generateContentRequest",
    }
)
# LiteLLM's context caching moves a request's tools into the cache it creates.
_OUT_GEMINI_CACHE = frozenset(
    {
        "model",
        "contents",
        "systemInstruction",
        "system_instruction",
        "tools",
        "toolConfig",
        "tool_config",
        "displayName",
        "display_name",
        "ttl",
        "expireTime",
        "expire_time",
    }
)
_OUT_CONVERSE = frozenset(
    {
        "messages",
        "system",
        "inferenceConfig",
        "toolConfig",
        "additionalModelRequestFields",
        "additionalModelResponseFieldPaths",
        "performanceConfig",
        "requestMetadata",
        "serviceTier",
    }
)
# Model-native passthrough inside a Converse body is itself a carrier.
_OUT_CONVERSE_EXTRA = frozenset(
    {
        "thinking",
        "anthropic_beta",
        "top_k",
        "reasoning_effort",
        "reasoningConfig",
        "output_config",
        "inferenceConfig",
        # Anthropic tools on Converse; checked like Messages tools below.
        "tools",
    }
)
# Anthropic tools the caller runs, by versioned type, with exactly the fields
# the pinned SDK defines. Server tools are absent; generated, never hand-edited.
# BEGIN generated by tools/anthropic_client_tools.py
# from anthropic==0.104.0; edit the generator, not this block.
ANTHROPIC_SDK_VERSION = "0.104.0"
ANTHROPIC_CUSTOM_TOOL = frozenset(
    [
        "allowed_callers",
        "cache_control",
        "defer_loading",
        "description",
        "eager_input_streaming",
        "input_examples",
        "input_schema",
        "name",
        "strict",
        "type",
    ]
)
ANTHROPIC_CLIENT_TOOLS = {
    "bash_20241022": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "bash_20250124": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "computer_20241022": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "display_height_px",
            "display_number",
            "display_width_px",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "computer_20250124": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "display_height_px",
            "display_number",
            "display_width_px",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "computer_20251124": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "display_height_px",
            "display_number",
            "display_width_px",
            "enable_zoom",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "memory_20250818": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "text_editor_20241022": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "text_editor_20250124": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "text_editor_20250429": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "input_examples",
            "name",
            "strict",
            "type",
        ]
    ),
    "text_editor_20250728": frozenset(
        [
            "allowed_callers",
            "cache_control",
            "defer_loading",
            "input_examples",
            "max_characters",
            "name",
            "strict",
            "type",
        ]
    ),
}
ANTHROPIC_CLIENT_CALLERS = frozenset(
    [
        "direct",
    ]
)
# END generated by tools/anthropic_client_tools.py
# Models whose provider searches or retrieves on its own, with no declared
# tool or request field to refuse (PY8-ACP-C14). Matched on the parsed
# model identity (model_identity), never on a spelling: "model" entries
# match the base model, "provider" entries the provider or a vendor path
# segment. Each entry: (kind, pattern, why, examples the tests refuse).
HOSTED_RETRIEVAL_MODELS = (
    (
        "model",
        r"gpt-4o(?:-mini)?-search-preview(?:-\d{4}-\d{2}-\d{2})?",
        "OpenAI search models search the web before every answer",
        ("openai/gpt-4o-search-preview", "gpt-4o-mini-search-preview-2025-03-11"),
    ),
    (
        "model",
        r"gpt-5-search-api(?:-\d{4}-\d{2}-\d{2})?",
        "OpenAI search models search the web before every answer",
        ("openai/gpt-5-search-api", "azure/gpt-5-search-api-2025-10-14"),
    ),
    (
        "model",
        r"o[34](?:-mini)?-deep-research(?:-\d{4}-\d{2}-\d{2})?",
        "OpenAI deep research models browse the web themselves",
        ("openai/o3-deep-research", "azure/o4-mini-deep-research-2025-06-26"),
    ),
    (
        # Google's deep-research-*, and Perplexity's deep research models
        # other providers serve under their own names (LiteLLM 1.103's
        # empiriolabs/perplexity-deep-research): whatever the vendor calls one.
        "model",
        r"(?:.*-)?deep-research(?:-.*)?",
        "Deep research models browse the web themselves",
        (
            "gemini/deep-research-pro-preview-12-2025",
            "empiriolabs/perplexity-deep-research",
            "empiriolabs/perplexity-advanced-deep-research",
        ),
    ),
    (
        "provider",
        r"perplexity",
        "Perplexity's API answers from live web search",
        (
            "perplexity/sonar",
            "perplexity/openai/gpt-5.1",
            "perplexity/llama-3.1-8b-instruct",
        ),
    ),
    (
        "model",
        r"(?:.*-)?sonar(?:-.*)?|pplx-.*-online",
        "Perplexity Sonar models answer from live web search",
        ("openrouter/perplexity/sonar-pro", "vercel_ai_gateway/perplexity/sonar"),
    ),
    (
        "model",
        r"compound(?:-beta)?(?:-mini)?",
        "Groq Compound systems run web search and code execution",
        ("groq/compound", "groq/compound-beta-mini"),
    ),
    (
        # A model OpenRouter itself serves (vendor openrouter, or a bare
        # openrouter/<name>) is a router: auto, fusion, fusion-flash.
        "router",
        "openrouter",
        "OpenRouter's own routers choose models and tools, web search "
        "included, per request",
        (
            "openrouter/openrouter/auto",
            "openrouter/openrouter/fusion",
            "openrouter/fusion-flash",
            "openai/openrouter/fusion",
        ),
    ),
)
# Routing variants a model id may carry as ":name" suffixes (OpenRouter's
# model variants). A value is why the variant itself reaches the web; None
# means it only changes routing or pricing.
MODEL_VARIANTS = {
    "free": None,
    "extended": None,
    "exacto": None,
    "thinking": None,
    "nitro": None,
    "floor": None,
    "beta": None,
    "online": "OpenRouter's :online variant adds web search",
}
# Providers whose model ids hold ":" themselves (versions, ARNs, tags).
_COLON_IN_MODEL_ID = frozenset(
    {
        "bedrock",
        "bedrock_converse",
        "sagemaker",
        "sagemaker_chat",
        "ollama",
        "ollama_chat",
    }
)


def split_provider(model):
    """(provider, rest) from pinned LiteLLM's provider registry, statically.

    Not get_llm_provider: that also runs provider hooks, and for
    github_copilot it starts a GitHub device login over the network and
    blocks. A known "<provider>/" prefix, an OpenAI fine-tune, or a bare
    name in LiteLLM's model lists decide; anything else has no provider.
    """
    try:
        import litellm

        providers = {str(getattr(p, "value", p)) for p in litellm.provider_list}
        by_provider = litellm.models_by_provider
    except Exception:
        return None, model
    head, sep, rest = model.partition("/")
    if sep and head in providers:
        return head, rest
    if model.startswith("ft:"):
        return "openai", model
    for provider, names in by_provider.items():
        if model in names:
            return provider, model
    return None, model


def model_identity(model, provider=None):
    """(provider, vendor path, base model, variants) of one model id.

    The provider prefix is split against pinned LiteLLM's provider registry
    (split_provider); a caller that knows the wire protocol passes its
    provider instead.
    """
    text = unquote(model).strip().removeprefix("models/")
    rest = text
    if provider is None:
        provider, rest = split_provider(text)
    provider = (provider or "").lower()
    if provider in _COLON_IN_MODEL_ID or rest.lower().startswith(
        ("arn:", "invoke/arn:")
    ):
        return provider, (), rest.lower(), ()
    *vendors, last = rest.lower().split("/")
    if last.startswith("ft:"):
        # OpenAI fine-tune ft:<base>:<org>:<name>:<id>: classified by its base.
        return provider, tuple(vendors), last.split(":")[1], ()
    base, *variants = last.split(":")
    return provider, tuple(vendors), base, tuple(variants)


def hosted_retrieval_model(model, provider=None):
    """Why ``model`` may not serve a restricted run, or None."""
    if not isinstance(model, str) or not model:
        return None
    if "@preset/" in unquote(model).lower():
        return (
            f"model {model} names an OpenRouter preset, which chooses its model and "
            "tools outside the request"
        )
    provider, vendors, base, variants = model_identity(model, provider)
    # Unknown variants are decided at launch (restricted_models: refused
    # unless the operator admits the configured model); this second check
    # matches only what is known to reach the web.
    for variant in variants:
        if MODEL_VARIANTS.get(variant):
            return (
                f"model {model} reaches the web by itself ({MODEL_VARIANTS[variant]})"
            )
    for kind, pattern, why, _examples in HOSTED_RETRIEVAL_MODELS:
        if (
            (kind == "provider" and pattern in (provider, *vendors))
            or (kind == "model" and re.fullmatch(pattern, base))
            or (
                kind == "router"
                and (pattern in vendors or (provider == pattern and not vendors))
            )
        ):
            return f"model {model} reaches the web by itself ({why})"
    return None


# Bedrock Invoke bodies are the model family's own protocol. The family is
# the one LiteLLM translated for: decided from the configured model.
_INVOKE_PROMPT_FIELDS = {
    "amazon": frozenset({"inputText", "textGenerationConfig"}),
    "ai21": frozenset(
        {
            "prompt",
            "maxTokens",
            "temperature",
            "topP",
            "top_k",
            "stopSequences",
            "countPenalty",
            "presencePenalty",
            "frequencyPenalty",
            "numResults",
        }
    ),
    "mistral": frozenset(
        {"prompt", "max_tokens", "temperature", "top_p", "top_k", "stop"}
    ),
    **dict.fromkeys(
        ("meta", "llama", "deepseek_r1", "qwen2", "qwen3"),
        frozenset({"prompt", "max_gen_len", "temperature", "top_p", "top_k", "stop"}),
    ),
}
_INVOKE_COHERE = frozenset(
    {
        "prompt",
        "message",
        "chat_history",
        "preamble",
        "max_tokens",
        "temperature",
        "p",
        "k",
        "top_k",
        "stop_sequences",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "tools",
    }
)
_OUT_NOVA = frozenset(
    {
        "messages",
        "system",
        "inferenceConfig",
        "toolConfig",
        "additionalModelRequestFields",
        "schemaVersion",
    }
)


def _invoke_family(path):
    """The Bedrock Invoke family LiteLLM translated this request for."""
    try:
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
    except Exception:
        return None
    configured = os.environ.get("BENCHFLOW_LITELLM_UPSTREAM_MODEL", "")
    candidates = (
        [configured.removeprefix("bedrock/")]
        if configured.startswith("bedrock/")
        else []
    )
    match = re.fullmatch(r".*/model/([^/]+)/[a-z-]+", path)
    if match:
        candidates.append(unquote(match[1]))
    for model in candidates:
        family = BaseAWSLLM.get_bedrock_invoke_provider(model)
        if family is not None:
            return family
    return None


_OUT_PATHS = (
    (re.compile(r".*/chat/completions"), "chat"),
    (re.compile(r".*/responses"), "responses"),
    (re.compile(r".*/messages(?:/count_tokens)?"), "messages"),
    (re.compile(r".*:(?:raw|streamRaw)Predict"), "messages"),
    (re.compile(r".*/model/[^/]+/(?:invoke|invoke-with-response-stream)"), "invoke"),
    (re.compile(r".*/model/[^/]+/(?:converse|converse-stream)"), "converse"),
    (re.compile(r".*/model/[^/]+/count-tokens"), "bedrock_count"),
    (
        re.compile(
            r".*/models/[^/:]+:(?:generateContent|streamGenerateContent|countTokens)"
        ),
        "gemini",
    ),
    (re.compile(r".*/cachedContents"), "gemini_cache"),
)
# Gemini caches created through this boundary, by provider resource name.
_OWN_CACHES: set[str] = set()
_WRAPPERS: set[object] = set()


def _unknown(obj, allowed, where):
    if not isinstance(obj, dict):
        return f"{where} is not an object"
    extra = sorted(obj.keys() - allowed)
    return f"unsupported {where} field(s) " + ", ".join(extra) if extra else None


def _openai_tools_refusal(tools, *, chat):
    if not isinstance(tools, list):
        return "tools is not a list"
    for tool in tools:
        kind = tool.get("type") if isinstance(tool, dict) else None
        if chat and kind in {"function", "custom"}:
            refusal = _unknown(tool, {"type", kind}, f"{kind} tool") or _unknown(
                tool.get(kind),
                {"name", "description", "parameters", "strict"}
                if kind == "function"
                else {"name", "description", "format"},
                f"{kind} declaration",
            )
        elif not chat and kind == "function":
            refusal = _unknown(
                tool,
                {
                    "type",
                    "name",
                    "description",
                    "parameters",
                    "strict",
                    "defer_loading",
                },
                "function tool",
            )
        elif not chat and kind == "custom":
            refusal = _unknown(
                tool, {"type", "name", "description", "format"}, "custom tool"
            )
        elif not chat and kind == "namespace":
            refusal = _unknown(
                tool, {"type", "name", "description", "tools"}, "namespace tool"
            ) or _openai_tools_refusal(tool.get("tools", []), chat=False)
        elif not chat and kind == "tool_search" and tool.get("execution") == "client":
            refusal = _unknown(
                tool, {"type", "execution", "description", "parameters"}, "tool_search"
            )
        else:
            return f"tool type {kind!r} is not a client-executed function"
        if refusal:
            return refusal
    return None


def _tool_choice_refusal(choice, kinds):
    if choice is None or isinstance(choice, str):
        return None
    kind = choice.get("type") if isinstance(choice, dict) else None
    if kind not in kinds:
        return f"tool_choice type {kind!r} is not a client-executed function"
    if kind == "allowed_tools":
        tools = choice.get("tools", [])
        if not isinstance(tools, list) or any(
            not isinstance(t, dict) or t.get("type") not in {"function", "custom"}
            for t in tools
        ):
            return "allowed_tools names a tool that is not a function"
    return None


def _chat_refusal(body):
    refusal = _unknown(body, _OUT_CHAT, "request")
    if refusal is None and "tools" in body:
        refusal = _openai_tools_refusal(body["tools"], chat=True)
    if refusal is None and "functions" in body:
        functions = body["functions"]
        if not isinstance(functions, list):
            return "functions is not a list"
        for function in functions:
            refusal = refusal or _unknown(
                function, {"name", "description", "parameters"}, "function"
            )
    if refusal is None:
        refusal = _tool_choice_refusal(
            body.get("tool_choice"), {"function", "custom", "allowed_tools"}
        )
    call = body.get("function_call")
    if refusal is None and call is not None and not isinstance(call, str):
        refusal = _unknown(call, {"name"}, "function_call")
    return refusal


def _responses_refusal(body):
    refusal = _unknown(body, _OUT_RESPONSES, "request")
    if refusal is None and "tools" in body:
        refusal = _openai_tools_refusal(body["tools"], chat=False)
    if refusal is None:
        refusal = _tool_choice_refusal(
            body.get("tool_choice"), {"function", "custom", "allowed_tools"}
        )
    items = body.get("input")
    if refusal is None and isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                return "input item is not an object"
            kind = item.get("type", "message" if "role" in item else None)
            if kind not in _RESPONSE_INPUT_FIELDS:
                return f"input item type {kind!r} is not supported"
            if kind in {"additional_tools", "tool_search_output"}:
                refusal = refusal or _openai_tools_refusal(
                    item.get("tools", []), chat=False
                )
    elif refusal is None and items is not None and not isinstance(items, str):
        refusal = "input is not text or a list"
    return refusal


def _anthropic_tools_refusal(tools):
    """Custom tools and Anthropic's versioned client tools; never server tools."""
    if not isinstance(tools, list):
        return "tools is not a list"
    for tool in tools:
        kind = tool.get("type", "custom") if isinstance(tool, dict) else None
        if kind == "custom":
            allowed = ANTHROPIC_CUSTOM_TOOL
        elif kind in ANTHROPIC_CLIENT_TOOLS:
            allowed = ANTHROPIC_CLIENT_TOOLS[kind]
        else:
            return f"tool type {kind!r} is not a client-executed function"
        refusal = _unknown(tool, allowed, f"{kind} tool")
        if refusal:
            return refusal
        # Programmatic callers that are server tools would run it provider-side.
        callers = tool.get("allowed_callers", [])
        if (
            not isinstance(callers, list)
            or not set(callers) <= ANTHROPIC_CLIENT_CALLERS
        ):
            return f"{kind} tool names a server-side caller"
    return None


def _messages_refusal(body, path=""):
    refusal = _unknown(body, _OUT_MESSAGES, "request")
    if refusal is None and "tools" in body:
        refusal = _anthropic_tools_refusal(body["tools"])
    choice = body.get("tool_choice")
    if refusal is None and choice is not None:
        refusal = _unknown(
            choice, {"type", "name", "disable_parallel_tool_use"}, "tool_choice"
        )
        if refusal is None and choice.get("type") not in {
            "auto",
            "any",
            "tool",
            "none",
        }:
            refusal = "tool_choice type is not a client-executed function"
    return refusal


def _gemini_refusal(body, fields):
    refusal = _unknown(body, fields, "request")
    tools = body.get("tools", [])
    if refusal is None and not isinstance(tools, list):
        return "tools is not a list"
    for tool in tools if refusal is None else []:
        refusal = refusal or _unknown(
            tool, {"functionDeclarations", "function_declarations"}, "tool"
        )
        for key in ("functionDeclarations", "function_declarations"):
            declarations = tool.get(key, []) if refusal is None else []
            if not isinstance(declarations, list):
                return "function declarations are not a list"
            for declaration in declarations:
                refusal = refusal or _unknown(
                    declaration,
                    {
                        "name",
                        "description",
                        "parameters",
                        "parametersJsonSchema",
                        "parameters_json_schema",
                        "response",
                        "responseJsonSchema",
                        "response_json_schema",
                        "behavior",
                    },
                    "function declaration",
                )
    for key in ("toolConfig", "tool_config"):
        if refusal is None and key in body:
            refusal = _unknown(
                body[key],
                {"functionCallingConfig", "function_calling_config"},
                "tool config",
            )
    for key in ("cachedContent", "cached_content"):
        # A cache holds tools. Only caches made through this check are known.
        if refusal is None and key in body and body[key] not in _OWN_CACHES:
            refusal = "cached content was not created by this run"
    inner = body.get("generateContentRequest")
    if refusal is None and inner is not None:
        refusal = _gemini_refusal(
            inner, (_OUT_GEMINI - {"generateContentRequest"}) | {"model"}
        )
    return refusal


def _converse_refusal(body, path="", *, fields=_OUT_CONVERSE):
    refusal = _unknown(body, fields, "request")
    extra = body.get("additionalModelRequestFields")
    if refusal is None and extra is not None:
        refusal = _unknown(extra, _OUT_CONVERSE_EXTRA, "additionalModelRequestFields")
        if refusal is None and "tools" in extra:
            refusal = _anthropic_tools_refusal(extra["tools"])
    config = body.get("toolConfig")
    if refusal is None and config is not None:
        refusal = _unknown(config, {"tools", "toolChoice"}, "toolConfig")
        tools = config.get("tools", []) if refusal is None else []
        if not isinstance(tools, list):
            return "tools is not a list"
        for tool in tools:
            if not isinstance(tool, dict) or len(tool) != 1:
                return "tool is not a single tool specification"
            if "cachePoint" in tool:
                continue
            refusal = refusal or _unknown(tool, {"toolSpec"}, "tool")
            refusal = refusal or _unknown(
                tool["toolSpec"],
                {"name", "description", "inputSchema", "strict"},
                "toolSpec",
            )
        choice = config.get("toolChoice") if refusal is None else None
        if choice is not None:
            refusal = _unknown(choice, {"auto", "any", "tool"}, "toolChoice")
    return refusal


def _bedrock_count_refusal(body, path):
    # One wrapped request: Converse fields, or a base64 Invoke body.
    refusal = _unknown(body, {"input"}, "request") or _unknown(
        body["input"], {"converse", "invokeModel"}, "input"
    )
    if refusal is None and len(body["input"]) != 1:
        refusal = "input is not a single request"
    if refusal is None and "converse" in body["input"]:
        refusal = _converse_refusal(body["input"]["converse"])
    elif refusal is None:
        invoke = body["input"]["invokeModel"]
        refusal = _unknown(invoke, {"body"}, "invokeModel")
        if refusal is None:
            try:
                inner = json.loads(base64.b64decode(invoke["body"], validate=True))
            except (ValueError, TypeError):
                return "invokeModel body is not base64 JSON"
            if not isinstance(inner, dict):
                return "invokeModel body is not an object"
            refusal = _invoke_body_refusal(inner, path)
    return refusal


def _invoke_body_refusal(body, path):
    family = _invoke_family(path)
    if family == "anthropic":
        return _messages_refusal(body)
    if family == "nova":
        return _converse_refusal(body, fields=_OUT_NOVA)
    if family in {"openai", "moonshot"}:
        return _chat_refusal(body)
    if family == "cohere":
        refusal = _unknown(body, _INVOKE_COHERE, "request")
        if refusal is None and "tools" in body:
            refusal = _openai_tools_refusal(body["tools"], chat=True)
        return refusal
    if family in _INVOKE_PROMPT_FIELDS:
        return _unknown(body, _INVOKE_PROMPT_FIELDS[family], "request")
    return f"Bedrock Invoke model family {family!r} is not supported"


_OUT_CHECKS = {
    "chat": lambda body, path: _chat_refusal(body),
    "bedrock_count": _bedrock_count_refusal,
    "responses": lambda body, path: _responses_refusal(body),
    "messages": _messages_refusal,
    "invoke": _invoke_body_refusal,
    "converse": _converse_refusal,
    "gemini": lambda body, path: _gemini_refusal(body, _OUT_GEMINI),
    "gemini_cache": lambda body, path: _gemini_refusal(body, _OUT_GEMINI_CACHE),
}


def outbound_refusal(method, path, content):
    """Why one translated upstream request may not leave, or None.

    Only inference POSTs on a known wire protocol can leave, plus the Gemini
    cache listing LiteLLM's context caching reads. A proxy-side fetch (a
    translator downloading a URL) is no inference request and is refused.
    The path is the request's encoded path, without its query: a Bedrock ARN
    keeps its slashes encoded inside one path segment.
    """
    protocol = next((name for rx, name in _OUT_PATHS if rx.fullmatch(path)), None)
    if method == "GET" and protocol == "gemini_cache" and not content:
        return None
    if method != "POST" or protocol is None:
        return f"{method} {path} is not a model inference request"
    try:
        body = json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        return "request body is not JSON"
    if not isinstance(body, dict):
        return "request body is not an object"
    # A model that searches by itself declares nothing to refuse: decide from
    # the controller-resolved model and the one this request names.
    # A Bedrock id's colons are part of the id; elsewhere LiteLLM parses it.
    wire = "bedrock" if protocol in {"invoke", "converse", "bedrock_count"} else None
    named = re.fullmatch(r".*/model/([^/]+)/[a-z-]+|.*/models/([^/:]+):\w+", path)
    for model, provider in (
        (os.environ.get("BENCHFLOW_LITELLM_UPSTREAM_MODEL"), None),
        (body.get("model"), wire),
        (named and (named[1] or named[2]), wire),
    ):
        why = hosted_retrieval_model(model, provider)
        if why:
            return why
    try:
        return _OUT_CHECKS[protocol](body, path)
    except (AttributeError, TypeError):
        return "malformed tool declaration"


def _wire_path(request):
    return request.url.raw_path.split(b"?", 1)[0].decode("ascii", "replace")


def _web_restricted():
    return os.environ.get("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS") == "1"


def _refused_response(httpx, request, reason):
    message = f"{_WEB_TOOLS_DISABLED}: {reason}"
    return httpx.Response(
        400,
        json={"error": {"message": message, "type": "invalid_request_error"}},
        request=request,
    )


def _record_cache(request, response):
    if request.method == "POST" and request.url.path.endswith("/cachedContents"):
        try:
            name = response.json().get("name") if response.is_success else None
        except Exception:
            name = None
        if isinstance(name, str):
            _OWN_CACHES.add(name)


def install_outbound_check() -> None:
    """Re-check every upstream request at the HTTP client, after translation.

    Every LiteLLM inference route (its own HTTP handlers, the OpenAI SDK
    clients, the native pass-through) sends through httpx.Client.send or
    httpx.AsyncClient.send; its aiohttp transport sits below that call. The
    aiohttp-only handler is refused unless it runs inside a checked send.
    """
    import contextvars

    import httpx

    if httpx.AsyncClient.send in _WRAPPERS:
        return
    checked = contextvars.ContextVar("benchflow_outbound_checked", default=False)
    sync_send = httpx.Client.send
    async_send = httpx.AsyncClient.send

    def send(self, request, **kwargs):
        if _web_restricted():
            reason = outbound_refusal(
                request.method, _wire_path(request), request.read()
            )
            if reason is not None:
                return _refused_response(httpx, request, reason)
        token = checked.set(True)
        attempt = None
        try:
            with suppress(Exception):
                attempt = _UpstreamAttempt(request)
            response = sync_send(self, request, **kwargs)
            if attempt is not None:
                with suppress(Exception):
                    response = _track_response(
                        httpx, response, attempt, asynchronous=False
                    )
        except BaseException:
            if attempt is not None:
                with suppress(Exception):
                    attempt.end()
            raise
        finally:
            checked.reset(token)
        if not kwargs.get("stream"):
            _record_cache(request, response)
        return response

    async def asend(self, request, **kwargs):
        if _web_restricted():
            content = await request.aread()
            reason = outbound_refusal(request.method, _wire_path(request), content)
            if reason is not None:
                return _refused_response(httpx, request, reason)
        token = checked.set(True)
        attempt = None
        try:
            with suppress(Exception):
                attempt = _UpstreamAttempt(request)
            response = await async_send(self, request, **kwargs)
            if attempt is not None:
                with suppress(Exception):
                    response = _track_response(
                        httpx, response, attempt, asynchronous=True
                    )
        except BaseException:
            if attempt is not None:
                with suppress(Exception):
                    attempt.end()
            raise
        finally:
            checked.reset(token)
        if not kwargs.get("stream"):
            _record_cache(request, response)
        return response

    _WRAPPERS.update((send, asend))
    httpx.Client.send = send  # ty: ignore[invalid-assignment]
    httpx.AsyncClient.send = asend  # ty: ignore[invalid-assignment]
    try:
        import aiohttp
    except ImportError:
        return
    session_request = aiohttp.ClientSession._request

    async def unchecked_request(self, method, url, *args, **kwargs):
        if _web_restricted() and not checked.get():
            raise aiohttp.ClientError(
                f"{_WEB_TOOLS_DISABLED}: request outside the checked HTTP client"
            )
        return await session_request(self, method, url, *args, **kwargs)

    aiohttp.ClientSession._request = unchecked_request  # ty: ignore[invalid-assignment]


class SingleRouteGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive, send):
        if scope["type"] == "websocket":
            # The vendor also exposes Responses/realtime sockets. They must
            # not provide a second, uninspected way to submit routing controls.
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        method = scope["method"]
        allowed = _POST_FIELDS.get(path) if method == "POST" else None
        query = parse_qsl(
            scope.get("query_string", b"").decode("ascii", errors="replace"),
            keep_blank_values=True,
        )
        valid = not query and (
            (method == "GET" and path in _GET_PATHS) or allowed is not None
        )
        if allowed is _MESSAGES and query == [("beta", "true")]:
            valid = True
        native = _GEMINI_PATH.fullmatch(path) if method == "POST" else None
        vertex_path = None
        if native is not None:
            # A native URL is itself a routing instruction. Permit only the
            # resolved upstream model and explicit inference operations.
            served = json.loads(os.environ.get("BENCHFLOW_LITELLM_SERVED_MODELS", "[]"))
            valid = f"gemini/{native[1]}" in served and all(
                k in {"key", "alt"} for k, _ in query
            )
            allowed = _GEMINI_FIELDS
            vertex = json.loads(os.environ.get("BENCHFLOW_LITELLM_VERTEX_NATIVE", "{}"))
            if f"vertex_ai/{native[1]}" in served and vertex.get("model") == native[1]:
                project = vertex.get("project", "")
                location = vertex.get("location", "")
                valid = (
                    isinstance(project, str)
                    and isinstance(location, str)
                    and re.fullmatch(r"[a-zA-Z0-9_-]+", project) is not None
                    and re.fullmatch(r"[a-zA-Z0-9_-]+", location) is not None
                    and all(k in {"key", "alt"} for k, _ in query)
                )
                vertex_path = f"/vertex_ai/v1/projects/{project}/locations/{location}/publishers/google/models/{native[1]}:{native[2]}"
        chunks = []
        unknown_fields = []
        if valid and method == "POST":
            # LiteLLM chooses its parser from Content-Type. Validate and replay
            # one JSON object, never bytes another parser can reinterpret as
            # form routing fields. Ambiguous duplicate media headers refuse.
            media = [
                v for k, v in scope.get("headers", []) if k.lower() == b"content-type"
            ]
            valid = (
                len(media) == 1
                and media[0].split(b";", 1)[0].strip().lower() == b"application/json"
            )
            more = True
            size = 0
            while more:
                message = await receive()
                if message["type"] != "http.request":
                    return
                chunks.append(message.get("body", b""))
                size += len(chunks[-1])
                if size > MAX_REQUEST_BYTES or len(chunks) > 16384:
                    valid = False
                    break
                more = message.get("more_body", False)
            try:
                payload = b"".join(chunks) if valid else b""
                chunks = [payload]
                body = json.loads(payload)
                if isinstance(body, dict):
                    unknown_fields = sorted(body.keys() - allowed)
                valid = valid and isinstance(body, dict) and body.keys() <= allowed
                if valid and allowed is _MESSAGES and "metadata" in body:
                    metadata = body["metadata"]
                    valid = (
                        isinstance(metadata, dict)
                        and metadata.keys() <= {"user_id"}
                        and (
                            "user_id" not in metadata
                            or isinstance(metadata["user_id"], str)
                        )
                    )
                if valid:
                    chunks = [json.dumps(body, separators=(",", ":")).encode()]
                if (
                    valid
                    and os.environ.get("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS") == "1"
                ):
                    valid = _local_tools(
                        body, native=native is not None, messages=allowed is _MESSAGES
                    )
                    if valid and allowed is _RESPONSES:
                        valid = _local_response_input(body)
                    elif valid and allowed in (_CHAT, _MESSAGES):
                        valid = _local_messages(body, messages=allowed is _MESSAGES)
                    elif valid and native is not None:
                        valid = _local_gemini_content(body)
                if valid and native is not None and "generateContentRequest" in body:
                    # countTokens can carry a whole generation request. Keep
                    # that request on this same model, rather than trusting a
                    # second model selector inside the envelope.
                    inner = body["generateContentRequest"]
                    valid = (
                        isinstance(inner, dict)
                        and inner.keys() <= (_GEMINI_FIELDS | {"model"})
                        and inner.get("model", f"models/{native[1]}")
                        == f"models/{native[1]}"
                    )
                    if (
                        valid
                        and os.environ.get("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS")
                        == "1"
                    ):
                        valid = _local_tools(inner, native=True, messages=False) and (
                            _local_gemini_content(inner)
                        )
                        # A generation request is a single envelope, not an
                        # arbitrarily nested countTokens carrier.
                        valid = valid and "generateContentRequest" not in inner
            except (ValueError, UnicodeError, RecursionError):
                valid = False
        if not valid:
            message = "request outside this run's protocol allowlist"
            if unknown_fields:
                message += ": unsupported fields " + ", ".join(unknown_fields)
            payload = json.dumps(
                {"error": {"message": message, "type": "invalid_request_error"}}
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": payload})
            return
        scope = {
            **scope,
            "headers": [
                (k, v)
                for k, v in scope.get("headers", [])
                if k.lower() in _HEADERS and k.lower() != b"content-type"
            ]
            + ([(b"content-type", b"application/json")] if method == "POST" else []),
        }
        if vertex_path is not None:
            scope = {**scope, "path": vertex_path, "raw_path": vertex_path.encode()}
            # Gemini authenticates with ?key= or x-goog-api-key; Vertex's
            # vendor handler expects Authorization. Translate the gateway
            # credential before authentication, then discard native key
            # carriers so they cannot travel to the upstream Vertex service.
            gateway_key = dict(query).get("key") or next(
                (
                    v.decode("latin-1")
                    for k, v in scope["headers"]
                    if k.lower() == b"x-goog-api-key"
                ),
                None,
            )
            scope["headers"] = [
                (k, v)
                for k, v in scope["headers"]
                if k.lower() != b"x-goog-api-key"
                and (not gateway_key or k.lower() != b"authorization")
            ]
            if gateway_key:
                scope["headers"].append(
                    (b"authorization", ("Bearer " + gateway_key).encode("latin-1"))
                )
            scope["query_string"] = urlencode(
                [(k, v) for k, v in query if k != "key"]
            ).encode()

        async def replay():
            if chunks:
                return {
                    "type": "http.request",
                    "body": chunks.pop(0),
                    "more_body": bool(chunks),
                }
            return await receive()

        if method != "POST":
            return await self.app(scope, receive, send)
        state = {
            "id": uuid.uuid4().hex,
            "upstream_seconds": 0.0,
            "retry_wait_seconds": 0.0,
            "failed_at": None,
        }
        token = _REQUEST.set(state)
        started = time.monotonic()
        status = None
        response_bytes = 0
        _lifecycle("request_start", request_id=state["id"], request_bytes=size)

        async def counted_send(message):
            nonlocal status, response_bytes
            if message["type"] == "http.response.start":
                status = message["status"]
                # The provider answered: from here a pending request is the
                # response on its way to the agent, not a provider stall.
                _lifecycle("response_start", request_id=state["id"], status=status)
            if message["type"] == "http.response.body":
                response_bytes += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, replay, counted_send)
        finally:
            _lifecycle(
                "request_end",
                request_id=state["id"],
                status=status,
                response_bytes=response_bytes,
                duration_seconds=time.monotonic() - started,
                upstream_seconds=state["upstream_seconds"],
                retry_wait_seconds=state["retry_wait_seconds"],
            )
            _REQUEST.reset(token)


def install() -> None:
    """Called by sitecustomize before the CLI can build its middleware stack."""
    from litellm.proxy.proxy_server import app

    install_outbound_check()
    app.add_middleware(SingleRouteGuard)
