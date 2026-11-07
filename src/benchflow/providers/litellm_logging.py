"""LiteLLM callback logger source and callback-log import helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._utils.reported_models import add_model
from benchflow.trajectories.call_purpose import infer_call_purpose
from benchflow.trajectories.token_capture import (
    TOKEN_CAPTURE_METADATA_KEY,
    build_token_capture,
    strip_captured_token_ids,
)
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)
from benchflow.usage_tracking import usage_unavailable

_PROVIDER_AUTH_STATUS_CODES = (401, 403)
_PROVIDER_FAILURE_STATUS_CODES = (400, 401, 402, 403, 404, 408, 422, 429, 500, 503)
_STATUS_KEYS = {
    "httpstatus",
    "httpstatuscode",
    "status",
    "statuscode",
    "status_code",
    "http_status",
    "http_status_code",
    "response_code",
    "response_status",
    "response_status_code",
}
# Keep free-text matching narrower than structured status-code handling. Broad
# codes like 400 or 500 are too common in tracebacks and payload snippets to infer
# provider failure unless they arrive through a real status field.
_PROVIDER_FAILURE_STATUS_RE = re.compile(r"\b(401|403|429|503)\b")
_PROVIDER_AUTH_HINT_RE = re.compile(
    r"\b("
    r"auth(?:entication|orization)?|"
    r"unauthori[sz]ed|"
    r"permission_denied|"
    r"forbidden|"
    r"bearer|"
    r"api[-_ ]?key|"
    r"credentials?"
    r")\b",
    re.IGNORECASE,
)
_PROVIDER_RATE_LIMIT_HINT_RE = re.compile(
    r"\b("
    r"rate[-_ ]?limit(?:ed)?|"
    r"too many requests|"
    r"too many tokens|"
    r"tokens per day|"
    r"quota"
    r")\b",
    re.IGNORECASE,
)
_PROVIDER_UNAVAILABLE_HINT_RE = re.compile(
    r"\b("
    r"service unavailable|"
    r"temporarily unavailable|"
    r"overloaded|"
    r"upstream unavailable"
    r")\b",
    re.IGNORECASE,
)
_CONTEXT_LIMIT_HINT_RE = re.compile(
    r"\b("
    r"context[-_ ]?(?:length|window)|"
    r"context_length_exceeded|"
    r"contextwindowexceeded|"
    r"max(?:imum)?[_ -]?model[_ -]?len|"
    r"prompt is too long|"
    r"requested token count exceeds|"
    r"maximum context"
    r")\b",
    re.IGNORECASE,
)


def callback_module_source() -> str:
    """Return the Python module written next to LiteLLM config.yaml.

    It ends with ``litellm_messages_bridge_patch.py``, so the proxy fixes
    LiteLLM's ``/v1/messages`` bridge to chat completions when it loads the
    callback.
    """
    return _CALLBACK_SOURCE + "\n\n" + _BRIDGE_PATCH.read_text(encoding="utf-8")


_BRIDGE_PATCH = Path(__file__).with_name("litellm_messages_bridge_patch.py")
_CALLBACK_SOURCE = r"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


_skill_catalog_gate_passed = False


def _required_skill_names() -> tuple[str, ...]:
    raw = os.environ.get("BENCHFLOW_REQUIRED_SKILL_NAMES_JSON", "")
    if not raw:
        return ()
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "experiment_fidelity/skill_catalog_gate_config_invalid"
        ) from exc
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise RuntimeError("experiment_fidelity/skill_catalog_gate_config_invalid")
    return tuple(sorted(set(values)))


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )
    return ""


def _opencode_catalog_names(data: dict[str, Any]) -> set[str]:
    system = "\n".join(
        _message_text(message.get("content"))
        for message in data.get("messages") or []
        if isinstance(message, dict) and message.get("role") == "system"
    )
    match = re.search(
        r"<available_skills>(.*?)</available_skills>", system, flags=re.DOTALL
    )
    if not match:
        return set()
    return set(re.findall(r"<name>\s*([^<]+?)\s*</name>", match.group(1)))


def _gate_opencode_skill_catalog(data: dict[str, Any]) -> None:
    global _skill_catalog_gate_passed
    if _skill_catalog_gate_passed:
        return
    if os.environ.get("BENCHFLOW_SKILL_CATALOG_GATE_AGENT") != "opencode":
        return
    expected = _required_skill_names()
    if not expected:
        return
    visible = _opencode_catalog_names(data)
    missing = sorted(set(expected) - visible)
    if missing:
        raise RuntimeError(
            "experiment_fidelity/skill_catalog_missing: "
            f"missing={','.join(missing)} expected={','.join(expected)} "
            f"visible={','.join(sorted(visible))}"
        )
    _skill_catalog_gate_passed = True


def _served_models() -> frozenset[str] | None:
    # Every name the run's config.yaml routes (BENCHFLOW_LITELLM_SERVED_MODELS,
    # set by the controller); None when the controller named none.
    raw = os.environ.get("BENCHFLOW_LITELLM_SERVED_MODELS", "")
    if not raw:
        return None
    try:
        values = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(values, list):
        return None
    return frozenset(value for value in values if isinstance(value, str) and value)


class ModelNotServed(Exception):
    # LiteLLM answers with the exception's status_code; 400 is what it sends
    # itself for a model its router does not know.
    status_code = 400

    def __init__(self, model: str) -> None:
        self.message = (
            f"model_not_served: this run's gateway serves only the run's model, "
            f"not {model!r}"
        )
        super().__init__(self.message)


# The trial's token and turn budget (benchflow.trial_budget), set by the
# controller in the proxy's environment, and what this proxy has counted: a
# turn is one model call it admitted, tokens are every admitted call's input
# (cached included) plus output, as total_tokens counts them. An earlier proxy
# of the same trial hands on what it counted (BENCHFLOW_TRIAL_BUDGET_USED). A
# variable that does not read as a positive whole number stops the proxy from
# starting: a budget is never silently dropped.
def _read_trial_budget() -> dict[str, Any]:
    limits: dict[str, int] = {}
    for measure, name in (
        ("tokens", "BENCHFLOW_TRIAL_TOKEN_BUDGET"),
        ("turns", "BENCHFLOW_TRIAL_TURN_BUDGET"),
    ):
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        if not raw.isdecimal() or int(raw) <= 0:
            raise RuntimeError(f"{name}={raw!r} is not a positive whole number")
        limits[measure] = int(raw)
    used = {"tokens": 0, "turns": 0}
    raw = os.environ.get("BENCHFLOW_TRIAL_BUDGET_USED", "").strip()
    if raw:
        carried = json.loads(raw)
        for measure in used:
            value = carried.get(measure, 0) if isinstance(carried, dict) else None
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"BENCHFLOW_TRIAL_BUDGET_USED={raw!r} is not usage")
            used[measure] = value
    return {"limits": limits, "used": used, "refused": 0}


_TRIAL_BUDGET = _read_trial_budget()


class TrialBudgetSpent(Exception):
    # A 400, which agents do not retry: the call would never be allowed.
    status_code = 400

    def __init__(self, measure: str, limit: int, used: dict[str, int]) -> None:
        self.message = (
            f"trial_budget_spent: this trial's {measure[:-1]} budget of {limit} is "
            f"spent ({used['tokens']} tokens in {used['turns']} model calls); "
            "the run allows it no more model calls"
        )
        super().__init__(self.message)


def _first_count(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _call_tokens(usage: Any) -> int:
    # One call's tokens, as BenchFlow totals them (trajectories.types
    # _exchange_token_usage): the provider's own total when it gives one, else
    # the input with any cache counted beside it folded in, plus the output.
    if not isinstance(usage, dict):
        return 0
    total = _first_count(
        usage.get("total_tokens"), usage.get("totalTokenCount"), usage.get("totalTokens")
    )
    if total is not None:
        return max(total, 0)
    prompt = usage.get("prompt_tokens_details")
    inputs = usage.get("input_tokens_details")
    inclusive = _first_count(
        prompt.get("cached_tokens") if isinstance(prompt, dict) else None,
        inputs.get("cached_tokens") if isinstance(inputs, dict) else None,
        usage.get("cachedContentTokenCount"),
    )
    cache_read = 0 if inclusive else (
        _first_count(
            usage.get("cache_read_input_tokens"),
            usage.get("cacheReadInputTokens"),
            usage.get("cacheReadInputTokenCount"),
        )
        or 0
    )
    cache_write = (
        _first_count(
            usage.get("cache_creation_input_tokens"),
            usage.get("cacheWriteInputTokens"),
            usage.get("cacheWriteInputTokenCount"),
        )
        or 0
    )
    raw_input = (
        _first_count(
            usage.get("input_tokens"),
            usage.get("prompt_tokens"),
            usage.get("inputTokens"),
            usage.get("promptTokenCount"),
        )
        or 0
    )
    output = (
        _first_count(
            usage.get("output_tokens"),
            usage.get("completion_tokens"),
            usage.get("outputTokens"),
            usage.get("candidatesTokenCount"),
        )
        or 0
    )
    return (
        raw_input
        + cache_read
        + cache_write
        + (_first_count(usage.get("toolUsePromptTokenCount")) or 0)
        + output
        + (_first_count(usage.get("thoughtsTokenCount")) or 0)
    )


def _is_model_call(data: dict[str, Any]) -> bool:
    # A token count asks the provider for no answer, so it is no turn.
    request = data.get("proxy_server_request")
    url = str(request.get("url") or "") if isinstance(request, dict) else ""
    return "count_tokens" not in url and "countTokens" not in url


def _spent_measure() -> str | None:
    state = _TRIAL_BUDGET
    for measure in ("tokens", "turns"):
        limit = state["limits"].get(measure)
        if limit is not None and state["used"][measure] >= limit:
            return measure
    return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    return str(value)


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _usage_from_response(response: Any) -> Any:
    data = _jsonable(response)
    if isinstance(data, dict):
        return data.get("usage") or data.get("usageMetadata")
    return None


def _failure_detail(response_obj: Any, exception: Any) -> Any:
    # On a deterministic provider reject (e.g. a context-window overflow,
    # #830) litellm fires the failure hook with response_obj=None and the real
    # cause in kwargs['exception']. Fall back to it so error.type/message carry
    # the upstream reason instead of the literal 'NoneType'/'None'. A non-None
    # response_obj still wins, so the existing success-shaped-failure path is
    # unchanged; both None degrades gracefully to the old behavior.
    return response_obj if response_obj is not None else exception


def _failure_traceback(detail: Any) -> str:
    # ``format_exc()`` reflects the exception that is CURRENTLY ACTIVE when the
    # callback fires. For the #830 case (litellm calls the hook from inside its
    # except block) that IS ``detail``, so the traceback agrees with the
    # exception-derived error.message. But if litellm clears the exception
    # context first, ``format_exc()`` returns the ``'NoneType: None'`` sentinel
    # while ``detail`` (recovered from kwargs['exception']) still holds the real
    # cause — format that exception directly so the traceback doesn't go blank
    # under a meaningful error.message (#830).
    tb = traceback.format_exc()
    if isinstance(detail, BaseException) and tb.startswith("NoneType: None"):
        tb = "".join(
            traceback.format_exception(type(detail), detail, detail.__traceback__)
        )
    return tb[-2000:]


_TRUTHY = {"1", "true", "yes", "on"}
# How a route provider's OpenAI-compatible server is asked for token ids:
# ``vllm``: ``return_token_ids`` (top-level ``prompt_token_ids``, per-choice
# ``token_ids``); ``sglang``: the ``sglext`` extension flags below, which also
# work on streamed chat (SGLang refuses ``return_token_ids`` there).
_TOKEN_ID_PROVIDERS = {"vllm": "vllm", "sglang": "sglang"}
_SGLANG_ID_FLAGS = ("return_input_ids_in_sglext", "return_output_ids_in_sglext")
_RESPONSES_LOGPROBS_INCLUDE = "message.output_text.logprobs"
_STREAM_TOKENS_KEY = "benchflow_stream_tokens"


def _upstream_api(upstream: str) -> str:
    head, _, rest = upstream.partition("/")
    lowered = rest.lower()
    if head in {"anthropic", "bedrock"} or (
        head in {"vertex_ai", "azure_ai"} and "claude" in lowered
    ):
        return "anthropic-messages"
    if head in {"gemini", "vertex_ai"}:
        return "gemini"
    if head in {"openai", "azure", "deepseek", "hosted_vllm"}:
        return "openai"
    return "other"


def _token_capture_plan(call_type: Any, model: Any) -> dict[str, Any] | None:
    # What token data to request for one call; None while capture is off.
    # ``wire`` is the API the upstream call uses: chat calls, Anthropic
    # Messages calls LiteLLM bridges to an OpenAI-compatible chat backend, and
    # Responses calls on the ``-responses-bridge`` aliases all reach the server
    # as chat completions; other Responses calls reach a native Responses API.
    if os.environ.get("BENCHFLOW_CAPTURE_TOKEN_LOGPROBS", "").strip().lower() not in _TRUTHY:
        return None
    provider = os.environ.get("BENCHFLOW_LITELLM_ROUTE_PROVIDER", "")
    upstream_api = _upstream_api(os.environ.get("BENCHFLOW_LITELLM_UPSTREAM_MODEL", ""))
    call = str(call_type or "")
    request = None
    if call in {"completion", "acompletion"}:
        request = "chat"
        wire = "openai-chat" if upstream_api in {"openai", "other"} else upstream_api
    elif call == "anthropic_messages" and upstream_api == "openai":
        request, wire = "chat", "openai-chat"
    elif call in {"responses", "aresponses"} and str(model or "").endswith(
        "-responses-bridge"
    ):
        request, wire = "chat", "openai-chat"
    elif call in {"responses", "aresponses"} and upstream_api == "openai":
        request, wire = "responses", "openai-responses"
    else:
        wire = upstream_api
    ids_setting = os.environ.get("BENCHFLOW_CAPTURE_TOKEN_IDS", "auto").strip().lower()
    token_ids = request == "chat" and wire == "openai-chat" and (
        ids_setting in _TRUTHY
        or (ids_setting in {"", "auto"} and provider in _TOKEN_ID_PROVIDERS)
    )
    token_id_style = _TOKEN_ID_PROVIDERS.get(provider, "vllm") if token_ids else None
    try:
        top_logprobs = int(os.environ.get("BENCHFLOW_CAPTURE_TOP_LOGPROBS", "") or 0)
    except ValueError:
        top_logprobs = 0
    return {
        "enabled": True,
        "wire": wire,
        "provider": provider or None,
        "request": request,
        "logprobs": request is not None,
        "top_logprobs": top_logprobs if request is not None and top_logprobs > 0 else None,
        "token_ids": token_ids,
        "token_id_style": token_id_style,
    }


def _apply_token_capture(data: dict[str, Any], plan: dict[str, Any] | None) -> dict[str, Any]:
    # Return ``data`` (copied when changed) with the plan's request fields.
    if not plan or plan.get("request") is None:
        return data
    cleaned = dict(data)
    top = plan.get("top_logprobs")
    if plan["request"] == "chat":
        cleaned["logprobs"] = True
        if top:
            cleaned["top_logprobs"] = top
        if plan.get("token_ids"):
            # extra_body survives drop_params on openai/ routes and LiteLLM's
            # Anthropic-Messages and Responses bridges to chat completions.
            extra = dict(cleaned.get("extra_body") or {})
            if plan.get("token_id_style") == "sglang":
                for flag in _SGLANG_ID_FLAGS:
                    extra[flag] = True
            else:
                extra["return_token_ids"] = True
            cleaned["extra_body"] = extra
    elif cleaned.get("input") is None:
        return data
    else:
        include = list(cleaned.get("include") or [])
        if _RESPONSES_LOGPROBS_INCLUDE not in include:
            include.append(_RESPONSES_LOGPROBS_INCLUDE)
        cleaned["include"] = include
        if top:
            cleaned["top_logprobs"] = top
    return cleaned


# Opt-in wire capture (BENCHFLOW_LITELLM_WIRE_LOG_DIR): each request as the
# agent sent it, before any routing change, for the native-harness wire-parity
# check. Only these headers are kept (they identify the client and the
# protocol, never credentials), and the proxy's own bookkeeping keys are left
# out of the body.
_WIRE_HEADERS = frozenset(
    {
        "user-agent",
        "anthropic-beta",
        "anthropic-version",
        "x-app",
        "originator",
        "openai-beta",
        "version",
        "x-codex-beta-features",
    }
)
_WIRE_PROXY_KEYS = frozenset(
    {
        "proxy_server_request",
        "secret_fields",
        "litellm_call_id",
        "litellm_logging_obj",
        "litellm_metadata",
        "litellm_parent_otel_span",
        "litellm_session_id",
        "litellm_trace_id",
        "provider_specific_header",
        "user_api_key_dict",
        "api_key",
    }
)


def _note_request(data: Any) -> None:
    # One line per request that reached the proxy, before any upstream call
    # (pass-through routes run this hook too): its LiteLLM call id, or "?"
    # when it has none. The host stops the proxy as soon as the callback log
    # is quiet when no request ever came: no record can still be on its way,
    # so the 12 s settle wait is not needed.
    _mark(".requests", data.get("litellm_call_id") if isinstance(data, dict) else None)


def _note_logged(kwargs: Any) -> None:
    # The call id of a request whose record is now whole in the callback log.
    # Once every id in ".requests" is here, the host stops the proxy at once,
    # with no settle wait (PERF-B).
    _mark(".logged", kwargs.get("litellm_call_id") if isinstance(kwargs, dict) else None)


def _mark(suffix: str, call_id: Any) -> None:
    path = os.environ.get("BENCHFLOW_LITELLM_LOG_PATH")
    if not path:
        return
    usable = isinstance(call_id, str) and call_id and call_id.isprintable()
    text = call_id if usable and call_id.strip() == call_id else "?"
    try:
        with open(path + suffix, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    except OSError:
        traceback.print_exc()


def _write_wire(data: dict[str, Any], call_type: Any) -> None:
    directory = os.environ.get("BENCHFLOW_LITELLM_WIRE_LOG_DIR")
    if not directory:
        return
    try:
        request = data.get("proxy_server_request") or {}
        url = str(request.get("url") or "")
        headers = {
            str(k).lower(): v
            for k, v in (request.get("headers") or {}).items()
            if str(k).lower() in _WIRE_HEADERS
        }
        body = {k: v for k, v in data.items() if k not in _WIRE_PROXY_KEYS}
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            # LiteLLM fills metadata with its own entries; the Messages API's
            # own metadata field carries only user_id.
            body["metadata"] = {k: v for k, v in metadata.items() if k == "user_id"}
        record = {
            "call_type": str(call_type),
            "path": "/" + url.split("://", 1)[-1].split("/", 1)[-1] if url else None,
            "headers": headers,
            "body": body,
        }
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"wire-{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), separators=(",", ":")) + "\n")
    except Exception:
        traceback.print_exc()


# Each tool_result of an Anthropic Messages request whose content holds a
# document or an image block, by its tool_use id.
def _tool_result_blocks(messages: Any) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}
    for message in messages if isinstance(messages, list) else ():
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else ():
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            inner = block.get("content")
            if isinstance(inner, list) and any(
                isinstance(item, dict) and item.get("type") in ("document", "image")
                for item in inner
            ):
                found[str(block.get("tool_use_id"))] = inner
    return found


# A part of a bridged tool message by its type. LiteLLM 1.103.3's bridge
# sends a document as an image_url part holding its data URL
# (data:application/pdf;base64,...), which a chat-completions model does not
# read as the document; such a part is named for what it holds
# (image_url:application/pdf), so it does not count as an image.
def _bridged_part(part: dict[str, Any]) -> Any:
    kind = part.get("type")
    if kind != "image_url":
        return kind
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    if isinstance(url, str) and url.startswith("data:"):
        media = url[5:].split(",", 1)[0].split(";", 1)[0].strip().lower()
        if media and not media.startswith("image/"):
            return f"image_url:{media}"
    return kind


# What LiteLLM's bridge made of each tool result holding a document or an
# image, by tool_use id: the parts and text of the tool message the provider
# got. Only for a /v1/messages call LiteLLM sent on as chat completions; the
# request this callback records is the agent's, before the translation, so it
# cannot show what the bridge kept. The translation is LiteLLM's own (with
# BenchFlow's bridge patch when installed), run again here: it reads a PDF from
# its cache. None when there is nothing to say.
def _bridge_delivery(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    litellm_params = kwargs.get("litellm_params") or {}
    call_type = kwargs.get("call_type") or litellm_params.get("call_type")
    if call_type not in ("anthropic_messages", "aanthropic_messages"):
        return None
    wanted = _tool_result_blocks(kwargs.get("messages"))
    if not wanted:
        return None
    provider = kwargs.get("custom_llm_provider") or litellm_params.get(
        "custom_llm_provider"
    )
    model = str(kwargs.get("model") or "")
    from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
    )
    from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
        _should_route_to_responses_api,
    )
    from litellm.utils import ProviderConfigManager

    if provider not in [item.value for item in litellm.LlmProviders]:
        return None
    if ProviderConfigManager.get_provider_anthropic_messages_config(
        model=model, provider=litellm.LlmProviders(provider)
    ) is not None or _should_route_to_responses_api(provider):
        # The provider's own Messages API takes the blocks as they are; the
        # Responses bridge is another translation, not read here.
        return None
    built = LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
        messages=kwargs.get("messages"), model=model
    )
    delivered: dict[str, Any] = {}
    for message in built if isinstance(built, list) else ():
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        call_id = str(message.get("tool_call_id"))
        if call_id not in wanted:
            continue
        content = message.get("content")
        if not isinstance(content, list):
            content = [{"type": "text", "text": content}]
        text = "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        delivered[call_id] = {
            "parts": [
                _bridged_part(part) for part in content if isinstance(part, dict)
            ],
            "chars": len(text),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "head": text[:240],
        }
    for call_id in wanted:
        # No tool message at all: the bridge dropped the whole result.
        delivered.setdefault(
            call_id,
            {
                "parts": [],
                "chars": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
                "head": "",
            },
        )
    return delivered


class BenchFlowLiteLLMLogger(CustomLogger):
    def _write(self, payload: dict[str, Any]) -> None:
        path = os.environ.get("BENCHFLOW_LITELLM_LOG_PATH")
        if not path:
            return
        payload["logged_at"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(payload), separators=(",", ":")) + "\n")

    def _base_record(self, kwargs: dict[str, Any], start_time: Any, end_time: Any) -> dict[str, Any]:
        litellm_params = kwargs.get("litellm_params") or {}
        optional_params = kwargs.get("optional_params") or {}
        metadata = kwargs.get("metadata") or litellm_params.get("metadata") or {}
        request_body = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages"),
            "input": kwargs.get("input"),
            "tools": optional_params.get("tools") or kwargs.get("tools"),
            "stream": optional_params.get("stream") or kwargs.get("stream"),
        }
        # The effort as each wire carries it: reasoning_effort (chat
        # completions), reasoning (the Responses API, Codex), thinking and
        # output_config (Anthropic); and a Responses request's own
        # instructions, previous_response_id and max_output_tokens.
        for key in (
            "reasoning_effort",
            "reasoning",
            "thinking",
            "output_config",
            "instructions",
            "previous_response_id",
            "max_output_tokens",
        ):
            value = optional_params.get(key)
            if value is None:
                value = kwargs.get(key)
            if value is None:
                value = litellm_params.get(key)
            if value is not None:
                request_body[key] = value
        for key in ("logprobs", "top_logprobs"):
            value = optional_params.get(key)
            if value is None:
                value = kwargs.get(key)
            if value is not None:
                request_body[key] = value
        # The sampling seed as it went upstream: a route whose parameters
        # drop it (drop_params) sends none, which the run's checks then say.
        if optional_params.get("seed") is not None:
            request_body["seed"] = optional_params["seed"]
        # The sampling the agent asked for, so a run can be checked against the
        # model vendor's setting (the run card's Vendor line) after the fact.
        # An OpenAI-compatible server's own parameters (top_k, and vLLM's
        # chat_template_kwargs, which turns a Qwen model's thinking off) arrive
        # in extra_body.
        extra_body = optional_params.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "max_tokens",
            "max_completion_tokens",
            "chat_template_kwargs",
        ):
            value = optional_params.get(key)
            if value is None:
                value = kwargs.get(key)
            if value is None:
                value = extra_body.get(key)
            if value is not None:
                request_body[key] = value
        # A pass-through request (Gemini's own generateContent, as the gemini
        # and antigravity agents send it) reaches LiteLLM unparsed, so none of
        # the above holds its settings: its generationConfig does, the
        # thinking config among them.
        passthrough = kwargs.get("passthrough_logging_payload")
        raw = passthrough.get("request_body") if isinstance(passthrough, dict) else None
        if not isinstance(raw, dict):
            server_request = litellm_params.get("proxy_server_request")
            raw = server_request.get("body") if isinstance(server_request, dict) else None
        config = raw.get("generationConfig") if isinstance(raw, dict) else None
        if isinstance(config, dict):
            request_body["generationConfig"] = {
                key: config[key]
                for key in (
                    "thinkingConfig",
                    "temperature",
                    "topP",
                    "topK",
                    "maxOutputTokens",
                    "seed",
                )
                if config.get(key) is not None
            }
        model_group = metadata.get("model_group") if isinstance(metadata, dict) else None
        call_type = kwargs.get("call_type") or litellm_params.get("call_type")
        token_capture = _token_capture_plan(call_type, model_group or kwargs.get("model"))
        extra_fields: dict[str, Any] = {}
        if token_capture is not None:
            # Record what was actually sent upstream for token capture.
            for key in ("include", "extra_body"):
                value = optional_params.get(key)
                if value is None:
                    value = kwargs.get(key)
                if value:
                    request_body[key] = value
            extra_fields["token_capture"] = token_capture
            stream_tokens = kwargs.get(_STREAM_TOKENS_KEY)
            if stream_tokens:
                extra_fields["stream_tokens"] = stream_tokens
        request_body = {k: v for k, v in request_body.items() if v is not None}
        call_type = kwargs.get("call_type") or litellm_params.get("call_type")
        if call_type in {"responses", "aresponses"}:
            path = "/v1/responses"
        elif call_type in {"anthropic_messages", "aanthropic_messages"}:
            path = "/v1/messages"
        else:
            path = "/v1/chat/completions"
        try:
            bridged = _bridge_delivery(kwargs)
        except Exception:
            bridged = None
        if bridged:
            extra_fields["bridge_tool_results"] = bridged
        return {
            **extra_fields,
            "request_model": kwargs.get("model"),
            "provider_model": litellm_params.get("model") or kwargs.get("model"),
            "model_group": metadata.get("model_group") if isinstance(metadata, dict) else None,
            "call_type": kwargs.get("call_type") or litellm_params.get("call_type"),
            "input_shape": {
                "has_messages": bool(kwargs.get("messages")),
                "has_input": kwargs.get("input") is not None,
                "n_messages": len(kwargs.get("messages") or []),
            },
            "request": {
                "method": "POST",
                "path": path,
                "body": request_body,
            },
            "start_time": _iso(start_time),
            "end_time": _iso(end_time),
            "duration_ms": max((getattr(end_time, "timestamp", lambda: time.time())() - getattr(start_time, "timestamp", lambda: time.time())()) * 1000, 0),
        }

    async def async_pre_call_hook(
        self, user_api_key_dict, cache, data, call_type
    ):
        _note_request(data)
        if not isinstance(data, dict):
            return None
        _write_wire(data, call_type)

        # Refuse, and record without the request body, a request for a model
        # the run's config does not route (codex-acp's title thread asks for
        # gpt-5.6-luna with the task prompt). LiteLLM would also refuse it;
        # the gate keeps that true whatever its routing fallbacks do, and the
        # record shows the attempt in the run.
        model = data.get("model")
        served = _served_models()
        if served is not None and isinstance(model, str) and model and model not in served:
            self._write(
                {
                    "event": "refused",
                    "rule": "model-not-served",
                    "request_model": model,
                    "call_type": call_type,
                }
            )
            raise ModelNotServed(model)

        _gate_opencode_skill_catalog(data)

        # The trial's budget: a call it no longer allows is refused here,
        # before any provider sees it, and recorded for the controller, which
        # stops the agent (benchflow.trial_budget).
        if _TRIAL_BUDGET["limits"] and _is_model_call(data):
            spent = _spent_measure()
            if spent is not None:
                _TRIAL_BUDGET["refused"] += 1
                self._write(
                    {
                        "event": "refused",
                        "rule": "trial-budget",
                        "budget": spent,
                        "limits": dict(_TRIAL_BUDGET["limits"]),
                        "used": dict(_TRIAL_BUDGET["used"]),
                        "refused": _TRIAL_BUDGET["refused"],
                        "request_model": model,
                        "call_type": call_type,
                    }
                )
                raise TrialBudgetSpent(
                    spent, _TRIAL_BUDGET["limits"][spent], _TRIAL_BUDGET["used"]
                )
            _TRIAL_BUDGET["used"]["turns"] += 1

        cleaned = data

        # Chat-completions backends reject the Responses-compatible ``input``
        # mirror when ``messages`` is already present. Copy before changing so
        # LiteLLM callers do not observe in-place mutation.
        if data.get("messages") is not None and data.get("input") is not None:
            cleaned = dict(cleaned)
            cleaned.pop("input", None)

        # Drop non-"function" tools before they reach a chat-only backend. A
        # responses-API client (codex) sends tools the Responses wire allows but
        # chat completions does not, e.g. a {"type": "namespace"} tool. When the
        # proxy bridges /v1/responses to /chat/completions for a chat-only backend
        # (deepseek, vllm), that stray tool makes the backend reject the whole
        # request ("unknown variant namespace, expected function"). The dropped
        # tools cannot be represented on the chat wire anyway; the function tools
        # (shell, file IO, ...) survive untouched.
        tools = cleaned.get("tools")
        chat_bridge_models = json.loads(os.environ.get("BENCHFLOW_LITELLM_CHAT_BRIDGE_MODELS", "[]"))
        chat_request = call_type in {"completion", "acompletion"} or cleaned.get("model") in chat_bridge_models
        if chat_request and isinstance(tools, list):
            kept = [
                t
                for t in tools
                if not isinstance(t, dict) or t.get("type", "function") == "function"
            ]
            if len(kept) != len(tools):
                if cleaned is data:
                    cleaned = dict(data)
                cleaned["tools"] = kept

        # Forward ``reasoning_effort`` VERBATIM on deepseek routes. LiteLLM's
        # deepseek transform consumes the top-level field (it maps it into its
        # own thinking handling and drops the raw param — even with drop_params
        # off), while
        # ``extra_body`` fields merge into the wire request untouched. Lifting
        # the param means the upstream receives exactly what the agent sent —
        # the same request a native (gateway-less) run produces. Scoped to
        # deepseek, today's only natively-routed completions provider
        # (_NATIVE_LITELLM_COMPLETIONS_PROVIDERS in litellm_config.py).
        model_id = str(cleaned.get("model") or "")
        if "deepseek" in model_id and cleaned.get("reasoning_effort") is not None:
            if cleaned is data:
                cleaned = dict(data)
            extra = dict(cleaned.get("extra_body") or {})
            extra.setdefault("reasoning_effort", cleaned.pop("reasoning_effort"))
            cleaned["extra_body"] = extra

        # Opt-in token capture (BENCHFLOW_CAPTURE_TOKEN_LOGPROBS): ask for
        # sampled-token logprobs, and token ids where the route supports them.
        plan = _token_capture_plan(call_type, cleaned.get("model"))
        if plan is not None and (
            plan.get("request") != "chat"
            or call_type not in {"completion", "acompletion"}
            or cleaned.get("messages") is not None
        ):
            cleaned = _apply_token_capture(cleaned, plan)
        if cleaned is data:
            return None
        return cleaned

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        record = self._base_record(kwargs, start_time, end_time)
        response = _jsonable(response_obj)
        # Cost comes from LiteLLM. Prefer the value the proxy already computed
        # (it honors per-deployment input_cost_per_token for custom models);
        # fall back to recomputing from litellm.model_cost. The proxy reports
        # 0.0 for models it cannot price, so a falsy value means "unknown"
        # (recorded as null) rather than a misleading $0.00.
        cost = None
        try:
            hidden = getattr(response_obj, "_hidden_params", None) or {}
            hidden_cost = hidden.get("response_cost")
            if hidden_cost:
                cost = hidden_cost
        except Exception:
            cost = None
        if cost is None:
            try:
                fallback = litellm.completion_cost(completion_response=response_obj)
                if fallback:
                    cost = fallback
            except Exception:
                cost = None
        usage = _usage_from_response(response_obj)
        _TRIAL_BUDGET["used"]["tokens"] += _call_tokens(usage)
        record.update(
            {
                "event": "success",
                "response": response,
                "usage": usage,
                "response_cost": cost,
            }
        )
        self._write(record)
        _note_logged(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        record = self._base_record(kwargs, start_time, end_time)
        detail = _failure_detail(response_obj, (kwargs or {}).get("exception"))
        record.update(
            {
                "event": "failure",
                "response": _jsonable(response_obj),
                "error": {
                    "type": type(detail).__name__,
                    "message": str(detail),
                    "traceback": _failure_traceback(detail),
                },
            }
        )
        self._write(record)
        _note_logged(kwargs)


proxy_handler_instance = BenchFlowLiteLLMLogger()
"""


def _parse_time(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now()


def _record_response_body(record: dict[str, Any]) -> dict[str, Any]:
    response = record.get("response")
    body = response if isinstance(response, dict) else {"raw": response}
    usage = record.get("usage")
    if isinstance(usage, dict) and "usage" not in body and "usageMetadata" not in body:
        # Gemini reports a usageMetadata block (promptTokenCount, …); other
        # providers report an OpenAI/Anthropic-style usage block. Place each
        # where Trajectory.has_provider_usage looks for it so a successful call
        # never silently degrades to usage_source='unavailable'.
        if any(
            key in usage
            for key in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
        ):
            body["usageMetadata"] = usage
        else:
            body["usage"] = usage
    if "model" not in body:
        for key in ("provider_model", "request_model", "model_group"):
            model = record.get(key)
            if isinstance(model, str) and model:
                body["model"] = model
                break
    if record.get("event") == "failure":
        body.setdefault("error", record.get("error") or {"message": "LiteLLM error"})
    return body


def _exchange_metadata(
    record: dict[str, Any],
    *,
    request_body: dict[str, Any],
    agent_name: str,
) -> dict[str, Any]:
    metadata = {
        key: record.get(key)
        for key in (
            "request_model",
            "provider_model",
            "model_group",
            "call_type",
            "input_shape",
            # What the bridge to chat completions made of a tool result that
            # held a document or an image (the callback's _bridge_delivery).
            "bridge_tool_results",
        )
        if record.get(key) is not None
    }
    metadata["call_purpose"] = infer_call_purpose(
        agent_name=agent_name,
        request_body=request_body,
    )
    token_capture = build_token_capture(record)
    if token_capture is not None:
        metadata[TOKEN_CAPTURE_METADATA_KEY] = token_capture
    return metadata


def _coerce_provider_failure_status(value: Any) -> int | None:
    if isinstance(value, int) and value in _PROVIDER_FAILURE_STATUS_CODES:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            status = int(stripped)
            if status in _PROVIDER_FAILURE_STATUS_CODES:
                return status
    return None


def _explicit_provider_failure_status(value: Any) -> int | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _STATUS_KEYS:
                status = _coerce_provider_failure_status(nested)
                if status is not None:
                    return status
        for nested in value.values():
            status = _explicit_provider_failure_status(nested)
            if status is not None:
                return status
    elif isinstance(value, list | tuple):
        for nested in value:
            status = _explicit_provider_failure_status(nested)
            if status is not None:
                return status
    return None


def _flatten_failure_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(
            part
            for item in value.items()
            if str(item[0]).lower() not in {"stack", "stacktrace", "traceback"}
            for part in (str(item[0]), _flatten_failure_text(item[1]))
            if part
        )
    if isinstance(value, list | tuple):
        return " ".join(_flatten_failure_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _flatten_context_traceback_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key, nested in value.items():
            if str(key).lower() == "traceback":
                parts.append(str(nested))
            else:
                nested_text = _flatten_context_traceback_text(nested)
                if nested_text:
                    parts.append(nested_text)
        return " ".join(parts)
    if isinstance(value, list | tuple):
        return " ".join(_flatten_context_traceback_text(item) for item in value)
    return ""


def _text_provider_failure_status(text: str) -> int | None:
    if _CONTEXT_LIMIT_HINT_RE.search(text):
        return 400
    match = _PROVIDER_FAILURE_STATUS_RE.search(text)
    status = int(match.group(1)) if match is not None else None
    if status in _PROVIDER_AUTH_STATUS_CODES:
        return status if _PROVIDER_AUTH_HINT_RE.search(text) else None
    if status == 429:
        return status if _PROVIDER_RATE_LIMIT_HINT_RE.search(text) else None
    if status == 503:
        return status if _PROVIDER_UNAVAILABLE_HINT_RE.search(text) else None
    if status is None and _PROVIDER_RATE_LIMIT_HINT_RE.search(text):
        return 429
    if status is None and _PROVIDER_UNAVAILABLE_HINT_RE.search(text):
        return 503
    return None


def _provider_failure_status_from_failure_record(record: dict[str, Any]) -> int | None:
    """Return a sanitized provider failure status from a LiteLLM failure record."""
    if record.get("event") != "failure":
        return None
    failure_payload = {
        "error": record.get("error"),
        "response": record.get("response"),
    }
    text = _flatten_failure_text(failure_payload)
    if _CONTEXT_LIMIT_HINT_RE.search(text):
        return 400

    traceback_text = _flatten_context_traceback_text(failure_payload)
    if _CONTEXT_LIMIT_HINT_RE.search(traceback_text):
        return 400

    status = _explicit_provider_failure_status(failure_payload)
    if status is not None:
        return status

    return _text_provider_failure_status(text)


def trajectory_from_litellm_callback_log(
    text: str,
    *,
    session_id: str,
    agent_name: str,
) -> Trajectory:
    """Convert LiteLLM callback JSONL into BenchFlow's trajectory schema."""
    reader = CallbackLogReader(session_id=session_id, agent_name=agent_name)
    for line in text.splitlines():
        reader.add_line(line)
    return reader.finish()


class CallbackLogReader:
    """:func:`trajectory_from_litellm_callback_log`, one callback line at a time.

    A long run's log does not have to be one string: :meth:`add_line` returns
    each new exchange whole, and ``keep`` (given to the constructor) decides
    what the trajectory holds of it.
    """

    def __init__(
        self,
        *,
        session_id: str,
        agent_name: str,
        keep: Callable[[LLMExchange], LLMExchange] | None = None,
    ) -> None:
        self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
        self._agent_name = agent_name
        self._keep = keep
        self._total_cost = 0.0
        self._saw_cost = False
        self._reported_models: list[str] = []

    def add_line(self, line: str) -> LLMExchange | None:
        """Read one callback line; the exchange it adds, whole, or None."""
        if not line.strip():
            return None
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(record, dict):
            return None
        if record.get("event") in {
            "request_start",
            "response_start",
            "request_end",
            "upstream_start",
            "upstream_end",
        }:
            return None
        trajectory = self.trajectory
        if record.get("event") == "refused":
            # The gateway refused the request before any provider call: not an
            # exchange, so replay, usage and token coverage never see it. A
            # refusal for the trial's budget keeps what it counted.
            trajectory.metadata.setdefault("refused_requests", []).append(
                {
                    key: record.get(key)
                    for key in (
                        "request_model",
                        "call_type",
                        "rule",
                        "logged_at",
                        *(
                            ("budget", "limits", "used", "refused")
                            if record.get("rule") == "trial-budget"
                            else ()
                        ),
                    )
                }
            )
            return None
        raw_response = record.get("response")
        if record.get("event") == "success" and isinstance(raw_response, dict):
            # The model the provider's response names, read before
            # _record_response_body fills in the requested one where the
            # response named none (a fill-in never counts as reported).
            add_model(self._reported_models, raw_response.get("model"))
        request = (
            record.get("request") if isinstance(record.get("request"), dict) else {}
        )
        request_body = request.get("body")
        request_body = request_body if isinstance(request_body, dict) else {}
        response_body = _record_response_body(record)
        if record.get("token_capture"):
            response_body = strip_captured_token_ids(response_body)
        if record.get("event") == "success":
            status = 200
        else:
            status = _provider_failure_status_from_failure_record(record) or 500
        exchange = LLMExchange(
            request=LLMRequest(
                timestamp=_parse_time(record.get("start_time")),
                method=str(request.get("method") or "POST"),
                path=str(request.get("path") or ""),
                body=request_body,
            ),
            response=LLMResponse(
                timestamp=_parse_time(record.get("end_time")),
                status_code=status,
                body=response_body,
            ),
            duration_ms=float(record.get("duration_ms") or 0.0),
            metadata=_exchange_metadata(
                record,
                request_body=request_body,
                agent_name=self._agent_name,
            ),
        )
        trajectory.exchanges.append(
            exchange if self._keep is None else self._keep(exchange)
        )
        cost = record.get("response_cost")
        if isinstance(cost, int | float):
            self._total_cost += float(cost)
            self._saw_cost = True
        return exchange

    def finish(self) -> Trajectory:
        """The trajectory of every line read so far."""
        trajectory = self.trajectory
        trajectory.finished_at = datetime.now()
        if self._saw_cost:
            trajectory.metadata["cost_usd"] = round(self._total_cost, 10)
        if self._reported_models:
            trajectory.metadata["reported_models"] = self._reported_models
        return trajectory


def extract_usage_from_trajectory(
    trajectory: Trajectory | None,
    *,
    fallback_model: str | None = None,
) -> dict[str, Any]:
    """Return aggregate usage metrics from a LiteLLM-imported trajectory.

    Token counts come from the provider response; cost is whatever LiteLLM
    computed (``litellm.completion_cost`` / per-deployment ``input_cost_per_token``
    for custom models), summed by the callback importer. BenchFlow performs no
    cost calculation of its own — LiteLLM is the single source of truth.
    """
    del fallback_model  # no longer needed; kept for call-site compatibility
    if trajectory is None or not trajectory.exchanges:
        return usage_unavailable()
    if not trajectory.has_provider_usage:
        return usage_unavailable()

    cost_usd = trajectory.total_cost_usd
    return {
        "n_input_tokens": trajectory.total_input_tokens,
        "n_output_tokens": trajectory.total_output_tokens,
        "n_cache_read_tokens": trajectory.total_cache_read_tokens,
        "n_cache_creation_tokens": trajectory.total_cache_creation_tokens,
        "total_tokens": trajectory.total_provider_tokens,
        "cost_usd": cost_usd,
        "usage_source": "provider_response",
        "price_source": "litellm" if cost_usd is not None else None,
    }
