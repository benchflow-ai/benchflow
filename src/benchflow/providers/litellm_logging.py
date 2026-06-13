"""LiteLLM callback logger source and callback-log import helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)
from benchflow.usage_tracking import usage_unavailable

_PROVIDER_AUTH_STATUS_CODES = (401, 403)
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
_PROVIDER_AUTH_STATUS_RE = re.compile(r"\b(401|403)\b")
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


def callback_module_source() -> str:
    """Return the Python module written next to LiteLLM config.yaml."""
    return r"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


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


# Map an upstream provider-model prefix (the LiteLLM ``litellm_params['model']``,
# e.g. ``gemini/...``, ``anthropic/...``, ``openai/...``) to the wire protocol it
# speaks on the wire. GenerateContent (Gemini/Vertex) and Anthropic Messages are
# distinct from the OpenAI chat-completions protocol the agent emits; when they
# differ we record the provider-facing (translated) view additively.
def _upstream_wire_protocol(provider_model: Any) -> str:
    model = str(provider_model or "").lower()
    prefix = model.split("/", 1)[0] if "/" in model else ""
    if prefix in ("gemini", "vertex_ai", "vertex_ai_beta", "google"):
        return "generate_content"
    if prefix in ("anthropic", "anthropic_text"):
        return "anthropic_messages"
    return "openai_chat"


# Derive the upstream wire protocol from the *real* upstream URL LiteLLM dialed.
# This is the authoritative signal at success time: ``litellm_params['model']``
# is ``None`` and ``kwargs['model']`` is the bare gateway alias (no ``gemini/``
# prefix), but ``litellm_params['api_base']`` carries the true resource URL
# (e.g. ``.../v1beta/models/<model>:generateContent``). The ``:generateContent``
# / ``:streamGenerateContent`` action suffix and the ``/v1beta/models/`` segment
# unambiguously identify a GenerateContent backend; an ``/anthropic`` path or an
# anthropic host identifies an Anthropic Messages backend; everything else is the
# OpenAI chat-completions protocol. Returns ``None`` when ``api_base`` carries no
# usable signal so callers can fall back to the model-name classifier.
def _wire_protocol_from_api_base(api_base: Any) -> str | None:
    base = str(api_base or "").lower()
    if not base:
        return None
    path = base.split("://", 1)[1] if "://" in base else base
    host = path.split("/", 1)[0]
    if (
        ":generatecontent" in base
        or ":streamgeneratecontent" in base
        or "/v1beta/models/" in base
        or "/v1beta1/models/" in base
        or "generativelanguage.googleapis.com" in host
    ):
        return "generate_content"
    if "/anthropic" in base or "anthropic.com" in host:
        return "anthropic_messages"
    return "openai_chat"


def _agent_facing_protocol(kwargs: dict[str, Any]) -> str:
    return (
        "anthropic_messages"
        if kwargs.get("call_type") == "anthropic_messages"
        else "openai_chat"
    )


def _upstream_request_path(api_base: Any, wire_protocol: str, streaming: Any) -> str:
    # GenerateContent upstreams append the action to the model resource as a
    # ``:method`` suffix; surface the *true* path (``:streamGenerateContent`` for
    # streaming, else ``:generateContent``) rather than the hardcoded OpenAI path.
    if wire_protocol == "generate_content":
        # The success-time ``api_base`` already carries the real action suffix
        # (``.../models/<model>:generateContent``); echo it back verbatim when
        # present so the path reflects the URL actually dialed, else fall back to
        # the streaming/non-streaming default.
        base = str(api_base or "")
        lowered = base.lower()
        if ":streamgeneratecontent" in lowered:
            return ":streamGenerateContent"
        if ":generatecontent" in lowered:
            return ":generateContent"
        return ":streamGenerateContent" if streaming else ":generateContent"
    if wire_protocol == "anthropic_messages":
        return "/v1/messages"
    base = str(api_base or "")
    if "://" in base:
        rest = base.split("://", 1)[1]
        path = rest[rest.find("/") :] if "/" in rest else ""
        if path:
            return path.split("?", 1)[0]
    return "/v1/chat/completions"


def _provider_facing_view(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    # Provider-facing (gateway-translated) capture, or None when same-protocol.
    #
    # Returns the LiteLLM-translated upstream request (complete_input_dict), the
    # real upstream URL (litellm_params['api_base']) and true path (e.g.
    # ':generateContent'). Emitted ONLY when the upstream wire protocol differs
    # from the agent-facing protocol (cross-protocol). For a same-protocol call
    # this returns None so the record stays byte-unchanged. The raw provider
    # response is attached by the success handler (original_response).
    litellm_params = kwargs.get("litellm_params") or {}
    additional_args = kwargs.get("additional_args") or {}
    api_base = litellm_params.get("api_base") or additional_args.get("api_base")
    # The real upstream URL is the authoritative protocol signal at success time:
    # ``litellm_params['model']`` is ``None`` and ``kwargs['model']`` is the bare
    # gateway alias (no ``gemini/`` prefix), so deriving from the model name alone
    # misclassifies a GenerateContent backend as OpenAI and silently drops the
    # cross-protocol block. Prefer the ``api_base``-derived protocol and fall back
    # to the model-name classifier only when ``api_base`` carries no usable signal.
    upstream_protocol = _wire_protocol_from_api_base(api_base)
    if upstream_protocol is None:
        provider_model = litellm_params.get("model") or kwargs.get("model")
        upstream_protocol = _upstream_wire_protocol(provider_model)
    if upstream_protocol == _agent_facing_protocol(kwargs):
        return None

    translated = additional_args.get("complete_input_dict")
    if translated is None and not api_base:
        # Nothing translated to record (e.g. the proxy never reached pre-call).
        return None

    optional_params = kwargs.get("optional_params") or {}
    streaming = optional_params.get("stream") or kwargs.get("stream")
    return {
        "protocol": upstream_protocol,
        "request": {
            "method": "POST",
            "path": _upstream_request_path(api_base, upstream_protocol, streaming),
            "url": api_base or None,
            "body": translated,
        },
    }


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
        for key in ("reasoning_effort", "thinking", "output_config"):
            value = optional_params.get(key)
            if value is None:
                value = kwargs.get(key)
            if value is None:
                value = litellm_params.get(key)
            if value is not None:
                request_body[key] = value
        request_body = {k: v for k, v in request_body.items() if v is not None}
        record = {
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
                "path": "/v1/messages" if kwargs.get("call_type") == "anthropic_messages" else "/v1/chat/completions",
                "body": request_body,
            },
            "start_time": _iso(start_time),
            "end_time": _iso(end_time),
            "duration_ms": max((getattr(end_time, "timestamp", lambda: time.time())() - getattr(start_time, "timestamp", lambda: time.time())()) * 1000, 0),
        }
        # Additively record the gateway-translated provider-facing request when
        # the upstream speaks a different wire protocol (e.g. an OpenAI-protocol
        # agent routed to a Gemini GenerateContent backend). The agent-facing
        # OpenAI view above is preserved unchanged; same-protocol calls add
        # nothing, keeping the record byte-identical.
        upstream = _provider_facing_view(kwargs)
        if upstream is not None:
            record["upstream"] = upstream
        return record

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
        record.update(
            {
                "event": "success",
                "response": response,
                "usage": _usage_from_response(response_obj),
                "response_cost": cost,
            }
        )
        # When the call crossed protocols, attach the RAW provider response
        # (LiteLLM ``original_response``) to the provider-facing block recorded
        # in _base_record. The agent-facing ``response`` above is untouched.
        upstream = record.get("upstream")
        if isinstance(upstream, dict):
            original = kwargs.get("original_response")
            if isinstance(original, str):
                try:
                    original = json.loads(original)
                except (ValueError, TypeError):
                    pass
            if original is not None:
                upstream["response"] = original
        self._write(record)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        record = self._base_record(kwargs, start_time, end_time)
        record.update(
            {
                "event": "failure",
                "response": _jsonable(response_obj),
                "error": {
                    "type": type(response_obj).__name__,
                    "message": str(response_obj),
                    "traceback": traceback.format_exc()[-2000:],
                },
            }
        )
        self._write(record)


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


def _coerce_auth_status(value: Any) -> int | None:
    if isinstance(value, int) and value in _PROVIDER_AUTH_STATUS_CODES:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            status = int(stripped)
            if status in _PROVIDER_AUTH_STATUS_CODES:
                return status
    return None


def _explicit_auth_status(value: Any) -> int | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).lower() in _STATUS_KEYS:
                status = _coerce_auth_status(nested)
                if status is not None:
                    return status
        for nested in value.values():
            status = _explicit_auth_status(nested)
            if status is not None:
                return status
    elif isinstance(value, list | tuple):
        for nested in value:
            status = _explicit_auth_status(nested)
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


def _provider_auth_status_from_failure_record(record: dict[str, Any]) -> int | None:
    """Return a sanitized provider auth status from a LiteLLM failure record."""
    if record.get("event") != "failure":
        return None
    failure_payload = {
        "error": record.get("error"),
        "response": record.get("response"),
    }
    status = _explicit_auth_status(failure_payload)
    if status is not None:
        return status

    text = _flatten_failure_text(failure_payload)
    if not _PROVIDER_AUTH_HINT_RE.search(text):
        return None
    match = _PROVIDER_AUTH_STATUS_RE.search(text)
    if match is None:
        return None
    return int(match.group(1))


def trajectory_from_litellm_callback_log(
    text: str,
    *,
    session_id: str,
    agent_name: str,
) -> Trajectory:
    """Convert LiteLLM callback JSONL into BenchFlow's trajectory schema."""
    trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
    total_cost = 0.0
    saw_cost = False
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        request = (
            record.get("request") if isinstance(record.get("request"), dict) else {}
        )
        request_body = request.get("body")
        request_body = request_body if isinstance(request_body, dict) else {}
        response_body = _record_response_body(record)
        if record.get("event") == "success":
            status = 200
        else:
            status = _provider_auth_status_from_failure_record(record) or 500
        # Carry the gateway-translated provider-facing block (translated request
        # body, real upstream URL/path, raw provider response) through into the
        # persisted Trajectory so the cross-protocol capture survives to
        # ``to_jsonl`` — where it is redacted alongside the rest of the exchange.
        # Absent for same-protocol records (None keeps the exchange unchanged).
        upstream = record.get("upstream")
        upstream = upstream if isinstance(upstream, dict) else None
        trajectory.exchanges.append(
            LLMExchange(
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
                upstream=upstream,
            )
        )
        cost = record.get("response_cost")
        if isinstance(cost, int | float):
            total_cost += float(cost)
            saw_cost = True
    trajectory.finished_at = datetime.now()
    if saw_cost:
        trajectory.metadata["cost_usd"] = round(total_cost, 10)
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
