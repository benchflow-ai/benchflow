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
    """Return the Python module written next to LiteLLM config.yaml."""
    return r"""
from __future__ import annotations

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


class BenchFlowLiteLLMLogger(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if isinstance(data, dict) and data.get("messages") is not None and "input" in data:
            cleaned = dict(data)
            cleaned.pop("input", None)
            return cleaned
        return None

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
        return {
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

    async def async_pre_call_hook(
        self, user_api_key_dict, cache, data, call_type
    ):
        if not isinstance(data, dict):
            return None

        _gate_opencode_skill_catalog(data)

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
        if isinstance(tools, list):
            kept = [
                t
                for t in tools
                if not isinstance(t, dict) or t.get("type", "function") == "function"
            ]
            if len(kept) != len(tools):
                if cleaned is data:
                    cleaned = dict(data)
                cleaned["tools"] = kept
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
        record.update(
            {
                "event": "success",
                "response": response,
                "usage": _usage_from_response(response_obj),
                "response_cost": cost,
            }
        )
        self._write(record)

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
            status = _provider_failure_status_from_failure_record(record) or 500
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
