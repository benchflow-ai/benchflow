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
_PROVIDER_FAILURE_STATUS_CODES = (*_PROVIDER_AUTH_STATUS_CODES, 429, 503)
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


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("output")
                if isinstance(text, list):
                    text = _content_to_text(text)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _normalize_tool_call(call: dict[str, Any], index: int = 0) -> dict[str, Any]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name") or call.get("name") or "tool"
    arguments = function.get("arguments", call.get("arguments", {}))
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {}, sort_keys=True)
    return {
        "id": str(call.get("id") or call.get("tool_call_id") or call.get("call_id") or f"call_{index:06d}"),
        "type": "function",
        "function": {"name": str(name), "arguments": arguments},
    }


def _normalize_role(role: Any) -> str:
    if role == "developer":
        return "system"
    if role == "model":
        return "assistant"
    return str(role or "user")


def _normalize_message(message: dict[str, Any], index: int = 0) -> dict[str, Any] | None:
    mtype = message.get("type")
    if mtype == "reasoning":
        return None
    if mtype == "function_call":
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [_normalize_tool_call(message, index)],
        }
    if mtype == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": str(message.get("call_id") or message.get("id") or ""),
            "content": _content_to_text(message.get("output")),
        }
    out = {"role": _normalize_role(message.get("role")), "content": _content_to_text(message.get("content"))}
    tool_calls = message.get("tool_calls")
    if tool_calls is None and isinstance(message.get("function_call"), dict):
        tool_calls = [message["function_call"]]
    if isinstance(tool_calls, list) and tool_calls:
        out["tool_calls"] = [
            _normalize_tool_call(call, i) for i, call in enumerate(tool_calls) if isinstance(call, dict)
        ]
    if out["role"] == "tool" and message.get("tool_call_id"):
        out["tool_call_id"] = str(message["tool_call_id"])
    return out


def _messages_from_request_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": _content_to_text(instructions)})
    raw = body.get("messages") if isinstance(body.get("messages"), list) else body.get("input")
    if isinstance(raw, str):
        messages.append({"role": "user", "content": raw})
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                normalized = _normalize_message(item, idx)
                if normalized is not None:
                    messages.append(normalized)
    return messages


def _assistant_from_response_body(body: dict[str, Any]) -> dict[str, Any] | None:
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and isinstance(first.get("message"), dict):
            return _normalize_message(first["message"], 0)
    if isinstance(body.get("message"), dict):
        return _normalize_message(body["message"], 0)
    output = body.get("output")
    if isinstance(output, list):
        texts = []
        tool_calls = []
        for idx, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                text = _content_to_text(item.get("content"))
                if text:
                    texts.append(text)
            elif item.get("type") in {"function_call", "tool_call"}:
                tool_calls.append(_normalize_tool_call(item, idx))
        if texts or tool_calls:
            out = {"role": "assistant", "content": "\n".join(texts)}
            if tool_calls:
                out["tool_calls"] = tool_calls
            return out
    if body.get("content"):
        return {"role": "assistant", "content": _content_to_text(body.get("content"))}
    return None


def _tool_defs_from_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tools = body.get("tools") or body.get("tool_defs") or []
    if not isinstance(raw_tools, list):
        return []
    tools = []
    for item in raw_tools:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function" and isinstance(item.get("function"), dict):
            function = dict(item["function"])
        elif isinstance(item.get("function"), dict):
            function = dict(item["function"])
        else:
            function = {
                "name": item.get("name"),
                "description": item.get("description", ""),
                "parameters": item.get("parameters", {"type": "object", "properties": {}}),
            }
        if not function.get("name"):
            continue
        function.setdefault("description", "")
        function.setdefault("parameters", {"type": "object", "properties": {}})
        tools.append({"type": "function", "function": function})
    return tools


def _verifiers_tracking(request_body: dict[str, Any], response_body: dict[str, Any]) -> dict[str, Any]:
    prompt = _messages_from_request_body(request_body)
    assistant = _assistant_from_response_body(response_body)
    tool_defs = _tool_defs_from_body(request_body)
    if assistant is None:
        return {"step": None, "tool_defs": tool_defs}
    step = {
        "prompt": prompt,
        "completion": [assistant],
        "response": response_body,
        "tokens": None,
        "reward": None,
        "advantage": None,
        "is_truncated": bool(response_body.get("incomplete_details") or response_body.get("truncation")),
        "trajectory_id": "",
        "extras": {"source": "litellm_callback"},
    }
    return {"step": step, "tool_defs": tool_defs}


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
        tracking = _verifiers_tracking(record["request"]["body"], response if isinstance(response, dict) else {})
        record["verifiers_step"] = tracking["step"]
        record["verifiers_tool_defs"] = tracking["tool_defs"]
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


def _text_provider_failure_status(text: str) -> int | None:
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
    status = _explicit_provider_failure_status(failure_payload)
    if status is not None:
        return status

    text = _flatten_failure_text(failure_payload)
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
                verifiers_step=record.get("verifiers_step")
                if isinstance(record.get("verifiers_step"), dict)
                else None,
                verifiers_tool_defs=record.get("verifiers_tool_defs")
                if isinstance(record.get("verifiers_tool_defs"), list)
                else [],
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
