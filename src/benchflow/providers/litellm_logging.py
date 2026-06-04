"""LiteLLM callback logger source and callback-log import helpers."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from benchflow.agents.providers import strip_provider_prefix
from benchflow.trajectories.pricing import PRICING_USD_PER_MTOK, PricingEntry
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)


def callback_module_source() -> str:
    """Return the Python module written next to LiteLLM config.yaml."""
    return r'''
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
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


class BenchFlowLiteLLMLogger(CustomLogger):
    def _write(self, payload: dict[str, Any]) -> None:
        path = os.environ.get("BENCHFLOW_LITELLM_LOG_PATH")
        if not path:
            return
        payload["logged_at"] = datetime.utcnow().isoformat()
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
        cost = None
        try:
            cost = litellm.completion_cost(completion_response=response_obj)
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
'''


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
        request = record.get("request") if isinstance(record.get("request"), dict) else {}
        request_body = request.get("body")
        request_body = request_body if isinstance(request_body, dict) else {}
        response_body = _record_response_body(record)
        status = 200 if record.get("event") == "success" else 500
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


def usage_unavailable() -> dict[str, Any]:
    return {
        "n_input_tokens": 0,
        "n_output_tokens": 0,
        "n_cache_read_tokens": 0,
        "n_cache_creation_tokens": 0,
        "total_tokens": 0,
        "cost_usd": None,
        "usage_source": "unavailable",
        "price_source": None,
    }


def _pricing_model_key(model: str) -> str:
    bare = strip_provider_prefix(model).lower()
    bare = bare.removeprefix("models/")
    marker = "anthropic."
    if marker in bare:
        bare = bare.split(marker, 1)[1]
    for prefix in ("bedrock/", "anthropic/", "openai/", "gemini/", "azure/", "azure_ai/"):
        bare = bare.removeprefix(prefix)
    return bare


def _pricing_for_model(model: str | None) -> PricingEntry | None:
    if not model:
        return None
    bare = _pricing_model_key(model)
    for prefix, pricing in PRICING_USD_PER_MTOK.items():
        if bare.startswith(prefix):
            return pricing
    return None


def _estimate_cost_usd(
    *,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> float | None:
    pricing = _pricing_for_model(model)
    if pricing is None:
        return None
    priced_input_tokens = max(
        input_tokens - cache_read_tokens - cache_creation_tokens,
        0,
    )
    cost = (
        priced_input_tokens * pricing.input
        + output_tokens * pricing.output
        + cache_read_tokens * pricing.cache_read
        + cache_creation_tokens * pricing.cache_creation
    ) / 1_000_000
    return round(cost, 10)


def _model_from_trajectory(trajectory: Trajectory, fallback: str | None) -> str | None:
    for exchange in trajectory.exchanges:
        for payload in (exchange.response.body, exchange.request.body):
            model = payload.get("model") if isinstance(payload, dict) else None
            if isinstance(model, str) and model:
                return model
    return fallback


def extract_usage_from_trajectory(
    trajectory: Trajectory | None,
    *,
    fallback_model: str | None,
) -> dict[str, Any]:
    """Return aggregate usage metrics from a LiteLLM-imported trajectory."""
    if trajectory is None or not trajectory.exchanges:
        return usage_unavailable()
    if not trajectory.has_provider_usage:
        return usage_unavailable()

    input_tokens = trajectory.total_input_tokens
    output_tokens = trajectory.total_output_tokens
    cache_read_tokens = trajectory.total_cache_read_tokens
    cache_creation_tokens = trajectory.total_cache_creation_tokens
    total_tokens = trajectory.total_provider_tokens
    model = _model_from_trajectory(trajectory, fallback_model)
    pricing = _pricing_for_model(model)
    cost_usd = trajectory.total_cost_usd
    if cost_usd is None:
        cost_usd = _estimate_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
    return {
        "n_input_tokens": input_tokens,
        "n_output_tokens": output_tokens,
        "n_cache_read_tokens": cache_read_tokens,
        "n_cache_creation_tokens": cache_creation_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "usage_source": "provider_response",
        "price_source": pricing.price_source if cost_usd is not None and pricing else None,
    }
