"""A trial as OpenTelemetry spans: one for the trial, one per model call.

Every trial writes ``trajectory/otel-spans.json``, an OTLP/JSON
``ExportTraceServiceRequest`` any OpenTelemetry collector reads (its
``otlpjsonfile`` receiver) or takes as the body of ``POST /v1/traces``:

- the trial's span (``trial <task>``): from ``result.json``'s start and finish,
  with the task, trial, agent, model, reward, outcome and error, and each
  phase's seconds from ``timing.json`` (``benchflow.timing.*``; the phases'
  own start times are not recorded, so they are attributes, not spans);
- one child span per model call (``chat <model>``, kind CLIENT), from the
  trial's model-call records (:mod:`benchflow.trajectories.model_calls`),
  with the OpenTelemetry GenAI attributes (``gen_ai.request.model``,
  ``gen_ai.response.finish_reasons``, ``gen_ai.usage.input_tokens``, ...) and
  BenchFlow's (``benchflow.model_call.source`` names where the record came
  from; a figure its source could not see is left out).

IDs come from the trial folder and its start time, so writing a trial again
gives the same spans. With ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` or
``OTEL_EXPORTER_OTLP_ENDPOINT`` set, the spans are also sent there as each
trial finishes (OTLP/HTTP JSON, ``OTEL_EXPORTER_OTLP_HEADERS`` honoured); a
failed send is logged and never fails the trial.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from benchflow._utils.disk_space import write_text_atomic
from benchflow.trajectories.model_calls import CALLS_FILE, model_calls

logger = logging.getLogger(__name__)

SPANS_FILE = Path("trajectory") / "otel-spans.json"
SCOPE = "benchflow"
_CLIENT, _INTERNAL = 3, 1
_OK, _ERROR = 1, 2


def _value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        values = [_value(v) for v in value]
        return {"arrayValue": {"values": [v for v in values if v is not None]}}
    return None


def _attributes(pairs: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for key, value in pairs.items():
        encoded = _value(value)
        if value is not None and encoded is not None:
            out.append({"key": key, "value": encoded})
    return out


def _nanos(when: datetime) -> str:
    return str(int(when.timestamp() * 1_000_000_000))


def _when(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _id(seed: str, nbytes: int) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[: nbytes * 2]


def _version() -> str | None:
    try:
        from importlib.metadata import version

        return version("benchflow")
    except Exception:
        return None


def _call_span(call: dict[str, Any], trace: str, parent: str) -> dict[str, Any] | None:
    start = _when(call.get("started_at"))
    if start is None:
        return None
    duration = call.get("duration_ms")
    end = (
        start + timedelta(milliseconds=duration)
        if isinstance(duration, (int, float))
        else start
    )
    status = call.get("http_status")
    model = call.get("model_requested") or call.get("model_served") or "unknown"
    finish = call.get("finish_reason")
    failed = isinstance(status, int) and status >= 400
    span: dict[str, Any] = {
        "traceId": trace,
        "spanId": _id(f"{trace}:call:{call.get('index')}", 8),
        "parentSpanId": parent,
        "name": f"chat {model}",
        "kind": _CLIENT,
        "startTimeUnixNano": _nanos(start),
        "endTimeUnixNano": _nanos(end),
        "attributes": _attributes(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": call.get("model_requested"),
                "gen_ai.response.model": call.get("model_served"),
                "gen_ai.request.max_tokens": call.get("max_output_tokens"),
                "gen_ai.request.temperature": call.get("temperature"),
                "gen_ai.request.top_p": call.get("top_p"),
                "gen_ai.response.finish_reasons": [finish] if finish else None,
                "gen_ai.usage.input_tokens": call.get("input_tokens"),
                "gen_ai.usage.output_tokens": call.get("output_tokens"),
                "gen_ai.usage.cache_read.input_tokens": call.get("cache_read_tokens"),
                "gen_ai.usage.cache_creation.input_tokens": call.get(
                    "cache_write_tokens"
                ),
                "http.response.status_code": status,
                "error.type": str(status) if failed else None,
                "benchflow.model_call.source": call.get("source"),
                "benchflow.model_call.index": call.get("index"),
                "benchflow.model_call.purpose": call.get("purpose"),
                "benchflow.model_call.effort": call.get("effort"),
                "benchflow.model_call.thinking": call.get("thinking"),
                "benchflow.model_call.reasoning_tokens": call.get("reasoning_tokens"),
                "benchflow.model_call.truncated": call.get("truncated"),
                "benchflow.model_call.refused": call.get("refused"),
                "benchflow.model_call.cost_usd": call.get("cost_usd"),
                "benchflow.model_call.subagent": call.get("subagent"),
            }
        ),
    }
    if failed:
        span["status"] = {"code": _ERROR, "message": f"HTTP {status}"}
    return span


def _calls(trial: Path) -> list[dict[str, Any]]:
    """The records teardown wrote (``trajectory/model-calls.jsonl``), else
    read again from their source: a large proxy log is read once."""
    try:
        lines = (trial / "trajectory" / CALLS_FILE).read_text().splitlines()
    except OSError:
        return model_calls(trial)
    calls = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            calls.append(record)
    return calls


def trial_spans(trial: Path) -> dict[str, Any] | None:
    """The trial's spans as an OTLP/JSON request, or None without a result."""
    trial = Path(trial)
    try:
        result = json.loads((trial / "result.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict):
        return None
    start = _when(result.get("started_at"))
    if start is None:
        return None
    end = _when(result.get("finished_at")) or start
    trace = _id(f"benchflow:{trial.name}:{result.get('started_at')}", 16)
    root = _id(f"{trace}:trial", 8)
    try:
        timing = json.loads((trial / "timing.json").read_text())
    except (OSError, ValueError):
        timing = {}
    rewards = result.get("rewards") if isinstance(result.get("rewards"), dict) else {}
    error = result.get("error") or result.get("verifier_error")
    task = result.get("task_name") or trial.name
    trial_span: dict[str, Any] = {
        "traceId": trace,
        "spanId": root,
        "name": f"trial {task}",
        "kind": _INTERNAL,
        "startTimeUnixNano": _nanos(start),
        "endTimeUnixNano": _nanos(end),
        "attributes": _attributes(
            {
                "benchflow.task.name": task,
                "benchflow.trial.name": result.get("rollout_name") or trial.name,
                "benchflow.agent.name": result.get("agent_name") or result.get("agent"),
                "gen_ai.request.model": result.get("model"),
                "benchflow.reward": rewards.get("reward"),
                "benchflow.error": error,
                "benchflow.error.category": result.get("error_category")
                or result.get("verifier_error_category"),
                **{
                    f"benchflow.timing.{phase}_s": float(seconds)
                    for phase, seconds in (
                        timing.items() if isinstance(timing, dict) else []
                    )
                    if isinstance(seconds, (int, float))
                    and not isinstance(seconds, bool)
                },
            }
        ),
        "status": {"code": _ERROR, "message": str(error)[:500]}
        if error
        else {"code": _OK},
    }
    spans = [trial_span]
    for call in _calls(trial):
        span = _call_span(call, trace, root)
        if span is not None:
            spans.append(span)
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _attributes(
                        {"service.name": "benchflow", "service.version": _version()}
                    )
                },
                "scopeSpans": [
                    {
                        "scope": {"name": SCOPE, "version": _version() or ""},
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _endpoint() -> tuple[str, dict[str, str]] | None:
    traces = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not traces and not base:
        return None
    protocol = (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL")
        or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL")
        or "http/json"
    ).strip()
    if protocol != "http/json":
        logger.info(
            "Trial spans not sent: BenchFlow sends OTLP as http/json, not %s", protocol
        )
        return None
    url = traces or base.rstrip("/") + "/v1/traces"
    headers = {"Content-Type": "application/json"}
    for variable in ("OTEL_EXPORTER_OTLP_HEADERS", "OTEL_EXPORTER_OTLP_TRACES_HEADERS"):
        for pair in os.environ.get(variable, "").split(","):
            key, sep, value = pair.partition("=")
            if sep and key.strip():
                headers[key.strip()] = urllib.parse.unquote(value.strip())
    return url, headers


def send_spans(payload: dict[str, Any], url: str, headers: dict[str, str]) -> int:
    """POST *payload* to an OTLP/HTTP traces endpoint; the HTTP status."""
    timeout = float(os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT") or 10000) / 1000
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status


def emit_trial_spans(trial: Path) -> dict[str, Any] | None:
    """Write the trial's spans, and send them where the environment says; never raises."""
    try:
        payload = trial_spans(trial)
        if payload is None:
            return None
        target = Path(trial) / SPANS_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(target, json.dumps(payload, separators=(",", ":")))
        endpoint = _endpoint()
        if endpoint is not None:
            try:
                send_spans(payload, *endpoint)
            except Exception as exc:
                logger.warning("Trial spans not sent to %s: %s", endpoint[0], exc)
        return payload
    except Exception as exc:
        logger.info("Trial spans not written: %s", exc)
        return None
