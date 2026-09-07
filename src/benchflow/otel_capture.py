"""Capture per-request token usage from Claude Code's OpenTelemetry export.

In OAuth / subscription mode the agent (Claude Code via ``claude-agent-acp``)
talks to the Anthropic API directly, so the LiteLLM proxy sees nothing and
``llm_trajectory.jsonl`` is empty. Claude Code can, however, export telemetry:
with ``CLAUDE_CODE_ENABLE_TELEMETRY=1`` and ``OTEL_LOGS_EXPORTER=otlp`` it emits
one ``claude_code.api_request`` log record per Anthropic API call carrying
``model``, ``input_tokens`` (uncached), ``output_tokens``, ``cache_read_tokens``,
``cache_creation_tokens``, ``cost_usd`` (Claude Code's own list-price estimate),
``duration_ms``, ``request_id`` and ``query_source`` (main / subagent / sdk …).
Verified against Claude Code 2.1.263 on 2026-09-07 with the OTLP http/json
protocol.

:class:`OtelUsageReceiver` is a minimal OTLP/HTTP (JSON) endpoint that accepts
``POST /v1/logs`` (and acknowledges ``/v1/metrics`` / ``/v1/traces``), extracts
the ``api_request`` / ``api_error`` events and appends one JSON line per event to
``trajectory/otel_usage.jsonl``. :func:`agent_env` returns the environment the
agent container needs to send its telemetry there. :mod:`benchflow.cost` prices
the file post hoc.

Runtime capture is deliberately independent of Claude Code's ``cost_usd``: the
tokens are the record, cost is derived from :mod:`benchflow.pricing`.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

API_REQUEST_EVENT = "claude_code.api_request"
API_ERROR_EVENT = "claude_code.api_error"
_TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)
_COPY_KEYS = (
    "model",
    "cost_usd",
    "duration_ms",
    "request_id",
    "client_request_id",
    "query_source",
    "effort",
    "speed",
    "session.id",
    "prompt.id",
    "event.sequence",
    "event.timestamp",
    "status_code",
    "error",
    "attempt",
)


def agent_env(
    endpoint: str, *, metrics: bool = False, export_interval_ms: int = 1000
) -> dict[str, str]:
    """Environment variables that make Claude Code export usage to ``endpoint``.

    ``endpoint`` is the receiver's base URL as seen from the agent, e.g.
    ``http://host.docker.internal:4318``. Metrics are off by default (the log
    events already carry every token count); turn them on for dashboards.
    """
    env = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "otlp" if metrics else "none",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
        "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint.rstrip("/"),
        "OTEL_LOGS_EXPORT_INTERVAL": str(export_interval_ms),
    }
    if metrics:
        env["OTEL_METRIC_EXPORT_INTERVAL"] = str(max(export_interval_ms, 1000))
    return env


def _attr_value(value: dict[str, Any]) -> Any:
    """Unwrap an OTLP ``AnyValue`` (``{"stringValue": …}`` etc.)."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_attr_value(v) for v in (value["arrayValue"] or {}).get("values", [])]
    if "kvlistValue" in value:
        return {
            kv["key"]: _attr_value(kv["value"])
            for kv in (value["kvlistValue"] or {}).get("values", [])
        }
    return value


def _attrs(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        key = item.get("key")
        if isinstance(key, str) and isinstance(item.get("value"), dict):
            out[key] = _attr_value(item["value"])
    return out


def _event_name(record: dict[str, Any], attrs: dict[str, Any]) -> str | None:
    body = record.get("body")
    if isinstance(body, dict) and isinstance(body.get("stringValue"), str):
        return body["stringValue"]
    name = record.get("eventName") or attrs.get("event.name")
    if isinstance(name, str):
        return name if name.startswith("claude_code.") else f"claude_code.{name}"
    return None


def parse_otlp_logs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract usage records from an OTLP/JSON ``ExportLogsServiceRequest``."""
    records: list[dict[str, Any]] = []
    for rl in payload.get("resourceLogs", []) or []:
        resource_attrs = _attrs((rl.get("resource") or {}).get("attributes"))
        for sl in rl.get("scopeLogs", []) or []:
            for lr in sl.get("logRecords", []) or []:
                attrs = _attrs(lr.get("attributes"))
                name = _event_name(lr, attrs)
                if name not in (API_REQUEST_EVENT, API_ERROR_EVENT):
                    continue
                ts_ns = lr.get("timeUnixNano") or lr.get("observedTimeUnixNano")
                try:
                    ts = datetime.fromtimestamp(int(ts_ns) / 1e9, tz=UTC).isoformat(
                        timespec="milliseconds"
                    )
                except (TypeError, ValueError):
                    ts = attrs.get("event.timestamp")
                rec: dict[str, Any] = {
                    "event": name.removeprefix("claude_code."),
                    "timestamp": ts,
                }
                for key in _TOKEN_KEYS:
                    if key in attrs:
                        try:
                            rec[key] = int(attrs[key])
                        except (TypeError, ValueError):
                            rec[key] = 0
                for key in _COPY_KEYS:
                    if key in attrs:
                        rec[key.replace(".", "_")] = attrs[key]
                if "service.name" in resource_attrs:
                    rec["service_name"] = resource_attrs["service.name"]
                records.append(rec)
    return records


class OtelUsageReceiver:
    """Threaded OTLP/HTTP JSON receiver writing ``otel_usage.jsonl``.

    >>> rx = OtelUsageReceiver(out_path)          # doctest: +SKIP
    >>> rx.start(); env = agent_env(rx.endpoint)   # doctest: +SKIP
    >>> ...run the agent with env...; rx.stop()    # doctest: +SKIP
    """

    def __init__(
        self, out_path: str | Path, *, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        self.out_path = Path(out_path)
        self.host, self._port = host, port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.n_records = 0
        self.n_requests = 0

    # -- lifecycle
    def start(self) -> OtelUsageReceiver:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                if self.headers.get("Content-Encoding", "").lower() == "gzip":
                    try:
                        body = gzip.decompress(body)
                    except OSError:
                        body = b""
                receiver._handle(self.path, body, self.headers.get("Content-Type", ""))
                out = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, format: str, *args: Any) -> None:  # stdlib signature
                return  # silence default request logging

        self._server = ThreadingHTTPServer((self.host, self._port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="otel-usage-receiver", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> OtelUsageReceiver:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("receiver not started")
        return int(self._server.server_address[1])

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- ingestion
    def _handle(self, path: str, body: bytes, content_type: str) -> None:
        self.n_requests += 1
        if not path.rstrip("/").endswith("/v1/logs"):
            return  # metrics / traces are acknowledged and dropped
        if "json" not in content_type.lower() and not body.lstrip().startswith(b"{"):
            logger.warning(
                "otel receiver: non-JSON logs payload (%s) ignored; use OTEL_EXPORTER_OTLP_PROTOCOL=http/json",
                content_type,
            )
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning("otel receiver: unparseable logs payload: %s", exc)
            return
        records = parse_otlp_logs(payload) if isinstance(payload, dict) else []
        if not records:
            return
        with self._lock, self.out_path.open("a") as fh:
            for rec in records:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            self.n_records += len(records)
