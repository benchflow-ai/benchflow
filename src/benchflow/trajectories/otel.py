"""OpenTelemetry OTLP collector — captures LLM call spans as trajectory."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from .types import LLMExchange, LLMRequest, LLMResponse, Trajectory

logger = logging.getLogger(__name__)

# Well-known OTel semantic convention attribute names for LLM/GenAI spans
_LLM_ATTRS = {
    "gen_ai.system": "system",
    "gen_ai.request.model": "model",
    "gen_ai.response.model": "response_model",
    "gen_ai.usage.input_tokens": "input_tokens",
    "gen_ai.usage.output_tokens": "output_tokens",
    "gen_ai.usage.total_tokens": "total_tokens",
    "gen_ai.request.max_tokens": "max_tokens",
    "gen_ai.request.temperature": "temperature",
    "gen_ai.response.finish_reasons": "finish_reasons",
    "gen_ai.prompt": "prompt",
    "gen_ai.completion": "completion",
    # Anthropic-specific
    "gen_ai.usage.cache_read_input_tokens": "cache_read_tokens",
    "gen_ai.usage.cache_creation_input_tokens": "cache_creation_tokens",
}


class OTelCollector:
    """Lightweight OTLP/HTTP receiver that captures GenAI spans as trajectory.

    Usage:
        collector = OTelCollector(session_id="trial-123")
        await collector.start()
        # Set agent's OTEL_EXPORTER_OTLP_ENDPOINT=http://host.docker.internal:{collector.port}
        # ... run agent ...
        trajectory = collector.trajectory
        await collector.stop()
    """

    def __init__(
        self,
        session_id: str = "",
        agent_name: str = "",
        host: str = "0.0.0.0",
        port: int = 0,
    ):
        self._host = host
        self._port = port
        self._trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
        self._server: asyncio.Server | None = None
        self._raw_spans: list[dict[str, Any]] = []

    @property
    def port(self) -> int:
        return self._port

    @property
    def endpoint(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def raw_spans(self) -> list[dict[str, Any]]:
        return self._raw_spans

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )
        sock = self._server.sockets[0]
        self._port = sock.getsockname()[1]
        logger.info(f"OTel collector listening on :{self._port}")

    async def stop(self) -> None:
        self._trajectory.finished_at = datetime.now()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info(
            f"OTel collector stopped. Captured {len(self._raw_spans)} spans, "
            f"{len(self._trajectory.exchanges)} LLM exchanges."
        )

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = await reader.readline()
            if not request_line:
                return
            parts = request_line.decode().strip().split(" ", 2)
            method = parts[0] if parts else "GET"
            path = parts[1] if len(parts) > 1 else "/"

            # Read headers
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()

            # Read body
            content_length = int(headers.get("content-length", "0"))
            body = b""
            if content_length > 0:
                body = await reader.readexactly(content_length)

            # Process OTLP traces
            if method == "POST" and "/v1/traces" in path:
                self._process_otlp_json(body, headers)
                # Return 200 OK
                resp = b'{"partialSuccess":{}}'
                writer.write(
                    b"HTTP/1.1 200 OK\r\n"
                    b"content-type: application/json\r\n"
                    + f"content-length: {len(resp)}\r\n\r\n".encode()
                    + resp
                )
            else:
                # Health check or unknown path
                writer.write(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\n\r\n")

            await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                logger.debug("Writer close failed during OTel connection teardown")

    def _process_otlp_json(self, body: bytes, headers: dict[str, str]) -> None:
        """Process OTLP/HTTP JSON payload."""
        content_type = headers.get("content-type", "")

        if "protobuf" in content_type:
            logger.warning("Protobuf OTLP not supported, use http/json protocol")
            return

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in OTLP payload")
            return

        # Extract spans from OTLP JSON structure
        for resource_spans in data.get("resourceSpans", []):
            for scope_spans in resource_spans.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    self._raw_spans.append(span)
                    self._maybe_extract_llm_exchange(span)

    def _maybe_extract_llm_exchange(self, span: dict[str, Any]) -> None:
        """Convert a GenAI span to an LLM exchange if it has the right attributes."""
        attrs = _parse_attributes(span.get("attributes", []))

        # Check if this is a GenAI/LLM span
        model = attrs.get("model") or attrs.get("response_model")
        if not model and not any(k.startswith("gen_ai.") for k in _raw_attr_keys(span)):
            return

        # Extract timing
        start_ns = int(span.get("startTimeUnixNano", 0))
        end_ns = int(span.get("endTimeUnixNano", 0))
        duration_ms = (end_ns - start_ns) / 1_000_000 if start_ns and end_ns else 0

        start_dt = (
            datetime.fromtimestamp(start_ns / 1e9, tz=UTC)
            if start_ns
            else datetime.now(tz=UTC)
        )
        end_dt = (
            datetime.fromtimestamp(end_ns / 1e9, tz=UTC)
            if end_ns
            else datetime.now(tz=UTC)
        )

        # Build request
        messages = []
        prompt = attrs.get("prompt")
        if prompt:
            try:
                messages = json.loads(prompt) if isinstance(prompt, str) else prompt
            except json.JSONDecodeError:
                messages = [{"role": "user", "content": prompt}]

        req = LLMRequest(
            timestamp=start_dt,
            method="POST",
            path="/v1/messages",
            body={
                "model": model or "",
                "messages": messages,
                "max_tokens": attrs.get("max_tokens"),
                "temperature": attrs.get("temperature"),
            },
        )

        # Build response
        content = []
        completion = attrs.get("completion")
        if completion:
            try:
                content = (
                    json.loads(completion)
                    if isinstance(completion, str)
                    else completion
                )
            except json.JSONDecodeError:
                content = [{"type": "text", "text": completion}]

        usage = {}
        if attrs.get("input_tokens"):
            usage["input_tokens"] = int(attrs["input_tokens"])
        if attrs.get("output_tokens"):
            usage["output_tokens"] = int(attrs["output_tokens"])

        resp = LLMResponse(
            timestamp=end_dt,
            status_code=200,
            body={
                "content": content,
                "usage": usage,
                "model": attrs.get("response_model") or model or "",
                "stop_reason": attrs.get("finish_reasons"),
            },
        )

        self._trajectory.exchanges.append(
            LLMExchange(request=req, response=resp, duration_ms=duration_ms)
        )


def _parse_attributes(attrs: list[dict]) -> dict[str, Any]:
    """Parse OTLP attribute list into a flat dict using GenAI semantic conventions."""
    result: dict[str, Any] = {}
    for attr in attrs:
        key = attr.get("key", "")
        value = attr.get("value", {})
        # Extract value from OTLP typed value wrapper
        val = (
            value.get("stringValue")
            or value.get("intValue")
            or value.get("doubleValue")
            or value.get("boolValue")
        )
        if key in _LLM_ATTRS:
            result[_LLM_ATTRS[key]] = val
    return result


def _raw_attr_keys(span: dict) -> list[str]:
    return [a.get("key", "") for a in span.get("attributes", [])]
