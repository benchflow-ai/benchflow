"""LLM API proxy server — captures all agent↔LLM traffic as trajectory.

Supports both non-streaming and streaming (SSE) responses.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any

import httpx

from .types import LLMExchange, LLMRequest, LLMResponse, Trajectory

logger = logging.getLogger(__name__)

_RAW_RESP_TRUNCATE = 10000  # max chars for non-JSON response body capture


class TrajectoryProxy:
    """HTTP proxy that forwards LLM API requests and captures exchanges.

    Handles both regular JSON responses and streaming SSE responses.
    For streaming, collects all events and reconstructs the final message
    for trajectory capture while forwarding chunks to the agent in real-time.
    """

    def __init__(
        self,
        target: str = "https://api.anthropic.com",
        session_id: str = "",
        agent_name: str = "",
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self._target = target.rstrip("/")
        self._host = host
        self._port = port
        self._trajectory = Trajectory(
            session_id=session_id,
            agent_name=agent_name,
        )
        self._server: asyncio.Server | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port
        )
        sock = self._server.sockets[0]
        self._port = sock.getsockname()[1]
        logger.info(f"Trajectory proxy listening on {self.base_url} → {self._target}")

    async def stop(self) -> None:
        self._trajectory.finished_at = datetime.now()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self._client:
            await self._client.aclose()
        logger.info(
            f"Proxy stopped. Captured {len(self._trajectory.exchanges)} exchanges."
        )

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    break
                request_line = request_line.decode().strip()
                if not request_line:
                    break

                parts = request_line.split(" ", 2)
                if len(parts) < 2:
                    break
                method, path = parts[0], parts[1]

                headers: dict[str, str] = {}
                while True:
                    header_line = await reader.readline()
                    if header_line in (b"\r\n", b"\n", b""):
                        break
                    key, _, value = header_line.decode().partition(":")
                    headers[key.strip().lower()] = value.strip()

                content_length = int(headers.get("content-length", "0"))
                body_bytes = b""
                if content_length > 0:
                    body_bytes = await reader.readexactly(content_length)

                body: dict[str, Any] = {}
                if body_bytes:
                    try:
                        body = json.loads(body_bytes)
                    except json.JSONDecodeError:
                        body = {"raw": body_bytes.decode(errors="replace")}

                req = LLMRequest(
                    method=method,
                    path=path,
                    headers={k: v for k, v in headers.items() if k != "content-length"},
                    body=body,
                )

                is_streaming = body.get("stream", False)

                start_time = time.monotonic()
                target_url = f"{self._target}{path}"
                forward_headers = {
                    k: v
                    for k, v in headers.items()
                    if k not in ("host", "content-length", "transfer-encoding")
                }

                try:
                    if is_streaming:
                        await self._handle_streaming(
                            req,
                            method,
                            target_url,
                            forward_headers,
                            body_bytes,
                            start_time,
                            writer,
                        )
                    else:
                        await self._handle_regular(
                            req,
                            method,
                            target_url,
                            forward_headers,
                            body_bytes,
                            start_time,
                            writer,
                        )
                except Exception as e:
                    logger.error(f"Proxy forward error: {e}")
                    error_body = json.dumps({"error": str(e)}).encode()
                    writer.write(
                        b"HTTP/1.1 502 Bad Gateway\r\n"
                        b"content-type: application/json\r\n"
                        + f"content-length: {len(error_body)}\r\n\r\n".encode()
                        + error_body
                    )
                    await writer.drain()
                    break

        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                logger.debug("Writer close failed during connection teardown")

    async def _handle_regular(
        self,
        req: LLMRequest,
        method: str,
        url: str,
        headers: dict,
        body_bytes: bytes,
        start_time: float,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a non-streaming request/response."""
        resp = await self._client.request(
            method=method,
            url=url,
            headers=headers,
            content=body_bytes if body_bytes else None,
        )
        duration_ms = (time.monotonic() - start_time) * 1000

        resp_body: dict[str, Any] = {}
        try:
            resp_body = resp.json()
        except (json.JSONDecodeError, ValueError):
            resp_body = {"raw": resp.text[:_RAW_RESP_TRUNCATE]}

        self._record_exchange(
            req, resp.status_code, dict(resp.headers), resp_body, duration_ms
        )

        resp_bytes = resp.content
        response_line = f"HTTP/1.1 {resp.status_code} OK\r\n"
        resp_headers = (
            f"content-type: {resp.headers.get('content-type', 'application/json')}\r\n"
            f"content-length: {len(resp_bytes)}\r\n"
            "\r\n"
        )
        writer.write(response_line.encode() + resp_headers.encode() + resp_bytes)
        await writer.drain()

    async def _handle_streaming(
        self,
        req: LLMRequest,
        method: str,
        url: str,
        headers: dict,
        body_bytes: bytes,
        start_time: float,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a streaming (SSE) request/response.

        Forwards SSE chunks to the agent in real-time while collecting
        events to reconstruct the full response for trajectory capture.
        """
        async with self._client.stream(
            method=method,
            url=url,
            headers=headers,
            content=body_bytes if body_bytes else None,
        ) as resp:
            # Send response headers with chunked transfer
            writer.write(
                f"HTTP/1.1 {resp.status_code} OK\r\n"
                f"content-type: {resp.headers.get('content-type', 'text/event-stream')}\r\n"
                "transfer-encoding: chunked\r\n"
                "\r\n".encode()
            )
            await writer.drain()

            # Collect SSE events while forwarding
            collected_events: list[dict[str, Any]] = []

            async for chunk in resp.aiter_bytes():
                # Forward chunk to agent
                chunk_hex = f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
                writer.write(chunk_hex)
                await writer.drain()

                # Parse SSE events from chunk
                for event in _parse_sse_events(chunk):
                    collected_events.append(event)

            # End chunked encoding
            writer.write(b"0\r\n\r\n")
            await writer.drain()

        duration_ms = (time.monotonic() - start_time) * 1000

        # Reconstruct the final response from collected SSE events
        resp_body = _reconstruct_response(collected_events)
        self._record_exchange(
            req, resp.status_code, dict(resp.headers), resp_body, duration_ms
        )

    def _record_exchange(
        self,
        req: LLMRequest,
        status_code: int,
        headers: dict,
        body: dict,
        duration_ms: float,
    ) -> None:
        llm_resp = LLMResponse(
            status_code=status_code,
            headers=headers,
            body=body,
        )
        exchange = LLMExchange(request=req, response=llm_resp, duration_ms=duration_ms)
        self._trajectory.exchanges.append(exchange)
        logger.debug(
            f"Captured: {req.method} {req.path} → {status_code} "
            f"({duration_ms:.0f}ms, stream={req.body.get('stream', False)})"
        )


def _parse_sse_events(chunk: bytes) -> list[dict[str, Any]]:
    """Parse SSE events from a chunk of bytes."""
    events = []
    text = chunk.decode(errors="replace")
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                continue
            try:
                events.append(json.loads(data))
            except json.JSONDecodeError:
                logger.debug("Skipping malformed SSE event")
    return events


def _reconstruct_response(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct a complete API response from SSE events.

    Works with both Anthropic (message_start/content_block_delta/message_delta)
    and OpenAI (choices[].delta) streaming formats.
    """
    # Try Anthropic format first
    message: dict[str, Any] = {}
    content_blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    model = ""

    for event in events:
        event_type = event.get("type", "")

        # Anthropic streaming format
        if event_type == "message_start":
            msg = event.get("message", {})
            model = msg.get("model", "")
            usage.update(msg.get("usage", {}))
            message = msg

        elif event_type == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block", {})
            content_blocks[idx] = block

        elif event_type == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta", {})
            if idx in content_blocks:
                block = content_blocks[idx]
                if delta.get("type") == "text_delta":
                    block["text"] = block.get("text", "") + delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    block.setdefault("input", "")
                    block["input"] += delta.get("partial_json", "")

        elif event_type == "message_delta":
            delta = event.get("delta", {})
            msg_usage = event.get("usage", {})
            usage.update(msg_usage)
            if "stop_reason" in delta:
                message["stop_reason"] = delta["stop_reason"]

    # If we got Anthropic events, build the response
    if message or content_blocks:
        content = [content_blocks[i] for i in sorted(content_blocks)]
        # Parse JSON input for tool_use blocks
        for block in content:
            if block.get("type") == "tool_use" and isinstance(block.get("input"), str):
                try:
                    block["input"] = json.loads(block["input"])
                except json.JSONDecodeError:
                    logger.debug(
                        "Could not parse tool_use input as JSON, keeping as string"
                    )
        return {
            "content": content,
            "model": model,
            "usage": usage,
            "stop_reason": message.get("stop_reason"),
            "role": "assistant",
        }

    # Try OpenAI format: collect delta chunks
    openai_content = ""
    openai_tool_calls: list[dict] = []
    openai_model = ""
    openai_usage: dict[str, Any] = {}
    for event in events:
        if "model" in event:
            openai_model = event["model"]
        if event.get("usage"):
            openai_usage.update(event["usage"])
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                openai_content += delta["content"]
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                while len(openai_tool_calls) <= idx:
                    openai_tool_calls.append(
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )
                if "id" in tc:
                    openai_tool_calls[idx]["id"] = tc["id"]
                if "function" in tc:
                    fn = tc["function"]
                    if "name" in fn:
                        openai_tool_calls[idx]["function"]["name"] = fn["name"]
                    if "arguments" in fn:
                        openai_tool_calls[idx]["function"]["arguments"] += fn[
                            "arguments"
                        ]

    if openai_content or openai_tool_calls or openai_model:
        msg: dict[str, Any] = {"role": "assistant", "content": openai_content}
        if openai_tool_calls:
            msg["tool_calls"] = openai_tool_calls
        return {
            "choices": [{"message": msg, "finish_reason": "stop"}],
            "model": openai_model,
            "usage": openai_usage,
        }

    # Fallback: return raw events
    return {"_raw_events": events}
