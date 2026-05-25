"""LLM API proxy server — captures all agent↔LLM traffic as trajectory.

Supports both non-streaming and streaming (SSE) responses.
"""

import asyncio
import gzip
import importlib
import io
import json
import logging
import time
import zlib
from datetime import datetime
from typing import Any

import httpx

from .types import LLMExchange, LLMRequest, LLMResponse, Trajectory

logger = logging.getLogger(__name__)

_RAW_RESP_TRUNCATE = 10000  # max chars for non-JSON response body capture
_RAW_REQ_TRUNCATE = 10000  # max chars for non-JSON request body capture
_PROMPT_CACHE_RETENTION_VALUES = {"in_memory", "24h"}


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
        prompt_cache_retention: str | None = None,
    ):
        if (
            prompt_cache_retention is not None
            and prompt_cache_retention not in _PROMPT_CACHE_RETENTION_VALUES
        ):
            raise ValueError(
                "prompt_cache_retention must be one of: "
                f"{', '.join(sorted(_PROMPT_CACHE_RETENTION_VALUES))}"
            )
        self._target = target.rstrip("/")
        self._host = host
        self._port = port
        self._prompt_cache_retention = prompt_cache_retention
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
    def target(self) -> str:
        """Upstream URL this proxy forwards to (normalized, no trailing slash)."""
        return self._target

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    async def start(self) -> None:
        logging.getLogger("httpx").setLevel(logging.WARNING)
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

                body = _parse_request_body(body_bytes, headers)
                body, body_bytes, headers = _apply_prompt_cache_retention(
                    body=body,
                    body_bytes=body_bytes,
                    headers=headers,
                    path=path,
                    prompt_cache_retention=self._prompt_cache_retention,
                )

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
        assert self._client is not None, (
            "proxy must be started before handling requests"
        )
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
        assert self._client is not None, (
            "proxy must be started before handling requests"
        )
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
            sse_buffer = ""

            async for chunk in resp.aiter_bytes():
                # Forward chunk to agent
                chunk_hex = f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n"
                writer.write(chunk_hex)
                await writer.drain()

                # Parse SSE events from chunk
                events, sse_buffer = _parse_sse_events_buffer(
                    sse_buffer + chunk.decode(errors="replace")
                )
                collected_events.extend(events)
            if sse_buffer.strip():
                collected_events.extend(_parse_sse_events(sse_buffer.encode()))

            # End chunked encoding
            writer.write(b"0\r\n\r\n")
            await writer.drain()

        duration_ms = (time.monotonic() - start_time) * 1000

        # Reconstruct the final response from collected SSE events. The stream
        # has already been fully forwarded to the agent at this point, so a
        # reconstruction failure must not propagate — it would write a bogus
        # 502 onto a completed response and drop the exchange. Degrade instead.
        try:
            resp_body = _reconstruct_response(collected_events)
        except Exception as e:
            logger.warning(f"SSE response reconstruction failed: {e}")
            resp_body = {}
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


def _parse_sse_events_buffer(buffer: str) -> tuple[list[dict[str, Any]], str]:
    """Parse complete SSE events and return unconsumed trailing text."""
    normalized = buffer.replace("\r\n", "\n")
    events: list[dict[str, Any]] = []
    while "\n\n" in normalized:
        raw_event, normalized = normalized.split("\n\n", 1)
        data_lines = []
        for line in raw_event.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        if data == "[DONE]":
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            logger.debug("Skipping malformed SSE event")
    return events, normalized


def _parse_request_body(body_bytes: bytes, headers: dict[str, str]) -> dict[str, Any]:
    """Parse a possibly-compressed JSON request body for telemetry capture."""
    if not body_bytes:
        return {}

    decoded = _decode_request_body(
        body_bytes, headers.get("content-encoding", "identity")
    )
    try:
        parsed = json.loads(decoded)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"raw": decoded.decode(errors="replace")[:_RAW_REQ_TRUNCATE]}

    if isinstance(parsed, dict):
        return parsed
    return {"raw": parsed}


def _apply_prompt_cache_retention(
    *,
    body: dict[str, Any],
    body_bytes: bytes,
    headers: dict[str, str],
    path: str,
    prompt_cache_retention: str | None,
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    """Inject OpenAI prompt_cache_retention when configured."""
    if not prompt_cache_retention or not _is_openai_generation_path(path):
        return body, body_bytes, headers
    if "raw" in body or "prompt_cache_retention" in body:
        return body, body_bytes, headers

    updated_body = {**body, "prompt_cache_retention": prompt_cache_retention}
    updated_body_bytes = json.dumps(updated_body, separators=(",", ":")).encode()
    updated_headers = {
        k: v
        for k, v in headers.items()
        if k not in {"content-length", "content-encoding"}
    }
    updated_headers.setdefault("content-type", "application/json")
    return updated_body, updated_body_bytes, updated_headers


def _is_openai_generation_path(path: str) -> bool:
    request_path = path.split("?", 1)[0].rstrip("/")
    return request_path.endswith("/responses") or request_path.endswith(
        "/chat/completions"
    )


def _decode_request_body(body_bytes: bytes, content_encoding: str) -> bytes:
    """Decode HTTP content encodings so JSON fields such as stream are visible."""
    decoded = body_bytes
    encodings = [
        part.strip().lower()
        for part in content_encoding.split(",")
        if part.strip() and part.strip().lower() != "identity"
    ]
    for encoding in reversed(encodings):
        try:
            if encoding == "gzip":
                decoded = gzip.decompress(decoded)
            elif encoding == "deflate":
                decoded = zlib.decompress(decoded)
            elif encoding == "zstd":
                decoded = _decompress_zstd(decoded)
            elif encoding == "br":
                decoded = _decompress_brotli(decoded)
            else:
                logger.debug("Unsupported request content-encoding: %s", encoding)
                return body_bytes
        except Exception as e:
            logger.debug("Could not decode %s request body: %s", encoding, e)
            return body_bytes
    return decoded


def _decompress_zstd(data: bytes) -> bytes:
    try:
        zstd = importlib.import_module("zstandard")
    except ImportError:
        logger.debug("zstandard is unavailable; leaving request body compressed")
        return data

    dctx = zstd.ZstdDecompressor()
    try:
        return dctx.decompress(data)
    except zstd.ZstdError:
        with dctx.stream_reader(io.BytesIO(data)) as reader:
            return reader.read()


def _decompress_brotli(data: bytes) -> bytes:
    try:
        brotli = importlib.import_module("brotli")
    except ImportError:
        logger.debug("brotli is unavailable; leaving request body compressed")
        return data

    return brotli.decompress(data)


def _reconstruct_response(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct a complete API response from SSE events.

    Works with Anthropic (message_start/content_block_delta/message_delta),
    OpenAI (choices[].delta), and Gemini (candidates[] + usageMetadata)
    streaming formats. The Gemini path matters for usage telemetry: every
    Gemini SSE chunk carries an incremental ``usageMetadata`` block, and
    losing it means Gemini Docker runs surface ``usage_source: unavailable``
    despite the proxy capturing every byte (see issue #375).
    """
    # Try Gemini format early — its events carry neither ``type`` nor
    # ``choices`` and would otherwise fall through to the raw-events
    # fallback, dropping ``usageMetadata`` on the floor.
    if any(
        ("candidates" in event and isinstance(event.get("candidates"), list))
        or "usageMetadata" in event
        or "modelVersion" in event
        for event in events
    ) and not any(event.get("type") or event.get("choices") for event in events):
        return _reconstruct_gemini_response(events)

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
                    # content_block_start seeds tool_use input as {}; the
                    # streamed partial_json fragments are accumulated as a
                    # string and parsed back to JSON once the block closes.
                    if not isinstance(block.get("input"), str):
                        block["input"] = ""
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
    for event in events:
        if event.get("type") == "response.completed" and isinstance(
            event.get("response"), dict
        ):
            return event["response"]

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


def _reconstruct_gemini_response(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct a Gemini generateContent response from SSE chunks.

    Each Gemini streaming chunk looks like::

        {
          "candidates": [{
            "content": {"role": "model", "parts": [{"text": "..."}]},
            "finishReason": "STOP",
            "index": 0
          }],
          "modelVersion": "gemini-3.1-flash-lite-preview",
          "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 4,
            "totalTokenCount": 15
          }
        }

    Incremental chunks carry partial text in ``candidates[0].content.parts[i].text``
    that we concatenate; ``usageMetadata`` is cumulative per the Gemini API
    contract, so the *last* chunk's value is the final total. We keep this
    shape verbatim under ``usageMetadata`` (not ``usage``) so the existing
    Trajectory accessors in ``trajectories.types`` find it.
    """
    # Aggregate text parts by (candidate_index, part_index). Tool/function
    # call parts and inline data are passed through from the most recent
    # chunk that supplied them (Gemini sends them whole, not as text deltas).
    candidates: dict[int, dict[str, Any]] = {}
    usage_metadata: dict[str, Any] = {}
    model_version: str = ""
    prompt_feedback: dict[str, Any] = {}

    for event in events:
        if isinstance(event.get("modelVersion"), str):
            model_version = event["modelVersion"]
        if isinstance(event.get("usageMetadata"), dict):
            # Cumulative per Gemini's contract — overwrite so the last chunk
            # wins. Falling back to .update() would be wrong if a key
            # disappeared mid-stream, which Gemini doesn't promise.
            usage_metadata = dict(event["usageMetadata"])
        if isinstance(event.get("promptFeedback"), dict):
            prompt_feedback = event["promptFeedback"]

        for cand in event.get("candidates", []) or []:
            if not isinstance(cand, dict):
                continue
            idx = cand.get("index", 0)
            existing = candidates.setdefault(
                idx,
                {
                    "content": {"role": "model", "parts": []},
                    "index": idx,
                },
            )
            content = cand.get("content")
            if isinstance(content, dict):
                role = content.get("role")
                if isinstance(role, str):
                    existing["content"]["role"] = role
                for part_idx, part in enumerate(content.get("parts", []) or []):
                    if not isinstance(part, dict):
                        continue
                    parts_list = existing["content"]["parts"]
                    while len(parts_list) <= part_idx:
                        parts_list.append({})
                    target = parts_list[part_idx]
                    if "text" in part:
                        target["text"] = target.get("text", "") + (part.get("text") or "")
                    # Function calls / inline data / other part shapes —
                    # carry the most recent value verbatim. Gemini sends
                    # these whole rather than as deltas.
                    for key, value in part.items():
                        if key == "text":
                            continue
                        target[key] = value
            if "finishReason" in cand:
                existing["finishReason"] = cand["finishReason"]
            if "safetyRatings" in cand:
                existing["safetyRatings"] = cand["safetyRatings"]
            if "citationMetadata" in cand:
                existing["citationMetadata"] = cand["citationMetadata"]

    response: dict[str, Any] = {
        "candidates": [candidates[i] for i in sorted(candidates)],
    }
    # The model field is consumed by _model_from_trajectory; mirror Gemini's
    # ``modelVersion`` into the standard ``model`` slot so downstream pricing
    # lookup works without a special case.
    if model_version:
        response["modelVersion"] = model_version
        response["model"] = model_version
    if usage_metadata:
        response["usageMetadata"] = usage_metadata
    if prompt_feedback:
        response["promptFeedback"] = prompt_feedback
    return response
