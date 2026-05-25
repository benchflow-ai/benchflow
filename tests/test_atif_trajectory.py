"""Tests for trajectory proxy, OTel collector, and types."""

import asyncio
import gzip
import json

import httpx
import pytest

from benchflow.trajectories.otel import OTelCollector
from benchflow.trajectories.proxy import TrajectoryProxy, _parse_request_body
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)


class TestTrajectoryTypes:
    def test_exchange_with_usage(self) -> None:
        traj = Trajectory(session_id="s1")
        traj.exchanges.append(
            LLMExchange(
                request=LLMRequest(
                    path="/v1/messages",
                    body={
                        "model": "claude-sonnet-4-20250514",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                ),
                response=LLMResponse(
                    body={
                        "content": [{"type": "text", "text": "hello"}],
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    }
                ),
                duration_ms=150.0,
            )
        )
        assert traj.total_input_tokens == 10
        assert traj.total_output_tokens == 5

        # Add second exchange (Anthropic format)
        traj.exchanges.append(
            LLMExchange(
                request=LLMRequest(
                    body={"messages": [{"role": "user", "content": "more"}]}
                ),
                response=LLMResponse(
                    body={
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 200, "output_tokens": 50},
                    }
                ),
            )
        )
        assert traj.total_input_tokens == 210  # 10 + 200
        assert traj.total_output_tokens == 55  # 5 + 50

        # Add third exchange (OpenAI format fallback)
        traj.exchanges.append(
            LLMExchange(
                request=LLMRequest(body={"messages": []}),
                response=LLMResponse(
                    body={
                        "choices": [
                            {"message": {"role": "assistant", "content": "hi"}}
                        ],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                    }
                ),
            )
        )
        assert traj.total_input_tokens == 260  # 210 + 50
        assert traj.total_output_tokens == 65  # 55 + 10

    def test_messages_extraction(self) -> None:
        traj = Trajectory(session_id="s1")
        traj.exchanges.append(
            LLMExchange(
                request=LLMRequest(
                    body={"messages": [{"role": "user", "content": "hello"}]},
                ),
                response=LLMResponse(
                    body={"content": [{"type": "text", "text": "hi back"}]},
                ),
            )
        )
        msgs = traj.messages
        assert len(msgs) == 2  # exactly 1 request message + 1 response message
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_to_jsonl(self) -> None:
        traj = Trajectory(session_id="s1")
        traj.exchanges.append(
            LLMExchange(
                request=LLMRequest(body={"messages": []}),
                response=LLMResponse(body={"content": []}),
            )
        )
        jsonl = traj.to_jsonl()
        parsed = json.loads(jsonl)
        assert "request" in parsed
        assert "response" in parsed

    def test_to_jsonl_redacts_bearer_authorization(self) -> None:
        traj = Trajectory(session_id="s1")
        traj.exchanges.append(
            LLMExchange(
                request=LLMRequest(
                    headers={"authorization": "Bearer secret.jwt.token"},
                    body={"messages": []},
                ),
                response=LLMResponse(body={"content": []}),
            )
        )

        jsonl = traj.to_jsonl()

        assert "secret.jwt.token" not in jsonl
        assert "Bearer ***REDACTED***" in jsonl


class TestTrajectoryProxy:
    def test_proxy_parses_zstd_compressed_request_body(self) -> None:
        """Guards commit 3e8b06c telemetry proxy work against zstd request bodies."""
        zstd = pytest.importorskip("zstandard")
        compressed_body = zstd.ZstdCompressor().compress(
            json.dumps({"stream": True, "messages": []}).encode()
        )

        body = _parse_request_body(
            compressed_body,
            {"content-encoding": "zstd"},
        )

        assert body["stream"] is True

    @pytest.mark.asyncio
    async def test_proxy_captures_exchange(self) -> None:
        """Test proxy with a real HTTP request to a mock target."""

        # Start a tiny echo server as the "LLM API"
        async def echo_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.readline()  # request line
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            cl = int(headers.get("content-length", "0"))
            if cl > 0:
                await reader.readexactly(cl)  # consume body

            resp_body = json.dumps(
                {
                    "content": [{"type": "text", "text": "echo"}],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: application/json\r\n"
                + f"content-length: {len(resp_body)}\r\n\r\n".encode()
                + resp_body
            )
            await writer.drain()
            writer.close()

        echo_server = await asyncio.start_server(echo_handler, "127.0.0.1", 0)
        echo_port = echo_server.sockets[0].getsockname()[1]

        proxy = TrajectoryProxy(
            target=f"http://127.0.0.1:{echo_port}",
            session_id="test",
        )
        await proxy.start()

        # Send a request through the proxy
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{proxy.base_url}/v1/messages",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 200

        traj = proxy.trajectory
        assert len(traj.exchanges) == 1
        assert traj.exchanges[0].request.path == "/v1/messages"
        assert traj.total_input_tokens == 100
        assert traj.total_output_tokens == 20

        await proxy.stop()
        echo_server.close()
        await echo_server.wait_closed()

    @pytest.mark.asyncio
    async def test_proxy_detects_streaming_from_compressed_request(self) -> None:
        """Guards commit 3e8b06c telemetry proxy work against compressed requests."""

        async def stream_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.readline()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            cl = int(headers.get("content-length", "0"))
            if cl > 0:
                await reader.readexactly(cl)

            frames = [
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-haiku-4-5-20251001",
                        "usage": {"input_tokens": 7, "output_tokens": 1},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                },
                {"type": "message_delta", "usage": {"output_tokens": 2}},
            ]
            body = b"".join(
                f"data: {json.dumps(frame)}\n\n".encode() for frame in frames
            )
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: text/event-stream\r\n"
                + f"content-length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            writer.close()

        stream_server = await asyncio.start_server(stream_handler, "127.0.0.1", 0)
        stream_port = stream_server.sockets[0].getsockname()[1]

        proxy = TrajectoryProxy(
            target=f"http://127.0.0.1:{stream_port}",
            session_id="test-compressed",
        )
        await proxy.start()

        try:
            compressed_body = gzip.compress(
                json.dumps(
                    {
                        "model": "claude-haiku-4-5-20251001",
                        "stream": True,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                ).encode()
            )
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{proxy.base_url}/v1/messages",
                    content=compressed_body,
                    headers={
                        "content-type": "application/json",
                        "content-encoding": "gzip",
                    },
                )
                assert resp.status_code == 200

            traj = proxy.trajectory
            assert len(traj.exchanges) == 1
            assert traj.exchanges[0].request.body["stream"] is True
            assert traj.total_input_tokens == 7
            assert traj.total_output_tokens == 2
        finally:
            await proxy.stop()
            stream_server.close()
            await stream_server.wait_closed()

    @pytest.mark.asyncio
    async def test_proxy_injects_prompt_cache_retention_for_openai_requests(
        self,
    ) -> None:
        forwarded: dict[str, object] = {}

        async def openai_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            request_line = (await reader.readline()).decode().strip()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            cl = int(headers.get("content-length", "0"))
            body_bytes = await reader.readexactly(cl)
            forwarded["request_line"] = request_line
            forwarded["headers"] = headers
            forwarded["body"] = json.loads(body_bytes)

            resp_body = json.dumps(
                {
                    "model": "gpt-5.5",
                    "usage": {
                        "prompt_tokens": 1200,
                        "completion_tokens": 20,
                        "total_tokens": 1220,
                        "prompt_tokens_details": {"cached_tokens": 1024},
                    },
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: application/json\r\n"
                + f"content-length: {len(resp_body)}\r\n\r\n".encode()
                + resp_body
            )
            await writer.drain()
            writer.close()

        openai_server = await asyncio.start_server(openai_handler, "127.0.0.1", 0)
        openai_port = openai_server.sockets[0].getsockname()[1]
        proxy = TrajectoryProxy(
            target=f"http://127.0.0.1:{openai_port}",
            session_id="test-cache-retention",
            prompt_cache_retention="24h",
        )
        await proxy.start()

        try:
            compressed_body = gzip.compress(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "input": "Your prompt goes here...",
                    }
                ).encode()
            )
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{proxy.base_url}/responses",
                    content=compressed_body,
                    headers={
                        "content-type": "application/json",
                        "content-encoding": "gzip",
                    },
                )
                assert resp.status_code == 200

            assert forwarded["request_line"] == "POST /responses HTTP/1.1"
            headers = forwarded["headers"]
            assert isinstance(headers, dict)
            assert "content-encoding" not in headers
            body = forwarded["body"]
            assert isinstance(body, dict)
            assert body["prompt_cache_retention"] == "24h"
            captured_body = proxy.trajectory.exchanges[0].request.body
            assert captured_body["prompt_cache_retention"] == "24h"
        finally:
            await proxy.stop()
            openai_server.close()
            await openai_server.wait_closed()

    @pytest.mark.asyncio
    async def test_proxy_reconstructs_openai_responses_completed_usage(self) -> None:
        async def stream_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.readline()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            cl = int(headers.get("content-length", "0"))
            if cl > 0:
                await reader.readexactly(cl)

            completed = {
                "type": "response.completed",
                "response": {
                    "model": "gpt-4.1",
                    "usage": {
                        "input_tokens": 123,
                        "output_tokens": 45,
                        "total_tokens": 168,
                        "input_tokens_details": {"cached_tokens": 12},
                    },
                },
            }
            body = f"data: {json.dumps(completed)}\n\n".encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: text/event-stream\r\n"
                b"transfer-encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            first, second = body[:15], body[15:]
            for chunk in (first, second):
                writer.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
            writer.close()

        stream_server = await asyncio.start_server(stream_handler, "127.0.0.1", 0)
        stream_port = stream_server.sockets[0].getsockname()[1]
        proxy = TrajectoryProxy(
            target=f"http://127.0.0.1:{stream_port}",
            session_id="test-openai-responses",
        )
        await proxy.start()

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{proxy.base_url}/responses",
                    json={"model": "gpt-4.1", "input": "hi", "stream": True},
                )
                assert resp.status_code == 200

            traj = proxy.trajectory
            assert len(traj.exchanges) == 1
            assert traj.total_input_tokens == 123
            assert traj.total_output_tokens == 45
            assert traj.total_cache_read_tokens == 12
            assert traj.total_provider_tokens == 168
        finally:
            await proxy.stop()
            stream_server.close()
            await stream_server.wait_closed()

    @pytest.mark.asyncio
    async def test_proxy_reconstructs_gemini_streaming_usage(self) -> None:
        """Regression for #375: Gemini's SSE chunks carry ``usageMetadata`` per
        chunk and no ``type`` / ``choices`` fields. Before this fix the proxy
        fell through to a raw-events fallback, so token counts and the model
        name were dropped — Docker Gemini runs reported ``usage_source:
        unavailable`` despite the proxy capturing every chunk."""

        async def gemini_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.readline()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            cl = int(headers.get("content-length", "0"))
            if cl > 0:
                await reader.readexactly(cl)

            # Gemini's streamGenerateContent?alt=sse chunks. usageMetadata is
            # cumulative — the last chunk carries the final totals.
            frames = [
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "Hello "}],
                            },
                            "index": 0,
                        }
                    ],
                    "modelVersion": "gemini-3.1-flash-lite-preview",
                    "usageMetadata": {
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 1,
                        "totalTokenCount": 12,
                    },
                },
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "world"}],
                            },
                            "finishReason": "STOP",
                            "index": 0,
                        }
                    ],
                    "modelVersion": "gemini-3.1-flash-lite-preview",
                    "usageMetadata": {
                        "promptTokenCount": 11,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 15,
                    },
                },
            ]
            body = b"".join(
                f"data: {json.dumps(frame)}\n\n".encode() for frame in frames
            )
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: text/event-stream\r\n"
                + f"content-length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            writer.close()

        stream_server = await asyncio.start_server(gemini_handler, "127.0.0.1", 0)
        stream_port = stream_server.sockets[0].getsockname()[1]

        proxy = TrajectoryProxy(
            target=f"http://127.0.0.1:{stream_port}",
            session_id="gemini-stream-test",
            agent_name="gemini",
        )
        await proxy.start()

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{proxy.base_url}/v1beta/models/"
                    "gemini-3.1-flash-lite-preview:streamGenerateContent?alt=sse",
                    json={
                        "contents": [
                            {"role": "user", "parts": [{"text": "say hi"}]}
                        ],
                        "stream": True,
                    },
                )
                assert resp.status_code == 200

            traj = proxy.trajectory
            assert len(traj.exchanges) == 1

            # The captured response body must carry Gemini-shaped fields the
            # downstream token/cost extractors expect — not the
            # ``_raw_events`` fallback that drops usageMetadata.
            body = traj.exchanges[0].response.body
            assert "_raw_events" not in body, (
                "Gemini SSE response fell through to raw-events fallback; "
                "usageMetadata would be lost"
            )
            assert body.get("usageMetadata", {}).get("promptTokenCount") == 11
            assert body.get("usageMetadata", {}).get("candidatesTokenCount") == 4
            assert body.get("usageMetadata", {}).get("totalTokenCount") == 15
            # Text parts are concatenated across chunks.
            assert body["candidates"][0]["content"]["parts"][0]["text"] == "Hello world"

            # End-to-end: the Trajectory accessors that feed extract_usage
            # see the cumulative counts.
            assert traj.total_input_tokens == 11
            assert traj.total_output_tokens == 4
            assert traj.total_provider_tokens == 15
        finally:
            await proxy.stop()
            stream_server.close()
            await stream_server.wait_closed()

    @pytest.mark.asyncio
    async def test_proxy_reconstructs_gemini_sse_without_stream_flag(self) -> None:
        """Guards the follow-up to PR #483: Gemini CLI's real
        streamGenerateContent request uses ``alt=sse`` without a JSON
        ``stream`` flag, so SSE detection must not depend on request body
        shape alone (#375)."""

        async def gemini_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.readline()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            cl = int(headers.get("content-length", "0"))
            if cl > 0:
                await reader.readexactly(cl)

            frame = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "done"}],
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-2.5-flash",
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 8,
                },
            }
            body = f"data: {json.dumps(frame)}\r\n\r\n".encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: text/event-stream\r\n"
                + f"content-length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            writer.close()

        stream_server = await asyncio.start_server(gemini_handler, "127.0.0.1", 0)
        stream_port = stream_server.sockets[0].getsockname()[1]

        proxy = TrajectoryProxy(
            target=f"http://127.0.0.1:{stream_port}",
            session_id="gemini-sse-no-stream-flag",
            agent_name="gemini",
        )
        await proxy.start()

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{proxy.base_url}/v1beta/models/"
                    "gemini-2.5-flash:streamGenerateContent?alt=sse",
                    json={"contents": []},
                )
                assert resp.status_code == 200

            body = proxy.trajectory.exchanges[0].response.body
            assert "raw" not in body
            assert "_raw_events" not in body
            assert body["usageMetadata"]["totalTokenCount"] == 8
            assert proxy.trajectory.total_provider_tokens == 8
        finally:
            await proxy.stop()
            stream_server.close()
            await stream_server.wait_closed()

    @pytest.mark.asyncio
    async def test_extract_usage_populates_gemini_telemetry_after_stream(self) -> None:
        """Regression for #375 — assert the full pipeline: a Gemini streaming
        response goes through the proxy and ``extract_usage`` returns
        populated token fields with ``usage_source=provider_response``
        (not the ``unavailable`` defaults seen in the bug)."""
        from benchflow.providers.runtime import ProviderRuntime, extract_usage

        async def gemini_handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.readline()
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
            cl = int(headers.get("content-length", "0"))
            if cl > 0:
                await reader.readexactly(cl)

            frame = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "done"}],
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "modelVersion": "gemini-2.5-flash",
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 50,
                    "totalTokenCount": 150,
                },
            }
            body = f"data: {json.dumps(frame)}\n\n".encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"content-type: text/event-stream\r\n"
                + f"content-length: {len(body)}\r\n\r\n".encode()
                + body
            )
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(gemini_handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        proxy = TrajectoryProxy(
            target=f"http://127.0.0.1:{port}",
            session_id="gemini-extract-usage",
            agent_name="gemini",
        )
        await proxy.start()

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{proxy.base_url}/v1beta/models/"
                    "gemini-2.5-flash:streamGenerateContent?alt=sse",
                    json={"contents": [], "stream": True},
                )

            runtime = ProviderRuntime(
                kind="usage-proxy",
                host="127.0.0.1",
                port=proxy.port,
                backend_model="gemini-2.5-flash",
                server=proxy,
            )
            usage = extract_usage(runtime)
        finally:
            await proxy.stop()
            server.close()
            await server.wait_closed()

        assert usage["usage_source"] == "provider_response"
        assert usage["n_input_tokens"] == 100
        assert usage["n_output_tokens"] == 50
        assert usage["total_tokens"] == 150
        # gemini-2.5-flash has a pricing entry, so cost must be populated.
        assert usage["cost_usd"] is not None
        assert usage["cost_usd"] > 0


class TestOTelCollector:
    @pytest.mark.asyncio
    async def test_captures_genai_spans(self) -> None:
        """Send an OTLP/JSON payload with GenAI spans and verify capture."""
        collector = OTelCollector(session_id="test")
        await collector.start()

        # Simulate an OTLP/HTTP export with a GenAI span
        otlp_payload = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "chat",
                                    "startTimeUnixNano": "1700000000000000000",
                                    "endTimeUnixNano": "1700000001500000000",
                                    "attributes": [
                                        {
                                            "key": "gen_ai.system",
                                            "value": {"stringValue": "anthropic"},
                                        },
                                        {
                                            "key": "gen_ai.request.model",
                                            "value": {
                                                "stringValue": "claude-haiku-4-5-20251001"
                                            },
                                        },
                                        {
                                            "key": "gen_ai.usage.input_tokens",
                                            "value": {"intValue": 42},
                                        },
                                        {
                                            "key": "gen_ai.usage.output_tokens",
                                            "value": {"intValue": 15},
                                        },
                                        {
                                            "key": "gen_ai.prompt",
                                            "value": {
                                                "stringValue": json.dumps(
                                                    [
                                                        {
                                                            "role": "user",
                                                            "content": "hello",
                                                        }
                                                    ]
                                                )
                                            },
                                        },
                                        {
                                            "key": "gen_ai.completion",
                                            "value": {
                                                "stringValue": json.dumps(
                                                    [{"type": "text", "text": "hi!"}]
                                                )
                                            },
                                        },
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{collector.endpoint}/v1/traces",
                json=otlp_payload,
                headers={"Content-Type": "application/json"},
            )
            assert resp.status_code == 200

        traj = collector.trajectory
        assert len(traj.exchanges) == 1
        assert traj.total_input_tokens == 42
        assert traj.total_output_tokens == 15

        ex = traj.exchanges[0]
        assert ex.request.body["model"] == "claude-haiku-4-5-20251001"
        assert ex.duration_ms == 1500.0

        assert len(collector.raw_spans) == 1

        await collector.stop()

    @pytest.mark.asyncio
    async def test_ignores_non_genai_spans(self) -> None:
        collector = OTelCollector(session_id="test")
        await collector.start()

        otlp_payload = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "http.request",
                                    "startTimeUnixNano": "1700000000000000000",
                                    "endTimeUnixNano": "1700000000100000000",
                                    "attributes": [
                                        {
                                            "key": "http.method",
                                            "value": {"stringValue": "GET"},
                                        }
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            await client.post(
                f"{collector.endpoint}/v1/traces",
                json=otlp_payload,
            )

        # Non-GenAI spans are stored raw but don't create exchanges
        assert len(collector.raw_spans) == 1
        assert len(collector.trajectory.exchanges) == 0

        await collector.stop()
