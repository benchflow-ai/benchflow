"""Tests for trajectory proxy, OTel collector, and types."""

import asyncio
import json

import httpx
import pytest

from benchflow.trajectories.otel import OTelCollector
from benchflow.trajectories.proxy import TrajectoryProxy
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


class TestTrajectoryProxy:
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
