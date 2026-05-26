"""Tests for the Bedrock proxy HTTP server."""

from __future__ import annotations

import asyncio
import json
from io import BytesIO

import httpx
import pytest

import benchflow.providers.bedrock_proxy as bedrock_proxy
from benchflow.providers.bedrock_proxy import BedrockProxyServer


class FakeBedrockClient:
    def converse(self, **kwargs):
        if "anthropic" in kwargs["modelId"]:
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "hello from claude"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7},
            }
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hello from codex"}],
                }
            },
            "usage": {"inputTokens": 5, "outputTokens": 6, "totalTokens": 11},
        }

    def count_tokens(self, **kwargs):
        return {"inputTokens": 42}

    def invoke_model(self, **kwargs):
        if "anthropic" in kwargs["modelId"]:
            payload = {
                "id": "msg_bedrock",
                "type": "message",
                "role": "assistant",
                "model": kwargs["modelId"],
                "content": [{"type": "text", "text": "hello from claude"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        else:
            payload = {
                "id": "msg_bedrock",
                "type": "message",
                "role": "assistant",
                "model": kwargs["modelId"],
                "content": [{"type": "text", "text": "hello from codex"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        return {"body": BytesIO(json.dumps(payload).encode())}

    def invoke_model_with_response_stream(self, **kwargs):
        return {
            "body": iter(
                [
                    {
                        "chunk": {
                            "bytes": json.dumps(
                                {
                                    "type": "message_start",
                                    "message": {
                                        "id": "msg_bedrock",
                                        "type": "message",
                                        "role": "assistant",
                                        "model": kwargs["modelId"],
                                        "content": [],
                                        "stop_reason": None,
                                        "stop_sequence": None,
                                        "usage": {
                                            "input_tokens": 0,
                                            "output_tokens": 0,
                                        },
                                    },
                                }
                            ).encode()
                        }
                    },
                    {
                        "chunk": {
                            "bytes": json.dumps(
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                }
                            ).encode()
                        }
                    },
                    {
                        "chunk": {
                            "bytes": json.dumps(
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": "hello"},
                                }
                            ).encode()
                        }
                    },
                    {
                        "chunk": {
                            "bytes": json.dumps(
                                {
                                    "type": "content_block_stop",
                                    "index": 0,
                                }
                            ).encode()
                        }
                    },
                    {
                        "chunk": {
                            "bytes": json.dumps(
                                {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": "end_turn"},
                                    "usage": {"output_tokens": 1},
                                }
                            ).encode()
                        }
                    },
                    {"chunk": {"bytes": json.dumps({"type": "message_stop"}).encode()}},
                ]
            )
        }


class FakeUnsupportedCountTokensBedrockClient(FakeBedrockClient):
    def count_tokens(self, **kwargs):
        raise RuntimeError(
            "An error occurred (ValidationException) when calling the CountTokens "
            "operation: The provided model doesn't support counting tokens."
        )


class EndpointConnectionError(Exception):
    pass


class FakeFlakyBedrockClient(FakeBedrockClient):
    def __init__(self) -> None:
        self.converse_calls = 0

    def converse(self, **kwargs):
        self.converse_calls += 1
        if self.converse_calls == 1:
            raise EndpointConnectionError("temporary dns failure")
        return super().converse(**kwargs)


class FakeBedrockHTTPClient:
    async def post(self, url, *, content, headers):
        assert headers["authorization"] == "Bearer bedrock-token"
        if url.endswith("/invoke"):
            payload = {
                "id": "msg_bedrock",
                "type": "message",
                "role": "assistant",
                "model": url.split("/model/", 1)[1].rsplit("/", 1)[0],
                "content": [{"type": "text", "text": "hello from invoke"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
            return httpx.Response(
                200,
                content=json.dumps(payload).encode(),
                headers={"content-type": "application/json"},
            )
        raise AssertionError(f"unexpected URL: {url}")

    def build_request(self, method, url, *, content, headers):
        return httpx.Request(method, url, content=content, headers=headers)

    async def send(self, request, *, stream):
        assert stream is True
        assert request.headers["authorization"] == "Bearer bedrock-token"
        return httpx.Response(
            200,
            content=b"raw-eventstream-data",
            headers={"content-type": "application/vnd.amazon.eventstream"},
            request=request,
        )


@pytest.mark.asyncio
async def test_bedrock_proxy_healthz_and_messages_route():
    server = BedrockProxyServer(
        host="127.0.0.1",
        port=0,
        client=FakeBedrockClient(),
        backend_model="anthropic.claude-haiku-4-5-20251001-v1:0",
        frontend_model="anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    await server.start()
    task = asyncio.create_task(server.serve_forever())
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}"
        ) as client:
            health = await client.get("/healthz")
            assert health.json() == {"ok": True}
            models = await client.get("/v1/models")
            assert (
                models.json()["data"][0]["id"]
                == "anthropic.claude-haiku-4-5-20251001-v1:0"
            )

            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                    ],
                    "max_tokens": 32,
                },
            )
            body = resp.json()
            assert body["content"][0]["text"] == "hello from claude"
            assert body["usage"]["output_tokens"] == 4
    finally:
        task.cancel()
        await server.stop()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_bedrock_proxy_responses_route():
    server = BedrockProxyServer(
        host="127.0.0.1",
        port=0,
        client=FakeBedrockClient(),
        backend_model="openai.gpt-oss-20b-1:0",
    )
    await server.start()
    task = asyncio.create_task(server.serve_forever())
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}"
        ) as client:
            resp = await client.post(
                "/v1/responses",
                json={
                    "model": "openai.gpt-oss-20b-1:0",
                    "input": [
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hi"}],
                        }
                    ],
                    "max_output_tokens": 32,
                },
            )
            body = resp.json()
            assert body["output_text"] == "hello from codex"
            assert body["usage"]["total_tokens"] == 11
    finally:
        task.cancel()
        await server.stop()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_bedrock_proxy_retries_transient_runtime_transport_errors(monkeypatch):
    """Guards v0.5-integration@e55219d against transient Bedrock DNS failures invalidating trials."""
    monkeypatch.setattr(bedrock_proxy, "_BEDROCK_RUNTIME_RETRY_DELAYS_SEC", (0, 0, 0))
    fake_client = FakeFlakyBedrockClient()
    server = BedrockProxyServer(
        host="127.0.0.1",
        port=0,
        client=fake_client,
        backend_model="anthropic.claude-haiku-4-5-20251001-v1:0",
        frontend_model="anthropic.claude-haiku-4-5-20251001-v1:0",
    )
    await server.start()
    task = asyncio.create_task(server.serve_forever())
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}"
        ) as client:
            resp = await client.post(
                "/v1/messages",
                json={
                    "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
                    ],
                    "max_tokens": 32,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["content"][0]["text"] == "hello from claude"
            assert fake_client.converse_calls == 2
    finally:
        task.cancel()
        await server.stop()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_bedrock_proxy_invoke_and_count_tokens_routes(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    server = BedrockProxyServer(
        host="127.0.0.1",
        port=0,
        client=FakeBedrockClient(),
        backend_model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        bedrock_http=FakeBedrockHTTPClient(),
        runtime_env={
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_REGION": "us-east-1",
        },
    )
    await server.start()
    task = asyncio.create_task(server.serve_forever())
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}"
        ) as client:
            invoke = await client.post(
                "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke",
                json={
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hi"}],
                        }
                    ],
                    "max_tokens": 32,
                },
            )
            assert invoke.json()["content"][0]["text"] == "hello from invoke"

            count_tokens = await client.post(
                "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/count-tokens",
                json={
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hi"}],
                        }
                    ],
                    "max_tokens": 32,
                },
            )
            assert count_tokens.json()["input_tokens"] == 42

            stream = await client.post(
                "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke-with-response-stream",
                json={
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hi"}],
                        }
                    ],
                    "max_tokens": 32,
                },
            )
            assert stream.text == "raw-eventstream-data"
    finally:
        task.cancel()
        await server.stop()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_bedrock_proxy_count_tokens_falls_back_to_invoke_usage(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    server = BedrockProxyServer(
        host="127.0.0.1",
        port=0,
        client=FakeUnsupportedCountTokensBedrockClient(),
        backend_model="us.anthropic.claude-sonnet-4-6",
        runtime_env={
            "AWS_BEARER_TOKEN_BEDROCK": "bedrock-token",
            "AWS_REGION": "us-east-1",
        },
    )
    await server.start()
    task = asyncio.create_task(server.serve_forever())
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{server.port}"
        ) as client:
            resp = await client.post(
                "/model/us.anthropic.claude-sonnet-4-6/count-tokens",
                json={
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hi"}],
                        }
                    ],
                    "max_tokens": 32,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["input_tokens"] == 3
    finally:
        task.cancel()
        await server.stop()
        with pytest.raises(asyncio.CancelledError):
            await task
