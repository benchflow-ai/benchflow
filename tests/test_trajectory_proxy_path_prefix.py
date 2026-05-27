"""Regression tests for external usage-proxy path-prefix routing."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from benchflow.trajectories.proxy import TrajectoryProxy


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, bytes]:
    request_line = (await reader.readline()).decode()
    method, path, _version = request_line.strip().split(" ", 2)
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        key, _, value = line.decode().partition(":")
        headers[key.lower().strip()] = value.strip()
    content_length = int(headers.get("content-length", "0"))
    body = await reader.readexactly(content_length) if content_length else b""
    return method, path, body


@pytest.mark.asyncio
async def test_path_prefix_gates_health_and_provider_requests():
    """Guards PR #568: external tunnel traffic must match the secret path prefix."""
    upstream_requests: list[tuple[str, str, bytes]] = []

    async def upstream_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        method, path, body = await _read_request(reader)
        upstream_requests.append((method, path, body))
        response_body = json.dumps(
            {
                "id": "msg_1",
                "model": "gpt-4.1-mini",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 5,
                    "total_tokens": 8,
                },
            },
            separators=(",", ":"),
        ).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"content-type: application/json\r\n"
            + f"content-length: {len(response_body)}\r\n\r\n".encode()
            + response_body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
    upstream_port = upstream.sockets[0].getsockname()[1]
    proxy = TrajectoryProxy(
        target=f"http://127.0.0.1:{upstream_port}",
        session_id="path-prefix",
        agent_name="openhands",
        path_prefix="/__benchflow/abc123",
    )
    await proxy.start()

    try:
        async with httpx.AsyncClient() as client:
            unprefixed_health = await client.get(f"{proxy.base_url}/__benchflow_health")
            assert unprefixed_health.status_code == 404
            assert upstream_requests == []

            prefixed_head = await client.head(
                f"{proxy.base_url}/__benchflow/abc123/__benchflow_health"
            )
            assert prefixed_head.status_code == 200
            assert prefixed_head.content == b""
            assert upstream_requests == []

            prefixed_health = await client.get(
                f"{proxy.base_url}/__benchflow/abc123/__benchflow_health"
            )
            assert prefixed_health.status_code == 200
            assert prefixed_health.json() == {"status": "ok"}
            assert upstream_requests == []

            unprefixed_provider = await client.post(
                f"{proxy.base_url}/v1/messages",
                json={"model": "gpt-4.1-mini"},
            )
            assert unprefixed_provider.status_code == 404
            assert upstream_requests == []

            prefixed_provider = await client.post(
                f"{proxy.base_url}/__benchflow/abc123/v1/messages?trace=1",
                json={"model": "gpt-4.1-mini"},
            )
            assert prefixed_provider.status_code == 200

        assert len(upstream_requests) == 1
        method, path, body = upstream_requests[0]
        assert method == "POST"
        assert path == "/v1/messages?trace=1"
        assert json.loads(body) == {"model": "gpt-4.1-mini"}
        assert proxy.trajectory.exchanges[0].request.path == "/v1/messages?trace=1"
    finally:
        await proxy.stop()
        upstream.close()
        await upstream.wait_closed()
