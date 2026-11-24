"""review F3: the model gate holds at the upstream send too.

The gate was checked only when the guard admitted a request. A request
admitted before the close could still send upstream after it: a retry
(BenchFlow's MiMo retry resends after 3 s and 10 s, LiteLLM's own), a
streaming request whose upstream call had not started, and a non-streaming
call already under way, which nothing cut. Now a send attempted after the
close is refused at the HTTP client, before any provider sees it, and the
close cuts every request still open, its upstream connection closed.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from benchflow.providers import litellm_request_guard as guard
from tests.test_en828_proxy_gate import (
    _receive_body_then_wait,
    _records,
    _scope,
    _sink,
)


@pytest.fixture
def gate_log(tmp_path, monkeypatch):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    monkeypatch.setattr(guard, "_GATE", guard._Gate())
    # The outbound check patches httpx's send for the process: put it back.
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(guard, "_WRAPPERS", set())
    guard.install_outbound_check()
    return log


async def test_a_retry_after_the_close_never_reaches_the_upstream(gate_log):
    """A provider answers 503, the client waits to try again, the gate closes
    meanwhile: the retry is refused at the send, never sent."""
    hits: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        return httpx.Response(503, json={"error": "busy"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(upstream), base_url="https://provider.test"
    ) as client:
        first = await client.post("/v1/chat/completions", json={"model": "m"})
        assert first.status_code == 503 and hits == ["/v1/chat/completions"]
        guard._close_gate("the wall-clock deadline")
        retry = await client.post("/v1/chat/completions", json={"model": "m"})
    assert hits == ["/v1/chat/completions"], "the retry reached the provider"
    assert retry.status_code == 400
    assert retry.json()["error"]["message"].startswith("trial_deadline_passed")

    with httpx.Client(transport=httpx.MockTransport(upstream)) as sync_client:
        again = sync_client.post("https://provider.test/v1/messages", json={})
    assert again.status_code == 400 and len(hits) == 1
    refused = [r for r in _records(gate_log) if r.get("event") == "refused"]
    assert refused and all(
        r.get("rule") == "agent-deadline" and r.get("at") == "upstream" for r in refused
    )


async def test_a_non_streaming_call_in_flight_at_the_close_is_cut(gate_log):
    """The provider has the request and has not answered; the call waits on
    it, not on the client. The close ends the call, closes its upstream
    connection, answers the agent trial_deadline_passed, and records it
    cut."""
    received = asyncio.Event()
    upstream_closed = asyncio.Event()

    async def provider(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        received.set()
        # Never answers; EOF is the client closing the connection.
        await reader.read()
        upstream_closed.set()
        writer.close()

    server = await asyncio.start_server(provider, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async def app(scope, receive, send):
        assert (await receive())["type"] == "http.request"
        async with httpx.AsyncClient(timeout=60) as client:
            answer = await client.post(
                f"http://127.0.0.1:{port}/v1/chat/completions", json={"model": "m"}
            )
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": answer.content})

    sent: list = []
    try:
        call = asyncio.create_task(
            guard.SingleRouteGuard(app)(
                _scope(), _receive_body_then_wait(), _sink(sent)
            )
        )
        await asyncio.wait_for(received.wait(), 10)
        guard._close_gate("the wall-clock deadline")
        await asyncio.wait_for(call, 5)
        await asyncio.wait_for(upstream_closed.wait(), 5)
    finally:
        server.close()
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 400
    assert json.loads(sent[1]["body"])["error"]["message"].startswith(
        "trial_deadline_passed"
    )
    events = _records(gate_log)
    cut = [e for e in events if e.get("event") == "deadline_cut"]
    ends = [e for e in events if e.get("event") == "request_end"]
    assert len(cut) == 1 and len(ends) == 1
    assert ends[0]["request_id"] == cut[0]["request_id"]
    assert ends[0]["response_complete"] is False
    assert any(e.get("event") == "upstream_start" for e in events)
