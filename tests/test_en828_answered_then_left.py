"""A client that leaves after its provider's whole answer reached the proxy.

Claude Code asks for a session title just after its last reply, and the
agent's stop can land while that call is open. In the 0.10.0-final gate the
provider had answered in full (200, 856 bytes, its usage in the stream) when
the client left; LiteLLM was told the client left, logged the call as a stream
cut short with its own estimate, and the trial read usage_incomplete with the
provider's count in hand. The guard now keeps a client's departure from
LiteLLM once the provider's answer is whole, so LiteLLM ends the stream it
holds and logs the provider's usage. A call whose provider had not answered
is still cut and still counted as an estimate; the deadline's gate still cuts
whatever is open.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from benchflow.providers import litellm_request_guard as guard
from tests.test_en828_proxy_gate import _records, _scope, _sink

BODY = json.dumps({"model": "m", "messages": [], "max_tokens": 8}).encode()
# A Messages API answer as the provider streams it: usage in message_start
# and message_delta, message_stop last.
ANSWER = [
    b'event: message_start\ndata: {"type":"message_start","message":{"usage":'
    b'{"input_tokens":0,"output_tokens":1}}}\n\n',
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"event: message_stop"}}\n\n',
    b'event: message_delta\ndata: {"type":"message_delta","delta":'
    b'{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]


@pytest.fixture
def gate_log(tmp_path, monkeypatch):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    monkeypatch.setattr(guard, "_GATE", guard._Gate())
    monkeypatch.setattr(httpx.AsyncClient, "send", httpx.AsyncClient.send)
    monkeypatch.setattr(httpx.Client, "send", httpx.Client.send)
    monkeypatch.setattr(guard, "_WRAPPERS", set())
    guard.install_outbound_check()
    return log


def _client(leave: asyncio.Event):
    """The agent: sends its request, then leaves when ``leave`` is set."""
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": BODY, "more_body": False}
        await leave.wait()
        return {"type": "http.disconnect"}

    return receive


def _litellm(provider_parts, leave: asyncio.Event, read_all: bool, outcome: dict):
    """LiteLLM standing in: calls the provider, then streams to its client,
    and ends the call as cut if it is told the client left."""

    async def provider(request: httpx.Request) -> httpx.Response:
        async def body():
            for part in provider_parts:
                yield part
            if not read_all:
                await asyncio.Event().wait()  # the provider is still writing

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()
        )

    async def app(scope, receive, send):
        assert (await receive())["type"] == "http.request"
        async with (
            httpx.AsyncClient(
                transport=httpx.MockTransport(provider),
                base_url="https://provider.test",
            ) as client,
            client.stream("POST", "/v1/messages", json={}) as upstream,
        ):
            got = []
            async for part in upstream.aiter_raw():
                got.append(part)
                if not read_all and len(got) == len(provider_parts):
                    break
            leave.set()  # the agent is stopped now
            try:
                message = await asyncio.wait_for(receive(), 3.0)
            except TimeoutError:
                message = None
            if message is not None and message["type"] == "http.disconnect":
                outcome["ended"] = "cut"
                return
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"".join(got)})
            outcome["ended"] = "logged whole"

    return app


async def test_a_client_that_leaves_after_the_whole_answer_does_not_cut_it(gate_log):
    leave, outcome = asyncio.Event(), {}
    route = guard.SingleRouteGuard(_litellm(ANSWER, leave, True, outcome))
    sent: list = []
    await asyncio.wait_for(route(_scope(), _client(leave), _sink(sent)), 10)
    assert outcome == {"ended": "logged whole"}, "LiteLLM was told the client left"
    records = _records(gate_log)
    (upstream,) = [r for r in records if r["event"] == "upstream_end"]
    assert upstream["status"] == 200 and upstream["answered"] is True
    (end,) = [r for r in records if r["event"] == "request_end"]
    assert end["response_complete"] is True and end["status"] == 200
    assert end["client_left"] >= 0


async def test_a_client_that_leaves_before_the_answer_still_cuts_it(gate_log):
    """The provider is still writing (no message_stop; the text that quotes
    one is inside a JSON string): the call is cut, as before."""
    leave, outcome = asyncio.Event(), {}
    route = guard.SingleRouteGuard(_litellm(ANSWER[:3], leave, False, outcome))
    sent: list = []
    await asyncio.wait_for(route(_scope(), _client(leave), _sink(sent)), 10)
    assert outcome == {"ended": "cut"}
    records = _records(gate_log)
    (end,) = [r for r in records if r["event"] == "request_end"]
    assert end["response_complete"] is False and end["client_left"] >= 0
    (upstream,) = [r for r in records if r["event"] == "upstream_end"]
    assert upstream["answered"] is False


async def test_the_deadline_still_cuts_a_call_whose_client_left(gate_log):
    leave, outcome = asyncio.Event(), {}

    async def app(scope, receive, send):
        assert (await receive())["type"] == "http.request"
        guard._REQUEST.get()["answered"] = True  # the provider's answer is in
        leave.set()
        task = asyncio.ensure_future(receive())
        await asyncio.sleep(0.2)
        assert not task.done(), "the client's departure reached LiteLLM"
        guard._close_gate("the wall-clock deadline")
        outcome["after_close"] = (await asyncio.wait_for(task, 2))["type"]

    route = guard.SingleRouteGuard(app)
    await asyncio.wait_for(route(_scope(), _client(leave), _sink([])), 10)
    assert outcome == {"after_close": "http.disconnect"}


def _arriving(gap: float, leave: asyncio.Event, outcome: dict):
    """LiteLLM standing in, while the provider's answer is still arriving:
    its first part is in when the client leaves, the rest comes ``gap``
    seconds later. It forwards the stream and listens for the client at
    once, as Starlette's streaming response does."""

    async def provider(request: httpx.Request) -> httpx.Response:
        async def body():
            yield ANSWER[0]
            await asyncio.sleep(gap)
            for part in ANSWER[1:]:
                yield part

        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body()
        )

    async def app(scope, receive, send):
        assert (await receive())["type"] == "http.request"
        async with (
            httpx.AsyncClient(
                transport=httpx.MockTransport(provider),
                base_url="https://provider.test",
            ) as client,
            client.stream("POST", "/v1/messages", json={}) as upstream,
        ):
            parts = upstream.aiter_raw()
            got = [await parts.__anext__()]
            leave.set()  # the agent is stopped now
            listener = asyncio.ensure_future(receive())

            async def pull():
                async for part in parts:
                    got.append(part)

            puller = asyncio.ensure_future(pull())
            done, _ = await asyncio.wait(
                {listener, puller}, return_when=asyncio.FIRST_COMPLETED
            )
            if listener in done and listener.result()["type"] == "http.disconnect":
                puller.cancel()
                outcome["ended"] = "cut"
                return
            listener.cancel()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"".join(got)})
            outcome["ended"] = "logged whole"

    return app


async def test_an_answer_still_arriving_when_the_client_leaves_is_read_whole(
    gate_log,
):
    """The provider finished; its last parts reach the proxy 0.3 s after the
    client left. The guard waits for them (at most _ANSWER_GRACE_SEC) before
    it would tell LiteLLM, so the provider's own count is logged."""
    leave, outcome = asyncio.Event(), {}
    route = guard.SingleRouteGuard(_arriving(0.3, leave, outcome))
    await asyncio.wait_for(route(_scope(), _client(leave), _sink([])), 10)
    assert outcome == {"ended": "logged whole"}
    (end,) = [r for r in _records(gate_log) if r["event"] == "request_end"]
    assert end["response_complete"] is True and end["client_left"] >= 0


async def test_a_provider_still_writing_past_the_grace_is_cut(gate_log):
    leave, outcome = asyncio.Event(), {}
    route = guard.SingleRouteGuard(_arriving(3.0, leave, outcome))
    await asyncio.wait_for(route(_scope(), _client(leave), _sink([])), 10)
    assert outcome == {"ended": "cut"}
