"""Tests for the record-replay router, SSE reconstruction, and HTTP proxy."""

from __future__ import annotations

import json

import httpx
import pytest

from benchflow.continue_run.replay_proxy import (
    ReplayDivergenceError,
    ReplayProxy,
    ReplayRouter,
    completion_to_sse,
)

from ._helpers import completion, exchange

# ── ReplayRouter ──────────────────────────────────────────────────────────


def test_serves_recorded_responses_in_order_then_live():
    recorded = [
        exchange(completion(content="first")),
        exchange(completion(content="second")),
    ]
    live = []

    def forwarder(req):
        live.append(req)
        return completion(content="LIVE")

    router = ReplayRouter(recorded, live_forwarder=forwarder)

    r1 = router.next_response({"messages": [{"role": "user"}]})
    assert r1.source == "replay"
    assert r1.body["choices"][0]["message"]["content"] == "first"

    r2 = router.next_response({"messages": [{"role": "user"}]})
    assert r2.source == "replay"
    assert r2.body["choices"][0]["message"]["content"] == "second"
    assert router.exhausted is True

    r3 = router.next_response({"messages": [{"role": "user"}]})
    assert r3.source == "live"
    assert r3.body["choices"][0]["message"]["content"] == "LIVE"
    # the live exchange was captured for stitching
    assert len(router.live_exchanges) == 1
    assert len(live) == 1


def test_exhausted_without_forwarder_returns_error():
    router = ReplayRouter([exchange(completion(content="a"))], live_forwarder=None)
    router.next_response({"messages": [{}]})  # consume the one recorded
    result = router.next_response({"messages": [{}]})
    assert result.source == "error"
    assert result.status == 503


def test_divergence_warns_by_default():
    recorded = [exchange(completion(content="a"), n_request_messages=3)]
    router = ReplayRouter(recorded)
    # agent sends 2 messages where the recorded turn had 3
    router.next_response({"messages": [{}, {}]})
    assert router.divergences == 1


def test_divergence_strict_raises():
    recorded = [exchange(completion(content="a"), n_request_messages=3)]
    router = ReplayRouter(recorded, strict_divergence=True)
    with pytest.raises(ReplayDivergenceError):
        router.next_response({"messages": [{}, {}]})


def test_recorded_failure_passed_through():
    recorded = [exchange({"error": {"message": "boom"}}, status=500)]
    router = ReplayRouter(recorded)
    result = router.next_response({"messages": [{}]})
    assert result.status == 500
    assert result.body["error"]["message"] == "boom"


# ── SSE reconstruction ────────────────────────────────────────────────────


def test_completion_to_sse_content_and_tools():
    body = completion(
        content="hello",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"cmd":"ls"}'},
            }
        ],
    )
    payloads = completion_to_sse(body)
    chunks = [json.loads(p) for p in payloads]

    # first chunk announces the assistant role
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    # content delta present
    assert any(c["choices"][0]["delta"].get("content") == "hello" for c in chunks)
    # tool call delta carries the full function name + arguments
    tool_deltas = [
        c["choices"][0]["delta"]["tool_calls"][0]
        for c in chunks
        if c["choices"][0]["delta"].get("tool_calls")
    ]
    assert tool_deltas[0]["function"]["name"] == "bash"
    assert tool_deltas[0]["function"]["arguments"] == '{"cmd":"ls"}'
    # final chunk carries finish_reason + usage
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["usage"]["total_tokens"] == 2


# ── HTTP proxy end-to-end (real server, httpx client) ─────────────────────


@pytest.fixture()
def proxy_with(request):
    proxies: list[ReplayProxy] = []

    def _make(router: ReplayRouter) -> ReplayProxy:
        proxy = ReplayProxy(router, host="127.0.0.1", port=0).start()
        proxies.append(proxy)
        return proxy

    yield _make
    for p in proxies:
        p.stop()


def test_http_non_stream_serves_recorded_then_live(proxy_with):
    recorded = [exchange(completion(content="r1"))]
    router = ReplayRouter(recorded, live_forwarder=lambda req: completion(content="L1"))
    proxy = proxy_with(router)

    with httpx.Client(base_url=proxy.base_url, timeout=10) as client:
        resp1 = client.post("/chat/completions", json={"messages": [{"role": "user"}]})
        assert resp1.status_code == 200
        assert resp1.json()["choices"][0]["message"]["content"] == "r1"

        resp2 = client.post("/chat/completions", json={"messages": [{"role": "user"}]})
        assert resp2.json()["choices"][0]["message"]["content"] == "L1"


def test_http_stream_emits_sse(proxy_with):
    recorded = [exchange(completion(content="streamed"))]
    proxy = proxy_with(ReplayRouter(recorded))

    with (
        httpx.Client(base_url=proxy.base_url, timeout=10) as client,
        client.stream(
            "POST",
            "/chat/completions",
            json={"messages": [{"role": "user"}], "stream": True},
        ) as resp,
    ):
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [ln for ln in resp.iter_lines() if ln.startswith("data:")]

    assert lines[-1].strip() == "data: [DONE]"
    payloads = [
        json.loads(ln[len("data: ") :]) for ln in lines if not ln.endswith("[DONE]")
    ]
    assert any(p["choices"][0]["delta"].get("content") == "streamed" for p in payloads)


def test_http_health_and_models(proxy_with):
    proxy = proxy_with(ReplayRouter([exchange(completion(content="a"))]))
    with httpx.Client(base_url=proxy.base_url, timeout=10) as client:
        assert client.get("/health").status_code == 200
        models = client.get("/models").json()
        assert models["object"] == "list"
