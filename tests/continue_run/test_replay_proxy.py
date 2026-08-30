"""Tests for the record-replay router, SSE reconstruction, and HTTP proxy."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from benchflow.continue_run import sandbox_replay_runtime as replay_runtime
from benchflow.continue_run.replay_proxy import (
    ReplayDivergenceError,
    ReplayProxy,
    ReplayRouter,
    completion_to_sse,
)
from benchflow.continue_run.sandbox_proxy import (
    SandboxReplayProxy,
    _ordered_live_exchange_log,
    sandbox_replay_runtime_source,
)
from benchflow.continue_run.sandbox_replay_runtime import (
    ReplayHandler,
    ReplayServer,
    ReplayState,
)
from benchflow.trajectories.redaction import canonical_redaction_module_source

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
    assert router.recorded_consumed_count == 1

    r2 = router.next_response({"messages": [{"role": "user"}]})
    assert r2.source == "replay"
    assert r2.body["choices"][0]["message"]["content"] == "second"
    assert router.exhausted is True
    assert router.recorded_consumed_count == 2

    r3 = router.next_response({"messages": [{"role": "user"}]})
    assert r3.source == "live"
    assert r3.body["choices"][0]["message"]["content"] == "LIVE"
    # the live exchange was captured for stitching
    assert len(router.live_exchanges) == 1
    assert router.live_attempt_count == 1
    assert router.live_errors == []
    assert len(live) == 1


def test_exhausted_without_forwarder_returns_error():
    router = ReplayRouter([exchange(completion(content="a"))], live_forwarder=None)
    router.next_response({"messages": [{}]})  # consume the one recorded
    result = router.next_response({"messages": [{}]})
    assert result.source == "error"
    assert result.status == 503
    assert router.live_attempt_count == 1
    assert router.live_errors


def test_live_forwarder_failure_retains_unpaired_attempt() -> None:
    """Guards PR #1057 against completing a lost host continuation call."""

    def fail(_request):
        raise RuntimeError("provider unavailable")

    router = ReplayRouter([], live_forwarder=fail)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        router.next_response({"messages": [{}]})

    assert router.live_attempt_count == 1
    assert router.live_exchanges == []
    assert router.live_errors == [
        "live provider request failed before a response could be captured"
    ]


def test_host_live_exchanges_preserve_attempt_order_under_concurrency() -> None:
    """Guards PR #1057 against stitching host calls in completion order."""

    first_started = threading.Event()
    release_first = threading.Event()

    def forward(request):
        request_id = request["request_id"]
        if request_id == 1:
            first_started.set()
            assert release_first.wait(timeout=5)
        return completion(content=f"live-{request_id}")

    router = ReplayRouter([], live_forwarder=forward)
    results: list[object] = []
    first = threading.Thread(
        target=lambda: results.append(router.next_response({"request_id": 1}))
    )
    second = threading.Thread(
        target=lambda: results.append(router.next_response({"request_id": 2}))
    )

    first.start()
    assert first_started.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    assert not second.is_alive()
    release_first.set()
    first.join(timeout=5)

    assert [row.request.body["request_id"] for row in router.live_exchanges] == [1, 2]
    assert [row.metadata["continuation_attempt"] for row in router.live_exchanges] == [
        1,
        2,
    ]


def test_sandbox_forwarding_failure_is_not_logged_as_provider_exchange(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1057 against labeling a synthesized sandbox 500 provider-wire."""

    state = ReplayState(
        recorded=[],
        upstream_url="https://provider.invalid/v1",
        upstream_api_key="test-key",
        upstream_model="openai/test-model",
        live_log_path=str(tmp_path / "live.jsonl"),
        state_path=str(tmp_path / "state.json"),
        port=61357,
    )

    def fail(*_args, **_kwargs):
        raise TimeoutError("provider unavailable")

    monkeypatch.setattr(replay_runtime.urllib.request, "urlopen", fail)
    source, status, body = state.next_response({"messages": [{"role": "user"}]})

    assert source == "live"
    assert status == 500
    assert "error" in body
    assert not (tmp_path / "live.jsonl").exists()
    capture_state = json.loads((tmp_path / "state.json").read_text())
    assert capture_state["live_attempt_count"] == 1
    assert capture_state["live_error_count"] == 1


def test_sandbox_recorded_prefix_is_journaled_before_response(tmp_path) -> None:
    """Guards PR #1057 against overstating a sandbox replay prefix."""

    state = ReplayState(
        recorded=[exchange(completion(content="recorded")).model_dump(mode="json")],
        upstream_url="https://provider.invalid/v1",
        upstream_api_key="test-key",
        upstream_model="openai/test-model",
        live_log_path=str(tmp_path / "live.jsonl"),
        state_path=str(tmp_path / "state.json"),
        port=61357,
    )

    source, status, _body = state.next_response({"messages": [{"role": "user"}]})

    assert source == "replay"
    assert status == 200
    capture_state = json.loads((tmp_path / "state.json").read_text())
    assert capture_state["recorded_consumed_count"] == 1
    assert capture_state["live_attempt_count"] == 0


def test_sandbox_live_exchange_is_redacted_before_journaling(tmp_path) -> None:
    """Guards PR #1057 against persisting raw continuation secrets."""

    live_log = tmp_path / "live.jsonl"
    state = ReplayState(
        recorded=[],
        upstream_url="https://provider.invalid/v1",
        upstream_api_key="test-key",
        upstream_model="openai/test-model",
        live_log_path=str(live_log),
        state_path=str(tmp_path / "state.json"),
        port=61357,
    )
    request_secret = "sk-ant-api03-requestsecret123456"
    response_secret = "sk-ant-api03-responsesecret123456"
    state._forward_live = lambda _request: (
        401,
        {"error": {"message": response_secret}},
        True,
    )

    _source, status, response = state.next_response(
        {"messages": [{"role": "user", "content": request_secret}]}
    )

    assert status == 401
    assert response["error"]["message"] == response_secret
    raw = live_log.read_text()
    assert request_secret not in raw
    assert response_secret not in raw
    assert raw.count("***REDACTED***") == 2


@pytest.mark.asyncio
async def test_sandbox_replay_uploads_canonical_redactor() -> None:
    """Guards PR #1057 against launching continuation without its redactor."""

    class FakeSandbox:
        def __init__(self) -> None:
            self.uploaded: dict[str, str] = {}

        async def exec(self, _command, **_kwargs):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def upload_file(self, source, target, *, mode):
            assert mode == "600"
            self.uploaded[target] = source.read_text()

    sandbox = FakeSandbox()
    proxy = await SandboxReplayProxy.start(
        sandbox=sandbox,
        recorded=[],
        upstream_url="https://provider.invalid/v1",
        upstream_api_key="test-key",
        upstream_model="openai/test-model",
    )

    redaction_path = f"{proxy.runtime_dir}/benchflow_trajectory_redaction.py"
    assert sandbox.uploaded[redaction_path] == canonical_redaction_module_source()
    assert (
        "from benchflow_trajectory_redaction import"
        in sandbox.uploaded[f"{proxy.runtime_dir}/replay_proxy.py"]
    )
    assert sandbox.uploaded[f"{proxy.runtime_dir}/replay_proxy.py"] == (
        sandbox_replay_runtime_source()
    )


def test_sandbox_attempt_journal_failure_invalidates_stale_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guards PR #1057 against completing an unjournaled sandbox call."""
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"live_attempt_count": 0, "live_error_count": 0}))
    state = ReplayState(
        recorded=[],
        upstream_url="https://provider.invalid/v1",
        upstream_api_key="test-key",
        upstream_model="openai/test-model",
        live_log_path=str(tmp_path / "live.jsonl"),
        state_path=str(state_path),
        port=61357,
    )

    def fail_replace(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(replay_runtime.os, "replace", fail_replace)

    with pytest.raises(OSError, match="disk full"):
        state.next_response({"messages": [{"role": "user"}]})

    assert state.live_attempt_count == 1
    assert state.live_error_count == 1
    assert not state_path.exists()


def test_sandbox_quiesce_waits_for_live_handler_before_snapshot(tmp_path) -> None:
    """Guards PR #1057 against snapshotting before live calls are quiescent."""
    state = ReplayState(
        recorded=[],
        upstream_url="https://provider.invalid/v1",
        upstream_api_key="test-key",
        upstream_model="openai/test-model",
        live_log_path=str(tmp_path / "live.jsonl"),
        state_path=str(tmp_path / "state.json"),
        port=61357,
    )
    forward_started = threading.Event()
    release_forward = threading.Event()
    responses: list[tuple] = []
    quiesced: list[bool] = []

    def forward(_request):
        forward_started.set()
        assert release_forward.wait(timeout=5)
        return 200, completion(content="done"), True

    state._forward_live = forward
    worker = threading.Thread(
        target=lambda: responses.append(
            state.next_response({"messages": [{"role": "user"}]})
        )
    )
    worker.start()
    assert forward_started.wait(timeout=5)
    barrier = threading.Thread(target=lambda: quiesced.append(state.quiesce(timeout=5)))
    barrier.start()
    assert barrier.is_alive()

    release_forward.set()
    worker.join(timeout=5)
    barrier.join(timeout=5)

    assert quiesced == [True]
    assert responses[0][0] == "live"
    assert state.live_attempt_count == 1
    assert len((tmp_path / "live.jsonl").read_text().splitlines()) == 1
    late = state.next_response({"messages": [{"role": "user"}]})
    assert late[0] == "error"
    assert late[1] == 503
    assert state.live_attempt_count == 2
    assert state.live_error_count == 1
    capture_state = json.loads((tmp_path / "state.json").read_text())
    assert capture_state["live_attempt_count"] == 2
    assert capture_state["live_error_count"] == 1


def test_sandbox_live_exchange_recovery_restores_attempt_order(tmp_path) -> None:
    """Guards PR #1057 against stitching sandbox calls in completion order."""

    live_log = tmp_path / "live.jsonl"
    state = ReplayState(
        recorded=[],
        upstream_url="https://provider.invalid/v1",
        upstream_api_key="test-key",
        upstream_model="openai/test-model",
        live_log_path=str(live_log),
        state_path=str(tmp_path / "state.json"),
        port=61357,
    )
    first_started = threading.Event()
    release_first = threading.Event()

    def forward(request):
        request_id = request["request_id"]
        if request_id == 1:
            first_started.set()
            assert release_first.wait(timeout=5)
        return 200, completion(content=f"live-{request_id}"), True

    state._forward_live = forward
    first = threading.Thread(target=lambda: state.next_response({"request_id": 1}))
    second = threading.Thread(target=lambda: state.next_response({"request_id": 2}))

    first.start()
    assert first_started.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    assert not second.is_alive()
    release_first.set()
    first.join(timeout=5)

    raw_rows = [json.loads(line) for line in live_log.read_text().splitlines()]
    assert [row["metadata"]["continuation_attempt"] for row in raw_rows] == [2, 1]
    exchanges, malformed = _ordered_live_exchange_log(live_log.read_text())
    assert malformed == 0
    assert [row.request.body["request_id"] for row in exchanges] == [1, 2]


def test_sandbox_quiesce_closes_listener_and_drains_accepted_handlers(
    tmp_path,
) -> None:
    """Guards PR #1057 against terminating a late rejected handler mid-journal."""

    state = ReplayState(
        recorded=[],
        upstream_url="https://provider.invalid/v1",
        upstream_api_key="test-key",
        upstream_model="openai/test-model",
        live_log_path=str(tmp_path / "live.jsonl"),
        state_path=str(tmp_path / "state.json"),
        port=0,
    )
    forward_started = threading.Event()
    release_forward = threading.Event()

    def forward(_request):
        forward_started.set()
        assert release_forward.wait(timeout=5)
        return 200, completion(content="done"), True

    state._forward_live = forward
    server = ReplayServer(("127.0.0.1", 0), ReplayHandler, state)
    port = server.server_address[1]
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    chat_responses: list[httpx.Response] = []
    chat_thread = threading.Thread(
        target=lambda: chat_responses.append(
            httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={"messages": [{"role": "user"}]},
                timeout=5,
            )
        )
    )
    chat_thread.start()
    assert forward_started.wait(timeout=5)
    quiesce_responses: list[httpx.Response] = []
    quiesce_thread = threading.Thread(
        target=lambda: quiesce_responses.append(
            httpx.post(
                f"http://127.0.0.1:{port}/benchflow/quiesce",
                timeout=5,
            )
        )
    )
    quiesce_thread.start()
    assert quiesce_thread.is_alive()

    release_forward.set()
    chat_thread.join(timeout=5)
    quiesce_thread.join(timeout=5)
    server_thread.join(timeout=5)

    assert chat_responses[0].status_code == 200
    assert quiesce_responses[0].status_code == 200
    assert state.active_handlers == 0
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)


@pytest.mark.asyncio
async def test_sandbox_attempt_journal_marker_reaches_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards PR #1057 by surfacing the sandbox journal marker at teardown."""
    proxy = SandboxReplayProxy(
        sandbox=object(),
        runtime_dir="/tmp/runtime",
        port=61357,
        pid_path="/tmp/runtime/pid",
        live_log_path="/tmp/runtime/live.jsonl",
        state_path="/tmp/runtime/state.json",
        stdout_path="/tmp/runtime/stdout.log",
        stderr_path="/tmp/runtime/stderr.log",
    )

    async def read_remote_text(_sandbox, path, **_kwargs):
        assert path == proxy.stderr_path
        return "BENCHFLOW_CAPTURE_STATE_WRITE_FAILED\n"

    monkeypatch.setattr(
        "benchflow.continue_run.sandbox_proxy._read_remote_text", read_remote_text
    )

    await proxy._load_runtime_errors()

    assert proxy.live_errors == ["sandbox live attempt journal failed 1 time(s)"]


@pytest.mark.asyncio
async def test_sandbox_replay_count_is_recovered_by_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guards PR #1057 against stitching unconsumed sandbox responses."""

    proxy = SandboxReplayProxy(
        sandbox=object(),
        runtime_dir="/tmp/runtime",
        port=61357,
        pid_path="/tmp/runtime/pid",
        live_log_path="/tmp/runtime/live.jsonl",
        state_path="/tmp/runtime/state.json",
        stdout_path="/tmp/runtime/stdout.log",
        stderr_path="/tmp/runtime/stderr.log",
        recorded_exchange_count=3,
    )

    async def read_remote_text(_sandbox, path, **_kwargs):
        assert path == proxy.state_path
        return json.dumps(
            {
                "recorded_consumed_count": 2,
                "live_attempt_count": 0,
                "live_error_count": 0,
            }
        )

    monkeypatch.setattr(
        "benchflow.continue_run.sandbox_proxy._read_remote_text", read_remote_text
    )

    await proxy._load_live_state()

    assert proxy.recorded_consumed_count == 2
    assert proxy.live_errors == []


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
