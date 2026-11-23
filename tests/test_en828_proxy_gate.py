"""the model proxy's one-way gate, closed at the wall-clock deadline.

A request the guard admitted before the close is cut (its ``receive()`` says
the client left, so LiteLLM ends the upstream stream); one that arrives after
it is refused with a 400 ``trial_deadline_passed`` and never reaches LiteLLM.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path

import pytest

from benchflow.providers import litellm_request_guard as guard

BODY = json.dumps(
    {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
).encode()
#: The OpenAI wires (chat completions, Responses) that most harnesses speak.
OPENAI = {
    "/v1/chat/completions": {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    },
    "/v1/responses": {"model": "m", "input": "hi", "stream": True},
}


def _scope(path: str = "/v1/messages") -> dict:
    return {
        "type": "http",
        "path": path,
        "method": "POST",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }


def _receive_body_then_wait(body: bytes = BODY):
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()

    return receive


def _sink(into: list):
    async def send(message):
        into.append(message)

    return send


def _records(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


@pytest.fixture
def gate_log(tmp_path, monkeypatch):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    monkeypatch.setattr(guard, "_GATE", guard._Gate())
    return log


@pytest.mark.asyncio
async def test_a_request_before_the_close_is_cut_and_one_after_it_is_refused(gate_log):
    seen: list[str] = []
    a_got: asyncio.Future = asyncio.get_running_loop().create_future()

    async def app(scope, receive, send):
        seen.append(scope["path"])
        first = await receive()
        assert first["type"] == "http.request"
        a_got.set_result(await receive())

    route = guard.SingleRouteGuard(app)
    a_sent: list = []
    a = asyncio.create_task(route(_scope(), _receive_body_then_wait(), _sink(a_sent)))
    for _ in range(100):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert seen == ["/v1/messages"]
    guard._close_gate("the wall-clock deadline")
    assert await asyncio.wait_for(a_got, 2) == {"type": "http.disconnect"}
    await asyncio.wait_for(a, 2)

    b_sent: list = []
    await route(_scope(), _receive_body_then_wait(), _sink(b_sent))
    assert seen == ["/v1/messages"], "the refused request reached LiteLLM"
    assert b_sent[0]["type"] == "http.response.start" and b_sent[0]["status"] == 400
    error = json.loads(b_sent[1]["body"])["error"]
    assert error["message"].startswith("trial_deadline_passed")
    assert error["type"] == "invalid_request_error"

    events = [(r["event"], r.get("rule")) for r in _records(gate_log)]
    assert ("gate_closed", None) in events
    assert ("deadline_cut", None) in events
    assert ("refused", "agent-deadline") in events
    ack = json.loads(Path(f"{gate_log}.gate").read_text())
    assert ack["closed_at"] and ack["open"] == 1 and ack["pid"] == os.getpid()


@pytest.mark.asyncio
async def test_the_gate_is_one_way_and_answers_without_any_request(gate_log):
    guard._close_gate("first")
    first = json.loads(Path(f"{gate_log}.gate").read_text())
    guard._close_gate("second")
    again = json.loads(Path(f"{gate_log}.gate").read_text())
    assert first == again and first["open"] == 0 and first["reason"] == "first"
    sent: list = []

    async def app(scope, receive, send):  # pragma: no cover - must not run
        raise AssertionError("LiteLLM saw a request after the close")

    await guard.SingleRouteGuard(app)(_scope(), _receive_body_then_wait(), _sink(sent))
    assert sent[0]["status"] == 400


@pytest.mark.asyncio
async def test_sigusr2_closes_the_gate_and_writes_the_ack(gate_log):
    previous = signal.getsignal(signal.SIGUSR2)
    try:
        guard._install_gate_signal()
        assert Path(f"{gate_log}.gate-armed").is_file()
        os.kill(os.getpid(), signal.SIGUSR2)
        ack = Path(f"{gate_log}.gate")
        for _ in range(100):
            if ack.is_file():
                break
            await asyncio.sleep(0.01)
        assert json.loads(ack.read_text())["closed_at"]
        sent: list = []

        async def app(scope, receive, send):  # pragma: no cover - must not run
            raise AssertionError("LiteLLM saw a request after the close")

        await guard.SingleRouteGuard(app)(
            _scope(), _receive_body_then_wait(), _sink(sent)
        )
        assert sent[0]["status"] == 400
    finally:
        signal.signal(signal.SIGUSR2, previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(OPENAI))
async def test_an_openai_request_is_cut_before_and_refused_after_the_close(
    gate_log, path
):
    body = json.dumps(OPENAI[path]).encode()
    seen: list[str] = []
    got: asyncio.Future = asyncio.get_running_loop().create_future()

    async def app(scope, receive, send):
        seen.append(scope["path"])
        assert (await receive())["type"] == "http.request"
        got.set_result(await receive())

    route = guard.SingleRouteGuard(app)
    first = asyncio.create_task(
        route(_scope(path), _receive_body_then_wait(body), _sink([]))
    )
    for _ in range(100):
        if seen:
            break
        await asyncio.sleep(0.01)
    assert seen == [path], "the guard refused a valid OpenAI request"
    guard._close_gate("the wall-clock deadline")
    assert await asyncio.wait_for(got, 2) == {"type": "http.disconnect"}
    await asyncio.wait_for(first, 2)
    sent: list = []
    await route(_scope(path), _receive_body_then_wait(body), _sink(sent))
    assert seen == [path] and sent[0]["status"] == 400
    assert json.loads(sent[1]["body"])["error"]["message"].startswith(
        "trial_deadline_passed"
    )
