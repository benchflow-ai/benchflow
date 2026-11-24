"""The normal stop of an agent waits for its unanswered model calls.

Claude Code asks for a session title just after its last reply. When the
agent's stop (``Rollout.disconnect``) lands before the provider has answered
it, stopping the agent cuts the call: LiteLLM logs a stream its client left,
with its own estimate, and the trial's usage reads incomplete (the 0.10.0-final
gate's eval-wrong-answer). The stop now waits, bounded, until no call is open
without its provider's whole answer, then stops the agent: a wait only, nothing
refused, nothing held. A deadline's or a budget's stop does not wait; there the
model gate cuts at once.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.providers import litellm_request_guard as guard
from tests.test_en828_proxy_gate import _scope, _sink

BODY = json.dumps({"model": "m", "messages": [], "max_tokens": 8}).encode()


@pytest.fixture
def gate_log(tmp_path, monkeypatch):
    log = tmp_path / "callback.jsonl"
    monkeypatch.setenv("BENCHFLOW_LITELLM_LOG_PATH", str(log))
    monkeypatch.delenv("BENCHFLOW_LITELLM_DISALLOW_WEB_TOOLS", raising=False)
    monkeypatch.setattr(guard, "_GATE", guard._Gate())
    usr1, usr2 = signal.getsignal(signal.SIGUSR1), signal.getsignal(signal.SIGUSR2)
    guard._install_tally_signal()
    guard._install_gate_signal()
    yield log
    signal.signal(signal.SIGUSR1, usr1)
    signal.signal(signal.SIGUSR2, usr2)


def _receive_body_then_wait():
    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": BODY, "more_body": False}
        await asyncio.Event().wait()

    return receive


def _host_proxy(log: Path):
    from benchflow.providers.litellm_runtime import HostLiteLLMProcess

    proxy = object.__new__(HostLiteLLMProcess)
    proxy.log_path = str(log)
    proxy.process = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    return proxy


async def test_the_proxy_counts_its_open_calls_not_yet_answered(gate_log):
    """The guard's count (SIGUSR1), read by the host proxy's controller: a
    call waiting on its provider counts; once its answer is whole it does
    not; an ended call does not."""
    answer, answered = asyncio.Event(), asyncio.Event()

    async def litellm(scope, receive, send):
        assert (await receive())["type"] == "http.request"
        await answered.wait()
        guard._REQUEST.get()["answered"] = True  # the provider's answer is in
        await answer.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    route = guard.SingleRouteGuard(litellm)
    proxy = _host_proxy(gate_log)
    call = asyncio.create_task(route(_scope(), _receive_body_then_wait(), _sink([])))
    await asyncio.sleep(0.05)
    assert await proxy.unanswered_calls() == 1
    answered.set()
    await asyncio.sleep(0.05)
    assert await proxy.unanswered_calls() == 0
    answer.set()
    await asyncio.wait_for(call, 5)
    assert await proxy.unanswered_calls() == 0


class _Proxy:
    """A trial's proxy, standing in: ``open`` calls not yet answered, each
    answered ``answer_after`` seconds after the stop began."""

    def __init__(self, order: list, answer_after: float) -> None:
        self.order = order
        self.answer_at: float | None = None
        self.answer_after = answer_after

    async def unanswered_calls(self) -> int | None:
        loop = asyncio.get_running_loop()
        if self.answer_at is None:
            self.answer_at = loop.time() + self.answer_after
            self.order.append("asked")
        return 1 if loop.time() < self.answer_at else 0


class _Env:
    async def exec(self, command, user=None, timeout_sec=None, **_):
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def _connected(tmp_path, order: list, proxy: _Proxy):
    from benchflow.rollout import Rollout, RolloutConfig

    async def close():
        loop = asyncio.get_running_loop()
        answered = proxy.answer_at is not None and loop.time() >= proxy.answer_at
        order.append("close after the answer" if answered else "close")

    rollout = Rollout(RolloutConfig(task_path=Path("task")))
    rollout._env = _Env()
    rollout._acp_client = SimpleNamespace(close=close)
    rollout._usage_runtime = SimpleNamespace(server=proxy)
    rollout._rollout_dir = tmp_path / "rollout"
    rollout._phase = "connected"
    return rollout


async def test_the_normal_stop_waits_for_an_unanswered_call(tmp_path):
    """The session title's answer comes 0.5 s into the stop: the agent is
    stopped after it, not before."""
    order: list = []
    await _connected(tmp_path, order, _Proxy(order, 0.5)).disconnect()
    assert order == ["asked", "close after the answer"]


async def test_a_deadline_stop_does_not_wait(tmp_path):
    order: list = []
    rollout = _connected(tmp_path, order, _Proxy(order, 5.0))
    rollout._agent_deadline_reason = "Agent timed out"
    started = asyncio.get_running_loop().time()
    await rollout.disconnect()
    assert order == ["close"]
    assert asyncio.get_running_loop().time() - started < 2


async def test_the_wait_is_bounded(tmp_path, monkeypatch):
    """A provider that never answers: the stop goes on at the bound."""
    from benchflow import rollout as rollout_module

    monkeypatch.setattr(rollout_module, "_STOP_WAIT_SEC", 0.3)
    order: list = []
    await _connected(tmp_path, order, _Proxy(order, 60.0)).disconnect()
    assert order == ["asked", "close"]


def test_the_bound_follows_the_load(monkeypatch):
    from benchflow import rollout as rollout_module

    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    monkeypatch.setattr(os, "getloadavg", lambda: (8.0, 0, 0))
    assert rollout_module._stop_wait_sec() == rollout_module._STOP_WAIT_SEC
    monkeypatch.setattr(os, "getloadavg", lambda: (96.0, 0, 0))
    assert rollout_module._stop_wait_sec() == 3 * rollout_module._STOP_WAIT_SEC
    monkeypatch.setattr(os, "getloadavg", lambda: (640.0, 0, 0))
    assert rollout_module._stop_wait_sec() == 4 * rollout_module._STOP_WAIT_SEC
