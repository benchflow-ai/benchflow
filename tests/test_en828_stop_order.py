"""D1/D2: the stop closes the model gate first, and on time.

R2: ``cancel_prompt_after_timeout`` awaits ``before_cancel`` (the gate) before
it asks the agent to stop, bounded, whatever the gate does. R3: with the idle
watchdog on, the wall clock fires at the deadline, not at the next poll.
"""

from __future__ import annotations

import asyncio

import pytest

from benchflow.acp import runtime as acp_runtime
from benchflow.acp.session import ACPSession
from benchflow.acp.timeout_cleanup import cancel_prompt_after_timeout
from benchflow.acp.watchdog import IdleWatchdog
from benchflow.diagnostics import AgentPromptTimeoutError


class _Hanging:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.cancelled_at: list[float] = []

    async def prompt(self, text):
        await asyncio.sleep(3600)

    async def cancel(self):
        self.order.append("cancel")
        self.cancelled_at.append(asyncio.get_running_loop().time())


@pytest.mark.asyncio
async def test_the_gate_closes_before_the_agent_is_asked_to_stop():
    order: list[str] = []

    async def gate():
        await asyncio.sleep(0.3)
        order.append("gate")

    prompt = asyncio.create_task(asyncio.sleep(3600))
    await cancel_prompt_after_timeout(_Hanging(order), prompt, before_cancel=gate)
    assert order[:2] == ["gate", "cancel"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["hangs", "raises"])
async def test_a_gate_that_hangs_or_fails_still_stops_the_agent_in_bounds(
    kind, monkeypatch
):
    from benchflow.acp import timeout_cleanup

    monkeypatch.setattr(timeout_cleanup, "GATE_CLOSE_TIMEOUT_SEC", 1.5)
    order: list[str] = []

    async def gate():
        if kind == "raises":
            raise RuntimeError("proxy gone")
        await asyncio.sleep(3600)

    loop = asyncio.get_running_loop()
    started = loop.time()
    prompt = asyncio.create_task(asyncio.sleep(3600))
    await cancel_prompt_after_timeout(_Hanging(order), prompt, before_cancel=gate)
    assert "cancel" in order
    assert loop.time() - started < 7.5


@pytest.mark.asyncio
@pytest.mark.parametrize("idle", [None, 600])
async def test_execute_prompts_closes_the_gate_first_at_the_deadline(idle):
    order: list[str] = []

    async def gate():
        order.append("gate")

    with pytest.raises((AgentPromptTimeoutError, TimeoutError)):
        await acp_runtime.execute_prompts(
            _Hanging(order),
            ACPSession("s"),
            ["p"],
            timeout=1,
            idle_timeout=idle,
            on_deadline=gate,
        )
    assert order[:2] == ["gate", "cancel"]


@pytest.mark.asyncio
async def test_the_wall_clock_fires_at_the_deadline_not_the_next_poll(monkeypatch):
    """45 s against an 11 s poll stopped the agent at 55 s (rc5's
    timeout-pass trial: cancel 10.6 s late). With a 30 s poll and a 2 s
    budget the cancel must still come at 2 s."""
    monkeypatch.setattr(IdleWatchdog, "poll_interval_sec", property(lambda self: 30))
    order: list[str] = []
    client = _Hanging(order)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises((AgentPromptTimeoutError, TimeoutError)):
        await asyncio.wait_for(
            acp_runtime._prompt_with_idle_watchdog(
                client, ACPSession("s"), "p", 2, 600
            ),
            6,
        )
    assert client.cancelled_at, "the wall clock never fired within 6 s"
    assert client.cancelled_at[0] - started < 2.5


class _Planes:
    def __init__(self) -> None:
        self.quiesced: list[str] = []

    async def quiesce_agent(self, env, sandbox_user, *, keep_ports=()):
        self.quiesced.append(sandbox_user)


class _Server:
    def __init__(self, state: str) -> None:
        self.state = state

    async def close_gate(self, reason="the agent's stop"):
        return {"state": self.state}


def _stand_in(runtimes, user="agent"):
    from types import SimpleNamespace

    planes = _Planes()
    stand_in = SimpleNamespace(
        _usage_runtimes=lambda: runtimes,
        _config=SimpleNamespace(sandbox_user=user),
        _env=object(),
        _planes=planes,
    )
    return stand_in, planes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("states", "killed"),
    [
        (["closed"], False),
        (["closed", "dead"], False),
        ([], True),  # a login or direct route: no proxy, no gate
        (["unarmed"], True),
        (["closed", "unacknowledged"], True),
    ],
)
async def test_without_a_closed_gate_the_agents_processes_are_killed(states, killed):
    from types import SimpleNamespace

    from benchflow.rollout import Rollout

    runtimes = [SimpleNamespace(server=_Server(state)) for state in states]
    stand_in, planes = _stand_in(runtimes)
    await Rollout._close_model_gates(stand_in)
    assert planes.quiesced == (["agent"] if killed else [])
