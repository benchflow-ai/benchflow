"""The live-process readline guard never undercuts the prompt's own silence budget.

Guards #1143: the Daytona PTY (and AgentCore shell) readline timeout was a
fixed 900 s unless ``BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT`` was exported, so
a stage allowed to stay silent longer (``--agent-idle-timeout 7200``, or a
reviewer with a raised idle budget) was killed at 900 s as
``PTY readline timeout (900s)`` before its own idle watchdog could decide.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.runtime import execute_prompts
from benchflow.acp.session import ACPSession
from benchflow.sandbox.process import DaytonaPtyProcess
from benchflow.sandbox.process._base import LiveProcess

_ENV = "BENCHFLOW_DAYTONA_PTY_READLINE_TIMEOUT"


def _pty() -> DaytonaPtyProcess:
    return DaytonaPtyProcess(
        sandbox=MagicMock(),
        compose_cmd_prefix="",
        compose_cmd_base="docker compose -p test",
    )


async def _readline_timeout(proc: LiveProcess) -> float:
    async def fake_wait_for(awaitable, *, timeout):
        awaitable.close()
        raise TimeoutError

    with (
        patch("asyncio.wait_for", side_effect=fake_wait_for) as wait_for,
        pytest.raises(Exception, match="readline timeout"),
    ):
        await proc.readline()
    return wait_for.call_args.kwargs["timeout"]


@pytest.mark.asyncio
async def test_pty_readline_waits_at_least_the_expected_silence(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    proc = _pty()
    proc.expect_silence(7200)
    assert await _readline_timeout(proc) == 7200


@pytest.mark.asyncio
async def test_a_short_silence_budget_keeps_the_default(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    proc = _pty()
    proc.expect_silence(300)
    assert await _readline_timeout(proc) == 900.0


@pytest.mark.asyncio
async def test_an_explicit_env_value_still_wins(monkeypatch):
    monkeypatch.setenv(_ENV, "120")
    proc = _pty()
    proc.expect_silence(7200)
    assert await _readline_timeout(proc) == 120.0


@pytest.mark.asyncio
async def test_agentcore_shell_readline_honours_the_silence_budget(monkeypatch):
    from benchflow.sandbox.process.agentcore import AgentCoreProcess

    monkeypatch.delenv("BENCHFLOW_AGENTCORE_READLINE_TIMEOUT", raising=False)
    proc = AgentCoreProcess.__new__(AgentCoreProcess)
    proc._closed = False
    proc._line_buffer = asyncio.Queue()
    proc.expect_silence(5000)
    assert await _readline_timeout(proc) == 5000


class _RecordingProcess(LiveProcess):
    def __init__(self) -> None:
        self.silence: list[float] = []

    def expect_silence(self, seconds: float) -> None:
        self.silence.append(seconds)

    async def start(self, command, env=None, cwd=None) -> None:
        return None

    async def readline(self) -> bytes:
        await asyncio.Future()
        return b""

    async def writeline(self, data: str) -> None:
        return None

    async def close(self) -> None:
        return None

    @property
    def is_running(self) -> bool:
        return True


class _InstantClient(ACPClient):
    async def prompt(self, _text):  # type: ignore[override]
        return SimpleNamespace(stop_reason="end_turn")


@pytest.mark.parametrize(
    ("idle_timeout", "timeout", "expected_floor"),
    [(7200, 86400, 7200), (None, 3600, 3600)],
)
@pytest.mark.asyncio
async def test_execute_prompts_hands_the_silence_budget_to_the_transport(
    idle_timeout, timeout, expected_floor
):
    """The idle watchdog (or, without one, the wall budget) owns silence."""
    proc = _RecordingProcess()
    client = _InstantClient(ContainerTransport(proc, command="agent"))
    session = ACPSession("silence-budget")
    await execute_prompts(
        client, session, ["solve"], timeout=timeout, idle_timeout=idle_timeout
    )
    assert proc.silence
    assert min(proc.silence) >= expected_floor


@pytest.mark.parametrize(
    ("idle_timeout", "timeout", "guard"),
    [(600, 1800, 3 * 600 + 60), (7200, 86400, 3 * 7200 + 60), (None, 3600, 3660)],
)
@pytest.mark.asyncio
async def test_the_read_guard_covers_the_pending_tool_grace(
    idle_timeout, timeout, guard
):
    """#1143: the idle watchdog lets a pending tool call stay silent for three
    idle budgets, so the transport must not cut it at idle + 60 s (a reviewer
    with the default 600 s idle budget was cut at 900 s)."""
    from benchflow.acp.runtime import transport_silence_budget
    from benchflow.acp.watchdog import PENDING_GRACE_MULTIPLIER

    assert PENDING_GRACE_MULTIPLIER == 3
    assert transport_silence_budget(timeout, idle_timeout) == guard
    proc = _RecordingProcess()
    client = _InstantClient(ContainerTransport(proc, command="agent"))
    await execute_prompts(
        client,
        ACPSession("grace"),
        ["solve"],
        timeout=timeout,
        idle_timeout=idle_timeout,
    )
    assert proc.silence == [guard]
