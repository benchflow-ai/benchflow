"""A lost terminal update no longer kills an active session (#1141).

IdleWatchdog gave each pending tool call its own grace clock (3 x the idle
budget) that only the call's own updates restarted. One pending call whose
terminal update was lost therefore killed a session that kept working: on the
VM, idle 7200 s, a tool call completing every 30 s, idle_expired fired at
21 600 s. A newer tool call now restarts the grace of every call already
pending, the grace is configurable, and the timeout names a lost update.
These tests drive a real ACPSession with ACP updates on a simulated clock,
and two short runs go through execute_prompts.
"""

import asyncio
import logging

import pytest

from benchflow.acp.session import ACPSession
from benchflow.acp.types import PromptResult
from benchflow.acp.watchdog import (
    PENDING_GRACE_ENV,
    IdleWatchdog,
    pending_grace_from_env,
)
from benchflow.diagnostics import IdleTimeoutDiagnostic

IDLE = 7200
GRACE = 3 * IDLE


def _start(session, call_id, title="Preparing file…"):
    session.handle_update(
        {
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": title,
            "kind": "edit",
            "status": "pending",
        }
    )


def _update(session, call_id, status):
    session.handle_update(
        {"sessionUpdate": "tool_call_update", "toolCallId": call_id, "status": status}
    )


def _watchdog(session, **kwargs):
    return IdleWatchdog.start(
        session, idle_timeout_sec=IDLE, wall_timeout_sec=10 * GRACE, now=0.0, **kwargs
    )


def _work_every_30s(session, watchdog, until, start=0.0):
    """A busy agent: a new call starts and completes every 30 s (one poll)."""
    t = start
    while t < until:
        t += 30.0
        call_id = f"done-{int(t)}"
        _start(session, call_id, title="Bash")
        _update(session, call_id, "completed")
        watchdog.observe(session, now=t)
        assert not watchdog.idle_expired(t), (t, watchdog.expired_pending_ids(t))
    return t


def test_placeholder_that_never_updates_does_not_kill_a_busy_session():
    """The #1141 timeline: the old watchdog fired at 21 600 s."""
    session = ACPSession("s")
    _start(session, "toolu_lost")  # its completion never arrives
    watchdog = _watchdog(session)
    last = _work_every_30s(session, watchdog, until=GRACE + 3 * 3600)
    assert watchdog.pending_ids == ("toolu_lost",)
    assert watchdog.lost_update_ids == ("toolu_lost",)

    # The agent then goes silent: the grace runs from the last new call.
    watchdog.observe(session, now=last + GRACE - 30)
    assert not watchdog.idle_expired(last + GRACE - 30)
    watchdog.observe(session, now=last + GRACE)
    assert watchdog.idle_expired(last + GRACE)
    error = watchdog.timeout_error(session, now=last + GRACE)
    assert "lost update" in str(error)
    assert "toolu_lost" in str(error)
    info = error.diagnostic.to_dict()
    assert info["lost_update_tool_call_ids"] == ["toolu_lost"]
    assert info["expired_pending_tool_call_ids"] == ["toolu_lost"]
    assert info["pending_grace_sec"] == GRACE


def test_call_that_streamed_then_lost_its_completion_is_covered_too():
    """codex-acp streams in_progress output before its terminal update."""
    session = ACPSession("s")
    _start(session, "exec-1", title="Run tests")
    _update(session, "exec-1", "in_progress")
    watchdog = _watchdog(session)
    _update(session, "exec-1", "in_progress")
    watchdog.observe(session, now=30.0)
    _work_every_30s(session, watchdog, until=GRACE + 3600, start=30.0)
    assert watchdog.lost_update_ids == ("exec-1",)


def test_a_silent_call_with_nothing_after_it_keeps_the_grace_cap():
    """PR #1066's bound still holds when no newer call starts."""
    session = ACPSession("s")
    _start(session, "long-build", title="Bash")
    watchdog = _watchdog(session)
    for t in range(30, GRACE, 30):
        watchdog.observe(session, now=float(t))
        assert not watchdog.idle_expired(float(t))
    watchdog.observe(session, now=float(GRACE))
    assert watchdog.idle_expired(float(GRACE))
    error = watchdog.timeout_error(session, now=float(GRACE))
    assert "lost update" not in str(error)
    assert error.diagnostic.to_dict()["lost_update_tool_call_ids"] == []


def test_parallel_siblings_started_together_are_not_called_lost():
    session = ACPSession("s")
    watchdog = _watchdog(session)
    _start(session, "task-a", title="Task")
    _start(session, "task-b", title="Task")
    watchdog.observe(session, now=30.0)
    _update(session, "task-b", "completed")
    watchdog.observe(session, now=60.0)
    assert watchdog.pending_ids == ("task-a",)
    assert watchdog.lost_update_ids == ()


def test_an_update_after_newer_calls_clears_the_lost_evidence():
    session = ACPSession("s")
    _start(session, "slow")
    watchdog = _watchdog(session)
    _work_every_30s(session, watchdog, until=90.0)
    assert watchdog.lost_update_ids == ("slow",)
    _update(session, "slow", "in_progress")
    watchdog.observe(session, now=120.0)
    assert watchdog.lost_update_ids == ()


@pytest.mark.parametrize(
    "raw,expected", [("", None), ("900", 900), ("0", None), ("-5", None), ("x", None)]
)
def test_pending_grace_env(monkeypatch, caplog, raw, expected):
    monkeypatch.setenv(PENDING_GRACE_ENV, raw)
    with caplog.at_level(logging.WARNING):
        assert pending_grace_from_env() == expected
    assert (PENDING_GRACE_ENV in caplog.text) == (raw not in ("", "900"))


def test_configured_grace_replaces_the_multiplier(monkeypatch):
    session = ACPSession("s")
    _start(session, "long-build", title="Bash")
    monkeypatch.setenv(PENDING_GRACE_ENV, "900")
    assert _watchdog(session).pending_grace_sec == 900
    assert _watchdog(session, pending_grace_sec=60).pending_grace_sec == 60
    monkeypatch.delenv(PENDING_GRACE_ENV)
    watchdog = _watchdog(session)
    assert watchdog.pending_grace_sec == GRACE
    watchdog = _watchdog(session, pending_grace_sec=900)
    watchdog.observe(session, now=899.0)
    assert not watchdog.idle_expired(899.0)
    watchdog.observe(session, now=900.0)
    assert watchdog.idle_expired(900.0)


def test_issue_line_says_lost_update():
    line = IdleTimeoutDiagnostic(
        idle_duration_sec=21600,
        pending_tool_call_ids=["toolu_lost"],
        expired_pending_tool_call_ids=["toolu_lost"],
        pending_grace_sec=21600,
        lost_update_tool_call_ids=["toolu_lost"],
    ).format_issue("task")
    assert "lost update" in line


class _BusyAfterLostPlaceholder:
    """Opens a placeholder that never completes, then keeps completing calls."""

    def __init__(self, session: ACPSession, busy_sec: float, then_hang: bool):
        self._session = session
        self._busy_sec = busy_sec
        self._then_hang = then_hang

    async def prompt(self, _prompt: str):
        _start(self._session, "toolu_lost")
        loop = asyncio.get_running_loop()
        end = loop.time() + self._busy_sec
        n = 0
        while loop.time() < end:
            await asyncio.sleep(0.3)
            n += 1
            _start(self._session, f"done-{n}", title="Bash")
            _update(self._session, f"done-{n}", "completed")
        if self._then_hang:
            await asyncio.Future()
        return PromptResult(stop_reason="end_turn")


@pytest.mark.asyncio
async def test_execute_prompts_survives_a_lost_placeholder_past_the_old_grace():
    """Idle 1 s, so the old per-call grace (3 s) fired mid-work."""
    from benchflow.acp.runtime import execute_prompts

    session = ACPSession("busy")
    _, n_tool_calls = await asyncio.wait_for(
        execute_prompts(
            _BusyAfterLostPlaceholder(session, busy_sec=5.0, then_hang=False),  # type: ignore[arg-type]
            session,
            ["solve"],
            timeout=60,
            idle_timeout=1,
        ),
        timeout=30.0,
    )
    assert n_tool_calls > 10
    assert session.pending_tool_call_ids() == ["toolu_lost"]


@pytest.mark.asyncio
async def test_execute_prompts_names_the_lost_update_when_the_agent_hangs():
    from benchflow.acp.runtime import IdleTimeoutError, execute_prompts

    session = ACPSession("hung")
    with pytest.raises(IdleTimeoutError) as raised:
        await asyncio.wait_for(
            execute_prompts(
                # Busy across at least two 1 s polls, so one of them sees the
                # placeholder first and a later one sees newer calls start.
                _BusyAfterLostPlaceholder(session, busy_sec=2.5, then_hang=True),  # type: ignore[arg-type]
                session,
                ["solve"],
                timeout=60,
                idle_timeout=1,
                pending_tool_grace=2,
            ),
            timeout=30.0,
        )
    assert "lost update" in str(raised.value)
    info = raised.value.diagnostic.to_dict()
    assert info["lost_update_tool_call_ids"] == ["toolu_lost"]
    assert info["pending_grace_sec"] == 2
