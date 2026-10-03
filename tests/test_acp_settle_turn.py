"""The settle rule: a turn the agent finished but never closed ends as end_turn.

Guards PR #1150 (BENCHFLOW_ACP_SETTLE_TIMEOUT) against the OpenCode-on-Daytona stall: the agent sends its final message, and
the ``session/prompt`` answer never arrives (OpenCode's own log shows its loop
exiting in the stalled attempts examined), so the attempt waited out the wall
clock and was recorded as a timeout.
"""

import asyncio
import json

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.runtime import (
    _prompt_with_idle_watchdog,
    _prompt_with_wall_clock_budget,
)
from benchflow.acp.session import ACPSession
from benchflow.acp.types import PromptResult, StopReason
from benchflow.diagnostics import AgentPromptTimeoutError
from benchflow.sandbox.process.daytona import DaytonaPtyProcess
from benchflow.trajectories._capture import _capture_session_trajectory

SETTLE_ENV = "BENCHFLOW_ACP_SETTLE_TIMEOUT"

MESSAGE = {
    "sessionUpdate": "agent_message_chunk",
    "content": {"type": "text", "text": "Done: the report is in /app/out.csv."},
}
THOUGHT = {
    "sessionUpdate": "agent_thought_chunk",
    "content": {"type": "text", "text": "Now I should check the totals."},
}
PENDING_TOOL = {
    "sessionUpdate": "tool_call",
    "toolCallId": "call-1",
    "title": "bash",
    "kind": "execute",
    "status": "pending",
}
FINISHED_TOOL = {
    "sessionUpdate": "tool_call_update",
    "toolCallId": "call-1",
    "status": "completed",
}


class SilentAfterUpdates:
    """An agent that streams some updates, then answers session/prompt only when cancelled."""

    def __init__(self, session, updates):
        self.session = session
        self.updates = updates
        self.cancels = 0
        self._cancelled = asyncio.Event()

    async def prompt(self, prompt):
        for update in self.updates:
            self.session.handle_update(update)
        await self._cancelled.wait()
        return PromptResult(stop_reason="cancelled")

    async def cancel(self):
        self.cancels += 1
        self._cancelled.set()


async def run_prompt(client, session, idle_watchdog, timeout):
    if idle_watchdog:
        return await _prompt_with_idle_watchdog(
            client, session, "go", timeout=timeout, idle_timeout=600
        )
    return await _prompt_with_wall_clock_budget(client, session, "go", timeout=timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_watchdog", [False, True])
async def test_quiet_after_the_final_message_ends_the_turn(monkeypatch, idle_watchdog):
    monkeypatch.setenv(SETTLE_ENV, "0.2")
    session = ACPSession("settled")
    session.record_user_prompt("go")
    client = SilentAfterUpdates(session, [PENDING_TOOL, FINISHED_TOOL, MESSAGE])
    loop = asyncio.get_running_loop()
    started = loop.time()

    result = await run_prompt(client, session, idle_watchdog, timeout=30)

    assert result.stop_reason == "end_turn"
    assert loop.time() - started < 5
    assert session.stop_reason == StopReason.END_TURN
    assert client.cancels == 1
    inferred = [e for e in session.events if e["type"] == "agent_turn_end_inferred"]
    assert len(inferred) == 1
    assert inferred[0]["reason"] == "quiet_after_final_message"
    assert inferred[0]["settle_timeout_sec"] == 0.2
    assert inferred[0]["quiet_sec"] >= 0.2
    trajectory = _capture_session_trajectory(session)
    assert trajectory[-2] == {
        "type": "agent_message",
        "text": MESSAGE["content"]["text"],
    }
    assert trajectory[-1]["type"] == "agent_turn_end_inferred"


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_watchdog", [False, True])
@pytest.mark.parametrize(
    "updates",
    [[MESSAGE, PENDING_TOOL], [MESSAGE, THOUGHT]],
    ids=["tool-call-pending", "thought-after-message"],
)
async def test_an_agent_still_working_is_not_cut_off(
    monkeypatch, idle_watchdog, updates
):
    monkeypatch.setenv(SETTLE_ENV, "0.2")
    session = ACPSession("working")
    session.record_user_prompt("go")

    with pytest.raises(AgentPromptTimeoutError, match="wall-clock budget 1s"):
        await run_prompt(
            SilentAfterUpdates(session, updates), session, idle_watchdog, timeout=1
        )

    assert not [e for e in session.events if e["type"] == "agent_turn_end_inferred"]


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_watchdog", [False, True])
async def test_the_settle_rule_is_off_by_default(monkeypatch, idle_watchdog):
    monkeypatch.delenv(SETTLE_ENV, raising=False)
    session = ACPSession("default")
    session.record_user_prompt("go")

    with pytest.raises(AgentPromptTimeoutError, match="wall-clock budget 1s"):
        await run_prompt(
            SilentAfterUpdates(session, [MESSAGE]), session, idle_watchdog, timeout=1
        )

    assert session.events[-1]["type"] == "agent_timeout"


def test_quiet_time_counts_only_after_a_final_message():
    session = ACPSession("quiet")
    session.record_user_prompt("go")
    assert session.quiet_after_final_message_sec() is None
    session.handle_update(MESSAGE)
    assert session.quiet_after_final_message_sec() >= 0
    session.handle_update(PENDING_TOOL)
    assert session.quiet_after_final_message_sec() is None
    session.handle_update(FINISHED_TOOL)
    assert session.quiet_after_final_message_sec() is None
    session.handle_update(THOUGHT)
    assert session.quiet_after_final_message_sec() is None
    session.handle_update(MESSAGE)
    quiet = session.quiet_after_final_message_sec(now=session._last_update_at + 7)
    assert quiet == 7
    # An unrecognized update type is not activity and leaves the final message last.
    session.handle_update({"sessionUpdate": "plan", "entries": []})
    assert session.quiet_after_final_message_sec() is not None


class LostAnswerPty(DaytonaPtyProcess):
    """A Daytona PTY that delivers what is queued and never the session/prompt answer."""

    def __init__(self):
        super().__init__(None, "", "")
        self.sent = []

    async def writeline(self, data):
        self.sent.append(json.loads(data))

    def enqueue(self, message):
        self._line_buffer.put_nowait(json.dumps(message).encode() + b"\n")


@pytest.mark.asyncio
async def test_the_real_client_ends_a_turn_whose_answer_never_arrives(monkeypatch):
    monkeypatch.setenv(SETTLE_ENV, "0.2")
    session = ACPSession("lost-answer")
    process = LostAnswerPty()
    process.enqueue(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session.session_id, "update": MESSAGE},
        }
    )
    client = ACPClient(ContainerTransport(process, command="never-started"))
    client._session = session
    session.record_user_prompt("go")

    result = await _prompt_with_wall_clock_budget(client, session, "go", timeout=30)

    assert result.stop_reason == "end_turn"
    assert session.full_message == MESSAGE["content"]["text"]
    assert [m.get("method") for m in process.sent] == [
        "session/prompt",
        "session/cancel",
    ]
