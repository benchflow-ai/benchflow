"""Regressions for buffered ACP starvation and delayed timeout observation."""

import asyncio
import json
import time

import pytest

from benchflow.acp.client import ACPClient
from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.runtime import (
    _prompt_with_idle_watchdog,
    _prompt_with_wall_clock_budget,
)
from benchflow.acp.session import ACPSession
from benchflow.acp.types import PromptResult
from benchflow.diagnostics import AgentPromptTimeoutError
from benchflow.sandbox.process.daytona import DaytonaPtyProcess


class BufferedPty(DaytonaPtyProcess):
    """Use the actual queue reader without creating a sandbox or connection."""

    def __init__(self):
        super().__init__(None, "", "")

    async def writeline(self, data):
        pass

    def enqueue(self, message):
        self._line_buffer.put_nowait(json.dumps(message).encode() + b"\n")


@pytest.mark.asyncio
async def test_buffered_daytona_notifications_allow_other_tasks_to_run():
    """Guards commit 4196c4a: a populated PTY queue must not starve the loop."""
    process = BufferedPty()
    session = ACPSession("buffered")
    for _ in range(32):
        process.enqueue(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": session.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "."},
                    },
                },
            }
        )
    process.enqueue(
        {"jsonrpc": "2.0", "id": 100001, "result": {"stopReason": "end_turn"}}
    )
    client = ACPClient(ContainerTransport(process, command="never-started"))
    client._session = session

    async def observe_backlog():
        await asyncio.sleep(0)
        return len(session.message_chunks)

    result, seen = await asyncio.gather(client.prompt("go"), observe_backlog())

    assert 0 < seen < 32
    assert result.stop_reason == "end_turn"
    assert session.full_message == "." * 32


class BlockingClient:
    def __init__(self, block_sec):
        self.block_sec = block_sec

    async def prompt(self, prompt):
        # Models synchronous capture during an otherwise inline buffered read.
        time.sleep(self.block_sec)
        return PromptResult(stop_reason="end_turn")


async def run_prompt(client, session, idle_watchdog):
    if idle_watchdog:
        return await _prompt_with_idle_watchdog(
            client, session, "go", timeout=1, idle_timeout=600
        )
    return await _prompt_with_wall_clock_budget(client, session, "go", timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_watchdog", [False, True])
async def test_completion_after_wall_deadline_is_timeout(idle_watchdog):
    """Guards commit 4196c4a: late completion cannot escape a starved watchdog."""
    session = ACPSession("late-completion")
    session.record_user_prompt("go")

    with pytest.raises(AgentPromptTimeoutError, match="wall-clock budget 1s"):
        await run_prompt(BlockingClient(1.1), session, idle_watchdog)

    timeouts = [event for event in session.events if event["type"] == "agent_timeout"]
    assert len(timeouts) == 1
    assert timeouts[0]["timeout_sec"] == 1
    assert timeouts[0]["terminal_trajectory_complete"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_watchdog", [False, True])
async def test_early_completion_observed_after_deadline_still_succeeds(idle_watchdog):
    """Guards commit 4196c4a: observer delay is not prompt execution time."""
    session = ACPSession("late-observation")
    session.record_user_prompt("go")

    async def block_after_prompt_completes():
        await asyncio.sleep(0)
        time.sleep(1.1)

    blocker = asyncio.create_task(block_after_prompt_completes())
    try:
        result = await run_prompt(BlockingClient(0), session, idle_watchdog)
    finally:
        await blocker

    assert result.stop_reason == "end_turn"
    assert not any(event["type"] == "agent_timeout" for event in session.events)
