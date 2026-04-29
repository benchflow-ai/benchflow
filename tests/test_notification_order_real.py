"""Verify that ACP notifications arrive interleaved (not batched).

This test hooks into ACPSession.handle_update to record the raw
notification order, then runs a real prompt through a mock agent.
It confirms that agent_thought_chunk and agent_message_chunk arrive
BETWEEN tool_call notifications, not all-at-the-end.

This is the critical assumption behind the _flush_agent_text fix in PR #214:
if notifications were batched, flushing at tool_call boundaries wouldn't
split the text.
"""

import sys
from pathlib import Path

import pytest

from benchflow._acp_run import execute_prompts
from benchflow.acp.client import ACPClient
from benchflow.acp.transport import StdioTransport

MOCK_AGENT_MULTI = str(
    Path(__file__).parent / "fixtures" / "mock_acp_agent_multi_turn.py"
)


class TestNotificationOrderReal:
    """Verify notification interleaving with a real ACP transport."""

    @pytest.mark.asyncio
    async def test_notifications_arrive_interleaved(self) -> None:
        """Record raw handle_update calls and verify they interleave."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_MULTI]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()

            # Monkey-patch to record raw notification order
            raw_updates: list[str] = []
            original_handle = session.handle_update

            def logging_handle(update: dict) -> None:
                raw_updates.append(update.get("sessionUpdate", "unknown"))
                original_handle(update)

            session.handle_update = logging_handle

            await execute_prompts(client, session, ["Prompt 1", "Prompt 2"], timeout=10)

            # Verify we got notifications
            assert len(raw_updates) > 0

            # Verify notifications are INTERLEAVED: text chunks appear between tool_calls
            # For multi-turn agent: each prompt sends thought_chunk, tool_call,
            # tool_call_update, then message_chunk.
            # So we expect: [thought, tool_call, tool_call_update, message, thought, ...]
            tc_indices = [i for i, u in enumerate(raw_updates) if u == "tool_call"]
            msg_indices = [
                i for i, u in enumerate(raw_updates) if u == "agent_message_chunk"
            ]
            thought_indices = [
                i for i, u in enumerate(raw_updates) if u == "agent_thought_chunk"
            ]

            assert len(tc_indices) == 2, f"Expected 2 tool_calls, got {tc_indices}"
            assert len(msg_indices) == 2, f"Expected 2 messages, got {msg_indices}"
            assert len(thought_indices) == 2, (
                f"Expected 2 thoughts, got {thought_indices}"
            )

            # Thought for turn 1 must come BEFORE tool_call for turn 1
            assert thought_indices[0] < tc_indices[0], (
                f"thought[0]={thought_indices[0]} should be before tc[0]={tc_indices[0]}"
            )
            # Message for turn 1 must come AFTER tool_call for turn 1
            assert msg_indices[0] > tc_indices[0], (
                f"msg[0]={msg_indices[0]} should be after tc[0]={tc_indices[0]}"
            )
            # Message for turn 1 must come BEFORE tool_call for turn 2
            assert msg_indices[0] < tc_indices[1], (
                f"msg[0]={msg_indices[0]} should be before tc[1]={tc_indices[1]}"
            )

            print(f"\nNotification order: {raw_updates}")

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_flush_produces_per_turn_events(self) -> None:
        """With interleaved notifications, flush splits text per-turn."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_MULTI]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()

            trajectory, _ = await execute_prompts(
                client, session, ["P1", "P2"], timeout=10
            )

            # Extract agent_message events
            messages = [e for e in trajectory if e["type"] == "agent_message"]
            thoughts = [e for e in trajectory if e["type"] == "agent_thought"]

            # CRITICAL: each turn should have its own message/thought,
            # NOT one cumulative blob
            assert len(messages) == 2, (
                f"Expected 2 separate agent_message events, got {len(messages)}. "
                f"Events: {[e['type'] for e in trajectory]}"
            )
            assert len(thoughts) == 2, (
                f"Expected 2 separate agent_thought events, got {len(thoughts)}. "
                f"Events: {[e['type'] for e in trajectory]}"
            )

            # Verify non-cumulative content
            assert "turn-1" in messages[0]["text"]
            assert "turn-2" in messages[1]["text"]
            assert "turn-1" not in messages[1]["text"], (
                "Second message should NOT contain first turn's text"
            )

        finally:
            await client.close()
