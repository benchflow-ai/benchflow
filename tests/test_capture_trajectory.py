"""Tests for _capture_session_trajectory — ensures partial trajectory is saved on timeout."""

import json
from pathlib import Path

from benchflow.acp.session import ACPSession
from benchflow.acp.types import ToolCallStatus
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _snapshot_session_trajectory,
    make_trajectory_sink,
)


class TestCaptureSessionTrajectory:
    def test_none_session_returns_empty(self) -> None:
        assert _capture_session_trajectory(None) == []

    def test_empty_session_returns_empty(self) -> None:
        session = ACPSession("s1")
        assert _capture_session_trajectory(session) == []

    def test_captures_tool_calls(self) -> None:
        session = ACPSession("s1")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "echo hello",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        result = _capture_session_trajectory(session)
        assert len(result) == 1
        assert result[0]["type"] == "tool_call"
        assert result[0]["tool_call_id"] == "tc_1"
        assert result[0]["kind"] == "bash"
        assert result[0]["title"] == "echo hello"
        assert result[0]["status"] == ToolCallStatus.COMPLETED.value

    def test_captures_message(self) -> None:
        session = ACPSession("s1")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Hello world"},
            }
        )
        result = _capture_session_trajectory(session)
        assert len(result) == 1
        assert result[0]["type"] == "agent_message"
        assert result[0]["text"] == "Hello world"

    def test_captures_thought(self) -> None:
        session = ACPSession("s1")
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thinking..."},
            }
        )
        result = _capture_session_trajectory(session)
        assert len(result) == 1
        assert result[0]["type"] == "agent_thought"
        assert result[0]["text"] == "thinking..."

    def test_captures_partial_state(self) -> None:
        """Simulates timeout mid-execution: tool calls exist but are still in-progress."""
        session = ACPSession("s1")
        # First tool call completed
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls /app",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        # Second tool call still in progress (timeout happened here)
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_2",
                "title": "cat README.md",
                "kind": "bash",
            }
        )
        # Partial message streamed before timeout
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Let me read"},
            }
        )

        result = _capture_session_trajectory(session)
        assert len(result) == 3  # 2 tool calls + 1 message
        assert result[0]["status"] == ToolCallStatus.COMPLETED.value
        assert result[1]["status"] == ToolCallStatus.PENDING.value
        assert result[2]["type"] == "agent_message"
        assert result[2]["text"] == "Let me read"

    def test_captures_all_types_together(self) -> None:
        session = ACPSession("s1")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "echo hi",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "hmm"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "done"},
            }
        )
        result = _capture_session_trajectory(session)
        assert len(result) == 3
        types = [e["type"] for e in result]
        assert types == ["tool_call", "agent_thought", "agent_message"]


class TestUserMessageRecording:
    """Verify that user prompts appear in the trajectory (issue #745)."""

    def test_user_prompt_recorded(self) -> None:
        session = ACPSession("s1")
        session.record_user_prompt("Solve the task")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "cat /app/instruction.md",
                "kind": "read",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Done."},
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert types == ["user_message", "tool_call", "agent_message"]
        assert result[0]["text"] == "Solve the task"

    def test_multi_prompt_interleaving(self) -> None:
        """Two prompts should each get their own user_message and per-prompt agent text."""
        session = ACPSession("s1")

        # Prompt 1
        session.record_user_prompt("First prompt")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Answer 1"},
            }
        )
        session.mark_prompt_end()

        # Prompt 2
        session.record_user_prompt("Second prompt")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_2",
                "title": "cat file",
                "kind": "read",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_2",
                "status": "completed",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Answer 2"},
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert types == [
            "user_message",
            "tool_call",
            "agent_message",
            "user_message",
            "tool_call",
            "agent_message",
        ]
        # Messages are per-prompt, not cumulative
        assert result[0]["text"] == "First prompt"
        assert result[2]["text"] == "Answer 1"
        assert result[3]["text"] == "Second prompt"
        assert result[5]["text"] == "Answer 2"

    def test_message_not_cumulative_across_prompts(self) -> None:
        """agent_message text must not accumulate previous prompts' text (issue #745 comment)."""
        session = ACPSession("s1")

        session.record_user_prompt("P1")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "msg-from-p1"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thought-from-p1"},
            }
        )
        session.mark_prompt_end()

        session.record_user_prompt("P2")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "msg-from-p2"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thought-from-p2"},
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        messages = [e for e in result if e["type"] == "agent_message"]
        thoughts = [e for e in result if e["type"] == "agent_thought"]

        assert len(messages) == 2
        assert messages[0]["text"] == "msg-from-p1"
        assert messages[1]["text"] == "msg-from-p2"

        assert len(thoughts) == 2
        assert thoughts[0]["text"] == "thought-from-p1"
        assert thoughts[1]["text"] == "thought-from-p2"


class TestChronologicalEventOrder:
    """Verify events appear in the actual order they occurred (PR #214)."""

    def test_thought_before_tool_call(self) -> None:
        """Agent thinks, then calls a tool — thought should appear first."""
        session = ACPSession("s1")
        session.record_user_prompt("Go")
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "Let me think..."},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Here you go"},
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert types == [
            "user_message",
            "agent_thought",
            "tool_call",
            "agent_message",
        ]

    def test_multiple_tool_calls_with_interleaved_text(self) -> None:
        """Agent: think → tool1 → message → think → tool2 → message."""
        session = ACPSession("s1")
        session.record_user_prompt("Do the thing")

        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "Step 1"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Found files."},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "Step 2"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_2",
                "title": "cat foo",
                "kind": "read",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_2",
                "status": "completed",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Done reading."},
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert types == [
            "user_message",
            "agent_thought",
            "tool_call",
            "agent_message",
            "agent_thought",
            "tool_call",
            "agent_message",
        ]
        assert result[1]["text"] == "Step 1"
        assert result[4]["text"] == "Step 2"
        assert result[3]["text"] == "Found files."
        assert result[6]["text"] == "Done reading."

    def test_partial_timeout_preserves_order(self) -> None:
        """Timeout mid-prompt: whatever was streamed so far is captured in order."""
        session = ACPSession("s1")
        session.record_user_prompt("Solve it")
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thinking..."},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        # Timeout here — no mark_prompt_end, no tool_call_update

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert types == ["user_message", "agent_thought", "tool_call"]
        assert result[0]["text"] == "Solve it"
        assert result[2]["status"] == ToolCallStatus.PENDING.value

    def test_openclaw_text_update_and_agent_thought(self) -> None:
        """openclaw shim uses text_update and agent_thought (not chunks)."""
        session = ACPSession("s1")
        session.record_user_prompt("Go")
        session.handle_update(
            {"sessionUpdate": "agent_thought", "text": "full thought"}
        )
        session.handle_update({"sessionUpdate": "text_update", "text": "full message"})
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert "user_message" in types
        assert "agent_thought" in types
        assert "agent_message" in types

    def test_tool_call_update_without_prior_tool_call(self) -> None:
        """Agents that skip the initial tool_call notification (Gemini CLI)."""
        session = ACPSession("s1")
        session.record_user_prompt("Go")
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_auto",
                "title": "auto-created",
                "kind": "tool",
                "status": "completed",
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        tc_events = [e for e in result if e["type"] == "tool_call"]
        assert len(tc_events) == 1
        assert tc_events[0]["tool_call_id"] == "tc_auto"
        assert tc_events[0]["status"] == ToolCallStatus.COMPLETED.value


class TestLegacyFallback:
    """Sessions with no events log (direct tool_calls manipulation) still work (PR #214)."""

    def test_direct_tool_calls_no_events(self) -> None:
        """Simulate an older shim that appends to session.tool_calls directly."""
        from benchflow.acp.session import ToolCallRecord

        session = ACPSession("s1")
        tc = ToolCallRecord("tc_1", "echo hi", "bash")
        tc.update_status(ToolCallStatus.COMPLETED)
        session.tool_calls.append(tc)
        session.message_chunks.append("hello")
        # events list is empty — should use legacy fallback

        result = _capture_session_trajectory(session)
        # Legacy path: tool_calls first, then message blob
        assert len(result) == 2
        assert result[0]["type"] == "tool_call"
        assert result[1]["type"] == "agent_message"
        assert result[1]["text"] == "hello"


class TestIdempotentCapture:
    """Calling _capture_session_trajectory multiple times is safe (PR #214)."""

    def test_repeated_capture_same_result(self) -> None:
        session = ACPSession("s1")
        session.record_user_prompt("Go")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "done"},
            }
        )
        session.mark_prompt_end()

        first = _capture_session_trajectory(session)
        second = _capture_session_trajectory(session)
        assert first == second

    def test_incremental_capture_after_second_prompt(self) -> None:
        """Trajectory grows correctly when captured between prompts."""
        session = ACPSession("s1")

        session.record_user_prompt("P1")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.mark_prompt_end()

        first = _capture_session_trajectory(session)
        assert len(first) == 2  # user_message + tool_call

        session.record_user_prompt("P2")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_2",
                "title": "cat",
                "kind": "read",
            }
        )
        session.mark_prompt_end()

        second = _capture_session_trajectory(session)
        assert len(second) == 4  # P1 events + P2 events
        assert second[:2] == first


class TestFlushArrivalOrder:
    """Verify _flush_agent_text preserves chunk arrival order (PR #214)."""

    def test_thought_before_message_in_same_flush(self) -> None:
        """Real Claude pattern: thinking block → text block → tool_use block.

        ACP server sends agent_thought_chunk, agent_message_chunk, then
        tool_call. The flush at tool_call must output thought BEFORE message.
        """
        session = ACPSession("s1")
        session.record_user_prompt("Go")
        # Claude response: [thinking, text, tool_use]
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "I should check the file"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Let me look at that."},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "cat file.txt",
                "kind": "read",
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert types == ["user_message", "agent_thought", "agent_message", "tool_call"]

    def test_message_before_thought_in_same_flush(self) -> None:
        """Reverse order: message arrives before thought."""
        session = ACPSession("s1")
        session.record_user_prompt("Go")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Here is the result."},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "Now let me continue."},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        # message arrived first → message should be before thought
        assert types == ["user_message", "agent_message", "agent_thought", "tool_call"]

    def test_interleaved_thought_message_thought_not_merged(self) -> None:
        """[thought, message, thought] must produce 3 events, not merge the thoughts."""
        session = ACPSession("s1")
        session.record_user_prompt("Go")
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "first thought"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "a message"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "second thought"},
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert types == [
            "user_message",
            "agent_thought",
            "agent_message",
            "agent_thought",
        ]
        thoughts = [e for e in result if e["type"] == "agent_thought"]
        assert thoughts[0]["text"] == "first thought"
        assert thoughts[1]["text"] == "second thought"

    def test_consecutive_same_type_chunks_merged(self) -> None:
        """Multiple thought chunks before a tool_call merge into one event."""
        session = ACPSession("s1")
        session.record_user_prompt("Go")
        for i in range(5):
            session.handle_update(
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": f"part{i} "},
                }
            )
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        thoughts = [e for e in result if e["type"] == "agent_thought"]
        assert len(thoughts) == 1
        assert thoughts[0]["text"] == "part0 part1 part2 part3 part4 "

    def test_only_user_message_no_agent_response(self) -> None:
        """Agent timeout immediately: only user_message in trajectory."""
        session = ACPSession("s1")
        session.record_user_prompt("Solve it")
        # Timeout — no agent response at all

        result = _capture_session_trajectory(session)
        assert len(result) == 1
        assert result[0] == {"type": "user_message", "text": "Solve it"}

    def test_pending_cleared_between_flushes(self) -> None:
        """Chunks after a flush go into a fresh pending list."""
        session = ACPSession("s1")
        session.record_user_prompt("Go")

        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "before tool"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        # After tool_call flush, new chunks start fresh
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "after tool"},
            }
        )
        session.mark_prompt_end()

        result = _capture_session_trajectory(session)
        types = [e["type"] for e in result]
        assert types == [
            "user_message",
            "agent_thought",
            "tool_call",
            "agent_message",
        ]
        assert result[1]["text"] == "before tool"
        assert result[3]["text"] == "after tool"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestSnapshotSessionTrajectory:
    """Non-destructive snapshot preserves chunk streaming until prompt_end."""

    def test_snapshot_does_not_flush_pending(self) -> None:
        session = ACPSession("s1")
        session.record_user_prompt("Solve")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Hel"},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "lo"},
            }
        )

        # Snapshotting must NOT collapse pending_text into committed events,
        # otherwise the next chunk would become a separate event rather than
        # being merged with its siblings at mark_prompt_end.
        snapshot = _snapshot_session_trajectory(session)
        assert len(session._pending_text) == 2
        assert [e["type"] for e in snapshot] == ["user_message", "agent_message"]
        assert snapshot[1]["text"] == "Hello"

        # Final capture after prompt_end produces the same merged text.
        session.mark_prompt_end()
        final = _capture_session_trajectory(session)
        assert [e["type"] for e in final] == ["user_message", "agent_message"]
        assert final[1]["text"] == "Hello"

    def test_snapshot_after_prompt_end_matches_capture(self) -> None:
        session = ACPSession("s1")
        session.record_user_prompt("Hi")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Done"},
            }
        )
        session.mark_prompt_end()
        assert _snapshot_session_trajectory(session) == _capture_session_trajectory(
            session
        )


class TestTrajectoryWriter:
    """Streams incremental snapshots to disk as the session evolves."""

    def test_writer_flushes_after_each_update(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "trajectory" / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        session = ACPSession("s1")
        session.on_change = writer

        # File doesn't exist until the first event lands.
        assert not traj_path.exists()

        session.record_user_prompt("Solve the task")
        assert traj_path.exists(), "file should appear on the first event"
        snapshot1 = _read_jsonl(traj_path)
        assert [e["type"] for e in snapshot1] == ["user_message"]
        assert snapshot1[0]["text"] == "Solve the task"

        # Tool call appears immediately at PENDING.
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls /app",
                "kind": "bash",
            }
        )
        snapshot2 = _read_jsonl(traj_path)
        assert [e["type"] for e in snapshot2] == ["user_message", "tool_call"]
        assert snapshot2[1]["status"] == ToolCallStatus.PENDING.value

        # Status transitions are visible on the next snapshot, same ID.
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        snapshot3 = _read_jsonl(traj_path)
        assert len(snapshot3) == 2
        assert snapshot3[1]["tool_call_id"] == "tc_1"
        assert snapshot3[1]["status"] == ToolCallStatus.COMPLETED.value

        # Streamed message chunks are visible mid-prompt as a merged event.
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "All "},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "done."},
            }
        )
        snapshot4 = _read_jsonl(traj_path)
        assert [e["type"] for e in snapshot4] == [
            "user_message",
            "tool_call",
            "agent_message",
        ]
        assert snapshot4[2]["text"] == "All done."

        # mark_prompt_end is idempotent w.r.t. on-disk content.
        session.mark_prompt_end()
        final = _read_jsonl(traj_path)
        assert final == _capture_session_trajectory(session)

    def test_writer_swallows_callback_errors(self, tmp_path: Path) -> None:
        """A broken sink must not propagate into ACP update handling."""
        session = ACPSession("s1")

        def boom(_session: ACPSession) -> None:
            raise RuntimeError("nope")

        session.on_change = boom
        session.record_user_prompt("Solve")  # must not raise
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )

    def test_writer_atomic_rewrite_creates_no_torn_lines(self, tmp_path: Path) -> None:
        """Each snapshot is a complete JSONL document — no .tmp leftover."""
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        session = ACPSession("s1")
        session.on_change = writer
        session.record_user_prompt("Solve")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
            }
        )
        # No .tmp sibling left behind, and every line parses as JSON.
        assert not traj_path.with_suffix(traj_path.suffix + ".tmp").exists()
        lines = traj_path.read_text().splitlines()
        for line in lines:
            json.loads(line)

    def test_write_final_overwrites_streamed_file(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        session = ACPSession("s1")
        session.on_change = writer
        session.record_user_prompt("Solve")
        assert traj_path.exists()
        writer.write_final(
            [{"type": "oracle", "command": "solve.sh", "return_code": 0}]
        )
        snapshot = _read_jsonl(traj_path)
        assert snapshot == [{"type": "oracle", "command": "solve.sh", "return_code": 0}]


class TestMultiSceneCumulativeStreaming:
    """The streaming writer must include events from prior scenes — not just
    the current session — so multi-scene rollouts don't lose history on
    disk between scene transitions.
    """

    def test_sink_includes_prior_trajectory(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        prior = [
            {"type": "user_message", "text": "scene 1 prompt"},
            {
                "type": "tool_call",
                "tool_call_id": "tc_s1",
                "kind": "bash",
                "title": "ls",
                "status": "completed",
                "content": [],
            },
            {"type": "agent_message", "text": "scene 1 done"},
        ]
        session = ACPSession("s2")
        session.on_change = make_trajectory_sink(writer, prior)

        session.record_user_prompt("scene 2 prompt")
        snapshot = _read_jsonl(traj_path)
        # Scene 1's 3 events must be preserved; scene 2's user_message appended.
        assert len(snapshot) == 4
        assert snapshot[0]["text"] == "scene 1 prompt"
        assert snapshot[1]["tool_call_id"] == "tc_s1"
        assert snapshot[2]["text"] == "scene 1 done"
        assert snapshot[3]["text"] == "scene 2 prompt"

    def test_empty_new_session_does_not_wipe_prior(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        prior = [{"type": "user_message", "text": "scene 1 only event"}]
        # Seed the file with the prior content so we can detect overwrites.
        writer.write_final(prior)
        session = ACPSession("s2")
        session.on_change = make_trajectory_sink(writer, prior)

        # No mutations on this session at all — but the sink fires anyway
        # (e.g. ACP sent an unknown sessionUpdate that mutated nothing).
        session.on_change(session)
        snapshot = _read_jsonl(traj_path)
        assert snapshot == prior, "empty session must not wipe prior events"

    def test_sink_isolates_prior_snapshot_from_caller_mutation(
        self, tmp_path: Path
    ) -> None:
        """The prior list reference is captured at wire-up time; later
        mutations by the caller (e.g. Rollout.execute extending its own
        trajectory) must NOT cause double-counting of the current session.
        """
        traj_path = tmp_path / "acp_trajectory.jsonl"
        writer = TrajectoryWriter(traj_path)
        prior: list[dict] = [{"type": "user_message", "text": "prior"}]
        session = ACPSession("s2")
        session.on_change = make_trajectory_sink(writer, prior)

        session.record_user_prompt("current")
        # Caller appends current session events to its own cumulative list
        # — simulates Rollout.execute extending self._trajectory after
        # execute_prompts returns. The sink must keep using the prior
        # slice from wire-up time and not double-count.
        prior.append({"type": "user_message", "text": "current"})

        # A subsequent on_change should still produce: ["prior", "current"],
        # NOT ["prior", "current", "current"].
        session.on_change(session)
        snapshot = _read_jsonl(traj_path)
        assert [e.get("text") for e in snapshot] == ["prior", "current"]


class TestHandleUpdateUnknownType:
    """Unknown sessionUpdate types should not trigger on_change — no state
    mutated, no reason to re-snapshot.
    """

    def test_unknown_update_type_skips_notify(self) -> None:
        session = ACPSession("s1")
        calls: list[int] = []
        session.on_change = lambda _s: calls.append(1)

        session.handle_update({"sessionUpdate": "unknown_future_type"})
        assert calls == [], "on_change must NOT fire for unrecognized update types"

    def test_known_update_type_still_notifies(self) -> None:
        session = ACPSession("s1")
        calls: list[int] = []
        session.on_change = lambda _s: calls.append(1)

        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "ls",
                "kind": "bash",
            }
        )
        assert calls == [1], "on_change must fire for recognized update types"


class TestTrajectoryWriterStaleTmpCleanup:
    """Stale .tmp file left by a previous crashed run must be swept on
    writer construction so a follow-up reader can't pick it up.
    """

    def test_init_unlinks_pre_existing_tmp(self, tmp_path: Path) -> None:
        traj_path = tmp_path / "acp_trajectory.jsonl"
        stale_tmp = traj_path.with_suffix(traj_path.suffix + ".tmp")
        stale_tmp.parent.mkdir(parents=True, exist_ok=True)
        stale_tmp.write_text('{"type":"user_message","text":"orphaned"}')
        assert stale_tmp.exists()

        TrajectoryWriter(traj_path)
        assert not stale_tmp.exists(), "stale .tmp must be cleaned up on init"

    def test_init_tolerates_no_pre_existing_tmp(self, tmp_path: Path) -> None:
        # Clean construction must not raise when no stale tmp is present.
        TrajectoryWriter(tmp_path / "acp_trajectory.jsonl")
