"""Tests for _capture_session_trajectory — ensures partial trajectory is saved on timeout.

Writer / streaming / multi-scene / partial-capture-fix tests live in
``tests/test_trajectory_streaming.py``.
"""

from benchflow.acp.session import ACPSession
from benchflow.acp.types import ToolCallStatus
from benchflow.trajectories._capture import _capture_session_trajectory


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
        assert result[0]["type"] == "user_message"
        assert result[0]["text"] == "Solve it"
        assert result[0]["ts"] > 0  # rows are arrival-stamped client-side

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


class TestArrivalTimestamps:
    """ACP carries no timestamps on the wire — rows are stamped client-side at
    arrival so think-gaps and turn latency are reconstructable."""

    def test_rows_carry_arrival_ts(self):
        import time
        t0 = time.time()
        session = ACPSession("s1")
        session.record_user_prompt("play")
        session.handle_update({
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "thinking about it"},
        })
        session.handle_update({
            "sessionUpdate": "tool_call",
            "toolCallId": "t1", "title": "casino observe", "kind": "execute",
        })
        rows = _capture_session_trajectory(session)
        assert [r["type"] for r in rows] == [
            "user_message", "agent_message", "tool_call"]
        for r in rows:
            assert t0 <= r["ts"] <= time.time(), r
        ts = [r["ts"] for r in rows]
        assert ts == sorted(ts)

    def test_merged_chunks_keep_first_arrival_ts(self):
        session = ACPSession("s1")
        session.handle_update({
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "a"},
        })
        first_ts = session._pending_text[0]["ts"]
        session.handle_update({
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "b"},
        })
        rows = _capture_session_trajectory(session)
        assert rows[0]["text"] == "ab"
        assert rows[0]["ts"] == first_ts  # TTFT semantics: first chunk arrival
