"""Tests for _capture_session_trajectory — ensures partial trajectory is saved on timeout."""

from benchflow.acp.session import ACPSession
from benchflow.acp.types import ToolCallStatus
from benchflow.sdk import _capture_session_trajectory


class TestCaptureSessionTrajectory:
    def test_none_session_returns_empty(self) -> None:
        assert _capture_session_trajectory(None) == []

    def test_empty_session_returns_empty(self) -> None:
        session = ACPSession("s1")
        assert _capture_session_trajectory(session) == []

    def test_captures_tool_calls(self) -> None:
        session = ACPSession("s1")
        session.handle_update({
            "sessionUpdate": "tool_call",
            "toolCallId": "tc_1",
            "title": "echo hello",
            "kind": "bash",
        })
        session.handle_update({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc_1",
            "status": "completed",
        })
        result = _capture_session_trajectory(session)
        assert len(result) == 1
        assert result[0]["type"] == "tool_call"
        assert result[0]["tool_call_id"] == "tc_1"
        assert result[0]["kind"] == "bash"
        assert result[0]["title"] == "echo hello"
        assert result[0]["status"] == ToolCallStatus.COMPLETED.value

    def test_captures_message(self) -> None:
        session = ACPSession("s1")
        session.handle_update({
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "Hello world"},
        })
        result = _capture_session_trajectory(session)
        assert len(result) == 1
        assert result[0]["type"] == "agent_message"
        assert result[0]["text"] == "Hello world"

    def test_captures_thought(self) -> None:
        session = ACPSession("s1")
        session.handle_update({
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "thinking..."},
        })
        result = _capture_session_trajectory(session)
        assert len(result) == 1
        assert result[0]["type"] == "agent_thought"
        assert result[0]["text"] == "thinking..."

    def test_captures_partial_state(self) -> None:
        """Simulates timeout mid-execution: tool calls exist but are still in-progress."""
        session = ACPSession("s1")
        # First tool call completed
        session.handle_update({
            "sessionUpdate": "tool_call",
            "toolCallId": "tc_1",
            "title": "ls /app",
            "kind": "bash",
        })
        session.handle_update({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc_1",
            "status": "completed",
        })
        # Second tool call still in progress (timeout happened here)
        session.handle_update({
            "sessionUpdate": "tool_call",
            "toolCallId": "tc_2",
            "title": "cat README.md",
            "kind": "bash",
        })
        # Partial message streamed before timeout
        session.handle_update({
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "Let me read"},
        })

        result = _capture_session_trajectory(session)
        assert len(result) == 3  # 2 tool calls + 1 message
        assert result[0]["status"] == ToolCallStatus.COMPLETED.value
        assert result[1]["status"] == ToolCallStatus.PENDING.value
        assert result[2]["type"] == "agent_message"
        assert result[2]["text"] == "Let me read"

    def test_captures_all_types_together(self) -> None:
        session = ACPSession("s1")
        session.handle_update({
            "sessionUpdate": "tool_call",
            "toolCallId": "tc_1",
            "title": "echo hi",
            "kind": "bash",
        })
        session.handle_update({
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "hmm"},
        })
        session.handle_update({
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "done"},
        })
        result = _capture_session_trajectory(session)
        assert len(result) == 3
        types = [e["type"] for e in result]
        assert types == ["tool_call", "agent_message", "agent_thought"]
