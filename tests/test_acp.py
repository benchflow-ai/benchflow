"""Tests for ACP client ↔ mock agent — Step 10."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient, ACPError
from benchflow.acp.session import ACPSession
from benchflow.acp.transport import StdioTransport
from benchflow.acp.types import StopReason, ToolCallStatus

MOCK_AGENT = str(Path(__file__).parent / "fixtures" / "mock_acp_agent.py")
MOCK_AGENT_INTERLEAVED = str(
    Path(__file__).parent / "fixtures" / "mock_acp_agent_interleaved.py"
)


class TestACPClient:
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            result = await client.initialize()
            assert result.protocol_version == 0
            assert result.agent_info is not None
            assert result.agent_info.name == "mock-agent"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_session_new(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            session = await client.session_new()
            assert session.session_id == "mock-session-1"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_prompt_and_response(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            await client.session_new()

            result = await client.prompt("Hello, agent!")
            assert result.stop_reason == StopReason.END_TURN

            session = client.session
            assert session is not None
            assert "I received: Hello, agent!" in session.full_message
            assert len(session.tool_calls) == 1
            assert session.tool_calls[0].kind == "bash"
            assert session.tool_calls[0].title == "echo hello"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_from_config_stdio(self) -> None:
        client = ACPClient.from_config(
            command=f"{sys.executable} {MOCK_AGENT}",
            transport_type="stdio",
        )
        try:
            await client.connect()
            result = await client.initialize()
            assert result.agent_info.name == "mock-agent"
        finally:
            await client.close()

    def test_from_config_missing_command(self) -> None:
        with pytest.raises(ValueError, match="command required"):
            ACPClient.from_config(transport_type="stdio")

    def test_from_config_unknown_transport(self) -> None:
        with pytest.raises(ValueError, match="Unknown transport"):
            ACPClient.from_config(transport_type="sse")

    @pytest.mark.asyncio
    async def test_set_model(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            await client.session_new()
            # Mock agent returns error for unknown methods, but set_model
            # should raise ACPError for the unknown method response
            with pytest.raises(ACPError):
                await client.set_model("some-model")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_prompt_without_session_raises(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            with pytest.raises(RuntimeError, match="No active session"):
                await client.prompt("hello")
        finally:
            await client.close()


class TestACPSession:
    def test_handle_tool_call(self):
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "echo hello",
                "kind": "bash",
            }
        )
        assert len(session.tool_calls) == 1
        assert session.tool_calls[0].tool_call_id == "tc_1"
        assert session.tool_calls[0].kind == "bash"

    def test_handle_tool_call_update(self):
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "echo",
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
        assert session.tool_calls[0].status == ToolCallStatus.COMPLETED

    def test_handle_invalid_tool_call_status(self):
        """Invalid status should fall back to IN_PROGRESS, not crash."""
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "echo",
                "kind": "bash",
            }
        )
        # Should not raise — invalid status handled gracefully
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "totally_invalid_status",
            }
        )
        assert session.tool_calls[0].status == ToolCallStatus.IN_PROGRESS

    def test_handle_message_chunks(self):
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "Hello "},
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "world"},
            }
        )
        assert session.full_message == "Hello world"

    def test_handle_thought_chunks(self):
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thinking..."},
            }
        )
        assert session.full_thought == "thinking..."

    def test_handle_unknown_update_type(self):
        """Unknown update types should be silently ignored."""
        session = ACPSession("test-session")
        # Should not raise
        session.handle_update({"sessionUpdate": "unknown_type"})
        assert len(session.tool_calls) == 0

    def test_tool_call_update_for_unknown_id(self):
        """Update for non-existent tool call auto-creates a record.

        Agents like Gemini CLI skip the initial tool_call notification and
        send only tool_call_update, so the session synthesizes a record.
        """
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "nonexistent",
                "status": "completed",
            }
        )
        assert len(session.tool_calls) == 1
        assert session.tool_calls[0].tool_call_id == "nonexistent"
        assert session.tool_calls[0].kind == "tool"


class TestStdioTransportOversizedLine:
    """StdioTransport must recover from oversized lines (LimitOverrunError)."""

    @pytest.mark.asyncio
    async def test_stdio_transport_drain_oversized_line(self) -> None:
        """Oversized line on stdout is skipped; following valid JSON returns normally.

        Feeds the oversized chunk first, lets receive() hit LimitOverrunError and
        call drain_oversized_line (which clears the buffer), then feeds the next
        valid JSON line so readline() can find it. This matches the real ordering
        (stdin -> drain -> next line) that a live process would produce.
        """
        limit = 64
        reader = asyncio.StreamReader(limit=limit)
        # Oversized chunk WITH a newline: triggers "Separator is found, but
        # chunk is longer than limit" on readline().
        reader.feed_data(b"x" * (limit * 3) + b"\n")

        transport = StdioTransport(sys.executable, [])
        fake_process = MagicMock()
        fake_process.stdout = reader
        fake_process.stdin = MagicMock()
        transport._process = fake_process

        async def feed_valid_later() -> None:
            # Give receive() time to drain, then supply the next line.
            # drain_oversized_line clears the buffer and consumes up to the
            # next \n (the oversized newline was flushed with the clear), so
            # we feed a dummy newline to satisfy drain's readuntil, then the
            # real JSON line that receive() should return.
            await asyncio.sleep(0.05)
            reader.feed_data(b"\n")
            await asyncio.sleep(0.05)
            reader.feed_data(b'{"ok": true}\n')
            reader.feed_eof()

        feeder = asyncio.create_task(feed_valid_later())
        try:
            msg = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await feeder
        assert msg == {"ok": True}


class TestACPInterleaving:
    """Test that _read_until_response handles interleaved notifications and agent requests."""

    @pytest.mark.asyncio
    async def test_prompt_with_interleaved_notifications_and_request(self) -> None:
        """Notification, agent request, more notifications, then final response."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT_INTERLEAVED]))
        try:
            await client.connect()
            await client.initialize()
            await client.session_new()

            result = await client.prompt("Go!")
            assert result.stop_reason == StopReason.END_TURN

            session = client.session
            assert session is not None
            # Notifications processed: tool_call + tool_call_update + message chunk
            assert len(session.tool_calls) == 1
            assert session.tool_calls[0].status == ToolCallStatus.COMPLETED
            assert session.full_message == "done"
        finally:
            await client.close()


class TestConnectAcpModelSelection:
    """Verify connect_acp passes the right model string to set_model."""

    @staticmethod
    def _make_mocks():
        mock_session = MagicMock()
        mock_session.session_id = "s1"
        mock_init = MagicMock()
        mock_init.agent_info = None

        mock_acp = AsyncMock(spec=ACPClient)
        mock_acp.connect = AsyncMock()
        mock_acp.initialize = AsyncMock(return_value=mock_init)
        mock_acp.session_new = AsyncMock(return_value=mock_session)
        mock_acp.set_model = AsyncMock()
        mock_acp.close = AsyncMock()
        return mock_acp

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_in, expected_model",
        [
            # Registered vllm/ prefix stripped; HF org/model intact — this is
            # what pi-acp and other ACP agents need for downstream routing.
            ("vllm/Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B"),
            ("zai/glm-5", "glm-5"),
            # Bare HF ID (no registered prefix) passes through unchanged.
            ("Qwen/Qwen3-Coder", "Qwen/Qwen3-Coder"),
            # Vertex ADC provider — prefix stripped like any other registered one.
            ("anthropic-vertex/claude-sonnet-4-6", "claude-sonnet-4-6"),
            # No prefix at all — unchanged.
            ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ],
        ids=["vllm-hf", "zai", "bare-hf", "vertex", "no-prefix"],
    )
    async def test_model_id_selection(self, model_in, expected_model, tmp_path):
        from benchflow._acp_run import connect_acp

        mock_acp = self._make_mocks()

        mock_env = AsyncMock()
        with (
            patch(
                "benchflow._acp_run.DockerProcess.from_harbor_env",
                return_value=MagicMock(),
            ),
            patch("benchflow._acp_run.ContainerTransport", return_value=MagicMock()),
            patch("benchflow._acp_run.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="test-agent",
                agent_launch="test-agent",
                agent_env={},
                sandbox_user=None,
                model=model_in,
                trial_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_awaited_once_with(expected_model)
