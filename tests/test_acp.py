"""Tests for ACP client ↔ mock agent — Step 10."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.acp.client import ACPClient, ACPError
from benchflow.acp.container_transport import ContainerTransport
from benchflow.acp.session import ACPSession
from benchflow.acp.transport import StdioTransport
from benchflow.acp.types import StopReason, ToolCallStatus

MOCK_AGENT = str(Path(__file__).parent / "fixtures" / "mock_acp_agent.py")
MOCK_AGENT_INTERLEAVED = str(
    Path(__file__).parent / "fixtures" / "mock_acp_agent_interleaved.py"
)


class TestACPClient:
    def test_initialize_params_send_current_protocol_version(self) -> None:
        """The client must advertise ACP protocol version 1, not 0.

        ``ACP_PROTOCOL_VERSION`` is sourced from the official SDK's
        ``acp.meta.PROTOCOL_VERSION``; the SDK-backed ``InitializeParams``
        (``acp.schema.InitializeRequest``) carries it on the wire.
        """
        from benchflow.acp.types import (
            ACP_PROTOCOL_VERSION,
            AuthCapabilities,
            ClientCapabilities,
            ClientInfo,
            FsCapabilities,
            InitializeParams,
        )

        assert ACP_PROTOCOL_VERSION == 1
        params = InitializeParams(
            protocol_version=ACP_PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                fs=FsCapabilities(read_text_file=True, write_text_file=True),
                terminal=True,
                auth=AuthCapabilities(),
            ),
            client_info=ClientInfo(name="benchflow", version="2.0.0"),
        )
        wire = params.model_dump(by_alias=True, exclude_none=True)
        assert wire["protocolVersion"] == 1

    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            result = await client.initialize()
            # Negotiated to v1 — the mock echoes min(requested, 1).
            assert result.protocol_version == 1
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
    async def test_initialize_advertises_auth_methods(self) -> None:
        """initialize() surfaces the agent's advertised ACP auth methods."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            result = await client.initialize()
            assert result.auth_methods is not None
            assert [m.id for m in result.auth_methods] == ["api-key"]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_authenticate_with_advertised_method(self) -> None:
        """authenticate() succeeds for a method ID the agent advertises."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            result = await client.initialize()
            method_id = result.auth_methods[0].id
            # authenticate runs after initialize, before session/new.
            response = await client.authenticate(method_id)
            assert response == {}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_authenticate_unknown_method_raises(self) -> None:
        """authenticate() raises ACPError for a method the agent rejects."""
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            with pytest.raises(ACPError):
                await client.authenticate("not-a-real-method")
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
        """Oversized line on stdout is skipped; following JSON-RPC returns normally.

        Feeds the oversized chunk first, lets receive() hit LimitOverrunError and
        call drain_oversized_line (which clears the buffer), then feeds the next
        valid JSON-RPC line so readline() can find it. This matches the real
        ordering (stdin -> drain -> next line) that a live process would produce.
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
            # real JSON-RPC line that receive() should return.
            await asyncio.sleep(0.05)
            reader.feed_data(b"\n")
            await asyncio.sleep(0.05)
            reader.feed_data(b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n')
            reader.feed_eof()

        feeder = asyncio.create_task(feed_valid_later())
        try:
            msg = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await feeder
        assert msg == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}


class TestTransportProtocolFiltering:
    """Transports skip JSON-encoded log scalars and wait for JSON-RPC objects."""

    @pytest.mark.asyncio
    async def test_stdio_transport_skips_json_scalars(self) -> None:
        """Guards PR #236 against treating JSON scalars as ACP responses."""
        reader = asyncio.StreamReader()
        reader.feed_data(b'"debug string from agent"\n')
        reader.feed_data(b'["debug", "list"]\n')
        reader.feed_data(b"123\n")
        reader.feed_data(b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n')
        reader.feed_eof()

        transport = StdioTransport(sys.executable, [])
        fake_process = MagicMock()
        fake_process.stdout = reader
        fake_process.stdin = MagicMock()
        transport._process = fake_process

        msg = await asyncio.wait_for(transport.receive(), timeout=5)
        assert msg == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}

    @pytest.mark.asyncio
    async def test_container_transport_skips_json_scalars(self, tmp_path) -> None:
        """Guards PR #236 against treating JSON scalars as ACP responses."""
        fake_process = AsyncMock()
        fake_process.readline = AsyncMock(
            side_effect=[
                b'"debug string from agent"\n',
                b'["debug", "list"]\n',
                b'{"jsonrpc": "2.0", "id": 2, "result": {"ok": true}}\n',
            ]
        )
        agent_log = tmp_path / "agent.log"
        transport = ContainerTransport(
            container_process=fake_process,
            command="agent acp",
            agent_log_path=agent_log,
        )

        await transport.start()
        try:
            msg = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await transport.close()

        assert msg == {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}}
        log_text = agent_log.read_text()
        assert '"debug string from agent"' in log_text
        assert '["debug", "list"]' in log_text

    @pytest.mark.asyncio
    async def test_stdio_transport_skips_structured_json_logs(self) -> None:
        """Guards PR #236 against treating JSON object logs as ACP responses."""
        reader = asyncio.StreamReader()
        reader.feed_data(b'{"id": 100001, "level": "info", "message": "startup"}\n')
        reader.feed_data(b'{"jsonrpc": "2.0", "id": 100001, "result": {"ok": true}}\n')
        reader.feed_eof()

        transport = StdioTransport(sys.executable, [])
        fake_process = MagicMock()
        fake_process.stdout = reader
        fake_process.stdin = MagicMock()
        transport._process = fake_process
        client = ACPClient(transport)

        result = await asyncio.wait_for(client._read_until_response(100001), timeout=5)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_container_transport_logs_structured_json_logs(
        self, tmp_path
    ) -> None:
        """Guards PR #236 against treating JSON object logs as ACP responses."""
        fake_process = AsyncMock()
        fake_process.readline = AsyncMock(
            side_effect=[
                b'{"id": 2, "level": "info", "message": "startup"}\n',
                b'{"jsonrpc": "2.0", "id": 2, "result": {"ok": true}}\n',
            ]
        )
        agent_log = tmp_path / "agent.log"
        transport = ContainerTransport(
            container_process=fake_process,
            command="agent acp",
            agent_log_path=agent_log,
        )

        await transport.start()
        try:
            msg = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await transport.close()

        assert msg == {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}}
        assert '{"id": 2, "level": "info", "message": "startup"}' in (
            agent_log.read_text()
        )


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


class TestACPIdleWatchdog:
    @pytest.mark.asyncio
    async def test_idle_watchdog_returns_even_when_prompt_cancel_drain_stalls(
        self,
    ) -> None:
        """Guards the 2026-05-22 Daytona/Gemini blocker fix against stuck cancel drain."""
        from benchflow.acp.runtime import execute_prompts

        class StubbornPromptClient:
            def __init__(self) -> None:
                self.release = asyncio.Event()
                self.task: asyncio.Task | None = None

            async def prompt(self, _prompt: str):
                self.task = asyncio.current_task()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    await self.release.wait()
                    raise

        client = StubbornPromptClient()
        session = ACPSession("idle-session")

        try:
            started = asyncio.get_running_loop().time()
            with pytest.raises(TimeoutError, match="Agent idle for 1s"):
                await asyncio.wait_for(
                    execute_prompts(
                        client,  # type: ignore[arg-type]
                        session,
                        ["solve"],
                        timeout=30,
                        idle_timeout=1,
                    ),
                    timeout=1.8,
                )
            elapsed = asyncio.get_running_loop().time() - started
            assert elapsed < 1.35
        finally:
            client.release.set()
            if client.task is not None:
                with pytest.raises(asyncio.CancelledError):
                    await client.task


class TestIdleTimeoutDiagnostics:
    """Guards ENG-149: idle timeouts must carry structured diagnostics."""

    @pytest.mark.asyncio
    async def test_idle_timeout_raises_with_structured_info(self) -> None:
        """Guards ENG-149: IdleTimeoutError carries idle_timeout_info dict."""
        from benchflow.acp.runtime import IdleTimeoutError, execute_prompts

        class HangingClient:
            async def prompt(self, _prompt: str):
                await asyncio.Future()

        session = ACPSession("diag-session")
        with pytest.raises(IdleTimeoutError) as exc_info:
            await execute_prompts(
                HangingClient(),  # type: ignore[arg-type]
                session,
                ["solve"],
                timeout=30,
                idle_timeout=1,
            )
        info = exc_info.value.idle_timeout_info
        assert info["reason"] == "idle_timeout"
        assert info["idle_timeout_sec"] == 1
        assert info["idle_duration_sec"] >= 1
        assert isinstance(info["n_tool_calls"], int)
        assert isinstance(info["n_message_chunks"], int)
        assert isinstance(info["n_thought_chunks"], int)
        assert isinstance(info["wall_clock_elapsed_sec"], int)
        assert "last_activity_at" in info

    @pytest.mark.asyncio
    async def test_idle_timeout_info_reflects_activity_counts(self) -> None:
        """Guards ENG-149: diagnostics include the session's activity counts."""
        from benchflow.acp.runtime import IdleTimeoutError, execute_prompts

        class OneToolThenHang:
            def __init__(self, session):
                self._session = session
                self._called = False

            async def prompt(self, _prompt: str):
                if not self._called:
                    self._called = True
                    self._session.tool_calls.append(
                        MagicMock(status=ToolCallStatus.COMPLETED)
                    )
                    await asyncio.sleep(0.1)
                await asyncio.Future()

        session = ACPSession("diag-activity")
        client = OneToolThenHang(session)
        with pytest.raises(IdleTimeoutError) as exc_info:
            await execute_prompts(
                client,  # type: ignore[arg-type]
                session,
                ["solve"],
                timeout=30,
                idle_timeout=1,
            )
        info = exc_info.value.idle_timeout_info
        assert info["n_tool_calls"] == 1

    @pytest.mark.asyncio
    async def test_wall_clock_timeout_has_no_idle_info(self) -> None:
        """Wall-clock timeouts (not idle) must NOT carry idle_timeout_info."""
        from benchflow.acp.runtime import execute_prompts

        class SlowClient:
            async def prompt(self, _prompt: str):
                await asyncio.Future()

        session = ACPSession("wall-clock-session")
        # Add continuous activity to prevent idle timeout
        session.tool_calls.append(MagicMock(status=ToolCallStatus.COMPLETED))
        with pytest.raises(TimeoutError) as exc_info:
            await execute_prompts(
                SlowClient(),  # type: ignore[arg-type]
                session,
                ["solve"],
                timeout=2,
                idle_timeout=None,
            )
        assert not hasattr(exc_info.value, "idle_timeout_info")


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
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()

        mock_env = AsyncMock()
        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="test-agent",
                agent_launch="test-agent",
                agent_env={},
                sandbox_user=None,
                model=model_in,
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_awaited_once_with(expected_model)

    @pytest.mark.asyncio
    async def test_pi_acp_preserves_registered_provider_prefix(self, tmp_path):
        """Guards PR #291: Pi set_model needs the registered provider key."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = AsyncMock()
        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="pi-acp",
                agent_launch="pi-acp",
                agent_env={},
                sandbox_user=None,
                model="vllm/Qwen/Qwen3.5-35B-A3B",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_awaited_once_with("vllm/Qwen/Qwen3.5-35B-A3B")

    @pytest.mark.asyncio
    async def test_opencode_keeps_modelsdev_formatting(self, tmp_path):
        """Registered BenchFlow providers must not become models.dev provider IDs."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = AsyncMock()
        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="opencode",
                agent_launch="opencode acp",
                agent_env={},
                sandbox_user=None,
                model="vllm/Qwen/Qwen3.5-35B-A3B",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_awaited_once_with("Qwen/Qwen3.5-35B-A3B")

    @pytest.mark.asyncio
    async def test_openhands_skips_set_model(self, tmp_path):
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = AsyncMock()
        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="openhands",
                agent_launch="openhands acp --always-approve --override-with-envs",
                agent_env={},
                sandbox_user=None,
                model="gemini-3.1-flash-lite-preview",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_claude_bedrock_sets_model_from_provider_mapping(self, tmp_path):
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = AsyncMock()
        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="claude-agent-acp",
                agent_launch="claude-agent-acp",
                agent_env={
                    "CLAUDE_CODE_USE_BEDROCK": "1",
                    "ANTHROPIC_MODEL": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                },
                sandbox_user=None,
                model="aws-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_awaited_once_with(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        )

    @pytest.mark.asyncio
    async def test_daytona_dind_uses_pty_transport(self, tmp_path):
        """Daytona compose tasks use PTY transport to avoid SSH pipe-closed failures."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = MagicMock()
        mock_env.exec = AsyncMock(return_value=MagicMock(return_code=1, stdout=""))
        mock_env._strategy = MagicMock()
        mock_env._strategy._compose_cmd = MagicMock(return_value="docker compose -p t")

        with (
            patch(
                "benchflow.acp.runtime.DaytonaPtyProcess.from_sandbox_env",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_pty,
            patch(
                "benchflow.acp.runtime.DaytonaProcess.from_sandbox_env",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_ssh,
            patch("benchflow.acp.runtime.ContainerTransport", return_value=MagicMock()),
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="test-agent",
                agent_launch="test-agent",
                agent_env={},
                sandbox_user=None,
                model=None,
                rollout_dir=tmp_path,
                environment="daytona",
                agent_cwd="/app",
            )

        mock_pty.assert_awaited_once_with(mock_env)
        mock_ssh.assert_not_awaited()


class TestSandboxStartupDiagnostics:
    """Guards ENG-147: sandbox startup failures must carry structured diagnostics."""

    def test_sandbox_startup_error_has_info_dict(self) -> None:
        """Guards ENG-147: SandboxStartupError carries sandbox_startup_info dict
        with all required fields for result.json."""
        from benchflow.sandbox.daytona import SandboxStartupError

        err = SandboxStartupError(
            "Sandbox creation failed after retries: timeout of 1200000ms exceeded",
            sandbox_id="e7d8ab0f-47da-40b1-b179-46e1363fe014",
            sandbox_state="creating",
            attempts=3,
            build_timeout_sec=600.0,
        )
        info = err.sandbox_startup_info
        assert info["reason"] == "sandbox_startup_failed"
        assert info["sandbox_id"] == "e7d8ab0f-47da-40b1-b179-46e1363fe014"
        assert info["sandbox_state"] == "creating"
        assert info["attempts"] == 3
        assert info["build_timeout_sec"] == 600.0
        assert "timeout of 1200000ms" in info["raw_message"]

    def test_sandbox_startup_error_is_runtime_error(self) -> None:
        """Guards ENG-147: SandboxStartupError is a RuntimeError subclass
        so existing except-RuntimeError paths still catch it."""
        from benchflow.sandbox.daytona import SandboxStartupError

        err = SandboxStartupError("test")
        assert isinstance(err, RuntimeError)

    def test_classify_error_sandbox_startup(self) -> None:
        """Guards ENG-147: classify_error recognises sandbox startup failures."""
        from benchflow._utils.scoring import SANDBOX_SETUP, classify_error

        assert classify_error("Sandbox startup failed: timeout") == SANDBOX_SETUP
        assert classify_error("Sandbox creation failed after retries") == SANDBOX_SETUP
        assert classify_error("normal error") != SANDBOX_SETUP

    def test_sandbox_startup_info_in_result_json(self, tmp_path: Path) -> None:
        """Guards ENG-147: _build_rollout_result writes sandbox_startup_info to result.json."""
        from benchflow.rollout import _build_rollout_result

        info = {
            "reason": "sandbox_startup_failed",
            "sandbox_id": "abc123",
            "sandbox_state": "error",
            "attempts": 3,
            "build_timeout_sec": 600.0,
            "raw_message": "timeout",
        }
        result = _build_rollout_result(
            tmp_path,
            task_name="test-task",
            rollout_name="run-1",
            agent="oracle",
            agent_name="oracle",
            model=None,
            n_tool_calls=0,
            prompts=["solve"],
            error="Sandbox startup failed: timeout",
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards=None,
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 0.0},
            sandbox_startup_info=info,
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["sandbox_startup_info"] == info
        assert rj["error_category"] == "sandbox_setup"
        assert result.error == "Sandbox startup failed: timeout"

    def test_sandbox_startup_info_null_when_no_startup_error(
        self, tmp_path: Path
    ) -> None:
        """Guards ENG-147: sandbox_startup_info is null for non-startup errors."""
        from benchflow.rollout import _build_rollout_result

        result = _build_rollout_result(
            tmp_path,
            task_name="test-task",
            rollout_name="run-1",
            agent="oracle",
            agent_name="oracle",
            model=None,
            n_tool_calls=5,
            prompts=["solve"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1.0},
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 5.0},
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["sandbox_startup_info"] is None
        assert result.rewards == {"reward": 1.0}

    def test_create_sandbox_retry_count_is_three(self) -> None:
        """Guards ENG-147: _create_sandbox retries 3 times, not 2."""
        from benchflow.sandbox.daytona import DaytonaSandbox

        retry_obj = DaytonaSandbox._create_sandbox.retry  # type: ignore[attr-defined]
        stop = retry_obj.stop
        assert stop.max_attempt_number == 3


class TestTransportErrorDiagnostics:
    """Guards ENG-148: ACP transport rc=255 must carry structured diagnostics."""

    def test_parse_transport_error_rc255(self) -> None:
        """Guards ENG-148: _parse_transport_error extracts rc, pid, diagnosis from
        the ConnectionError message produced by process.py when SSH dies."""
        from benchflow.rollout import _parse_transport_error

        err = ConnectionError(
            "Process closed stdout (rc=255): "
            "Local subprocess exited with rc=255 before stdout closed.\n"
            "stderr: Connection to sandbox lost"
        )
        info = _parse_transport_error(err)
        assert info["reason"] == "transport_closed"
        assert info["process_exit_code"] == 255
        assert info["transport_diagnosis"] == "process_exited"
        assert "Connection to sandbox lost" in info["stderr_snippet"]

    def test_parse_transport_error_rc_none_remote_killed(self) -> None:
        """Guards ENG-148: rc=None means the local process is alive but the remote
        transport (SSH/Daytona) was killed."""
        from benchflow.rollout import _parse_transport_error

        err = ConnectionError(
            "Process closed stdout (rc=None): "
            "Local subprocess (pid=12345) is still alive but its "
            "stdout/transport closed. This usually means the remote "
            "container or SSH session was killed"
        )
        info = _parse_transport_error(err)
        assert info["process_exit_code"] is None
        assert info["process_pid"] == 12345
        assert info["transport_diagnosis"] == "remote_session_killed"

    def test_parse_transport_error_pty(self) -> None:
        """Guards ENG-148: PTY readline errors get distinct diagnosis."""
        from benchflow.rollout import _parse_transport_error

        err = ConnectionError("PTY readline timeout (900s)")
        info = _parse_transport_error(err)
        assert info["transport_diagnosis"] == "pty_error"

    def test_parse_transport_error_unknown(self) -> None:
        """Guards ENG-148: unrecognized ConnectionError gets diagnosis=unknown."""
        from benchflow.rollout import _parse_transport_error

        err = ConnectionError("something unexpected")
        info = _parse_transport_error(err)
        assert info["transport_diagnosis"] == "unknown"
        assert info["reason"] == "transport_closed"

    def test_transport_error_info_in_result_json(self, tmp_path) -> None:
        """Guards ENG-148: transport_error_info is written to result.json."""
        from benchflow.rollout import _build_rollout_result

        transport_info = {
            "reason": "transport_closed",
            "process_exit_code": 255,
            "transport_diagnosis": "process_exited",
            "sandbox_reachable": False,
        }
        result = _build_rollout_result(
            tmp_path,
            task_name="video-filler-word-remover",
            rollout_name="video-filler__abc123",
            agent="gemini",
            agent_name="gemini-cli",
            model="gemini-2.0-flash-lite",
            n_tool_calls=8,
            prompts=["solve"],
            error="Process closed stdout (rc=255): Local subprocess exited with rc=255",
            verifier_error=None,
            trajectory=[],
            partial_trajectory=True,
            rewards=None,
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 10.0},
            transport_error_info=transport_info,
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["transport_error_info"] == transport_info
        assert rj["error_category"] == "pipe_closed"
        assert result.error is not None

    def test_transport_error_info_none_when_no_transport_error(self, tmp_path) -> None:
        """Guards ENG-148: transport_error_info is null for non-transport errors."""
        from benchflow.rollout import _build_rollout_result

        result = _build_rollout_result(
            tmp_path,
            task_name="hello-world",
            rollout_name="hello__abc",
            agent="gemini",
            agent_name="gemini-cli",
            model="gemini-2.0-flash-lite",
            n_tool_calls=5,
            prompts=["solve"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1.0},
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 5.0},
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["transport_error_info"] is None
        assert result.rewards == {"reward": 1.0}
