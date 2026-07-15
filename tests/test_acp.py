"""Tests for ACP client ↔ mock agent — Step 10."""

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

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
    async def test_set_config_option(self) -> None:
        client = ACPClient(StdioTransport(sys.executable, [MOCK_AGENT]))
        try:
            await client.connect()
            await client.initialize()
            await client.session_new()
            with pytest.raises(ACPError):
                await client.set_config_option("model", "some-model")
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

    def test_handle_openhands_invoke_skill_update_marks_kind_skill(self):
        """Guards issue #507: OpenHands invoke_skill ACP calls are canonicalized."""
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "Load PDF skill for processing",
                "kind": "other",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
                "content": [
                    {
                        "content": {
                            "type": "text",
                            "text": "Tool: invoke_skill\nResult:\n[skill: pdf]",
                        },
                        "type": "content",
                    }
                ],
            }
        )

        assert session.tool_calls[0].kind == "skill"

    def test_handle_tool_output_quoting_skill_marker_keeps_kind(self):
        """Guards #507: a real tool whose output quotes an invoke_skill
        envelope is not reclassified as a skill (live-capture false positive)."""
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "cat prior_trajectory.txt",
                "kind": "execute",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
                "content": [
                    {
                        "content": {
                            "type": "text",
                            "text": "Tool: invoke_skill\nResult:\n[skill: pdf]",
                        },
                        "type": "content",
                    }
                ],
            }
        )

        assert session.tool_calls[0].kind == "execute"

    def test_handle_other_kind_mid_output_skill_marker_keeps_kind(self):
        """Guards #507: an unclassified tool whose output merely mentions the
        marker mid-stream (not as the result header) stays unclassified."""
        session = ACPSession("test-session")
        session.handle_update(
            {
                "sessionUpdate": "tool_call",
                "toolCallId": "tc_1",
                "title": "grep -n invoke_skill logs/",
                "kind": "other",
            }
        )
        session.handle_update(
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "tc_1",
                "status": "completed",
                "content": [
                    {
                        "content": {
                            "type": "text",
                            "text": "logs/run.txt:42:Tool: invoke_skill\n[skill: pdf]",
                        },
                        "type": "content",
                    }
                ],
            }
        )

        assert session.tool_calls[0].kind == "other"

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
    async def test_container_transport_does_not_create_empty_agent_log(
        self, tmp_path
    ) -> None:
        """Guards PR #832's fix for issue #535 against empty ACP agent logs."""
        fake_process = AsyncMock()
        fake_process.readline = AsyncMock(
            return_value=b'{"jsonrpc": "2.0", "id": 2, "result": {"ok": true}}\n'
        )
        agent_log = tmp_path / "agent" / "gemini.txt"
        transport = ContainerTransport(
            container_process=fake_process,
            command="agent acp",
            agent_log_path=agent_log,
        )

        await transport.start()
        assert not agent_log.exists()

        try:
            msg = await asyncio.wait_for(transport.receive(), timeout=5)
        finally:
            await transport.close()

        assert msg == {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}}
        assert not agent_log.exists()

    @pytest.mark.asyncio
    async def test_container_transport_clears_stale_log_on_retry(
        self, tmp_path
    ) -> None:
        """Guards #535/PR#832: a failed connect attempt that logged a warning must
        not leave stale text behind when a later JSON-RPC-only retry succeeds.

        _connect_acp_session reuses the same agent/<agent>.txt path across retry
        attempts, so start() must clear any stale log from a prior attempt.
        """
        agent_log = tmp_path / "agent" / "gemini.txt"

        # Attempt 0: agent emits a non-protocol warning (captured to the log),
        # then the connection "fails" (the caller discards the transport).
        first_process = AsyncMock()
        first_process.readline = AsyncMock(
            side_effect=[
                b"WARNING: provider hiccup\n",
                b'{"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n',
            ]
        )
        first = ContainerTransport(
            container_process=first_process,
            command="agent acp",
            agent_log_path=agent_log,
        )
        await first.start()
        await asyncio.wait_for(first.receive(), timeout=5)
        await first.close()
        assert "provider hiccup" in agent_log.read_text()

        # Attempt 1: a fresh transport on the SAME path, JSON-RPC only (no
        # non-protocol output). start() must wipe the stale log.
        second_process = AsyncMock()
        second_process.readline = AsyncMock(
            return_value=b'{"jsonrpc": "2.0", "id": 2, "result": {"ok": true}}\n'
        )
        second = ContainerTransport(
            container_process=second_process,
            command="agent acp",
            agent_log_path=agent_log,
        )
        await second.start()
        assert not agent_log.exists()
        try:
            msg = await asyncio.wait_for(second.receive(), timeout=5)
        finally:
            await second.close()
        assert msg == {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}}
        # The successful retry never logged non-protocol output, so no stale text.
        assert not agent_log.exists()

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
        """Guards the 2026-05-22 Daytona/Gemini blocker fix against stuck cancel drain.

        When the idle watchdog fires it cancels the prompt task and drains it.
        A non-cooperative agent — here one that swallows the cancellation and
        blocks on an event that never arrives — must not be able to wedge the
        watchdog: it bounds the drain and raises ``IdleTimeoutError`` anyway.

        Determinism: this asserts the *behaviour* (idle error raised; the stuck
        prompt task abandoned while still pending) rather than a wall-clock upper
        bound, so it cannot be squeezed into a spurious failure when the full
        suite loads the event loop. The outer ``wait_for`` is only a hang guard;
        its timeout is deliberately loose (far above the real ~1.25s runtime), so
        load can never trip it, while a regression to an unbounded drain hangs
        past it and fails.
        """
        from benchflow.acp.runtime import IdleTimeoutError, execute_prompts

        # release is never set while the watchdog runs, so the prompt task stays
        # wedged in its cancellation handler — exactly the stuck-drain scenario.
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

        # Loose hang guard: ~4x the real runtime (1s idle detect + 0.25s bounded
        # drain). Not a tight assertion — it only catches a true hang/regression.
        hang_guard_sec = 5.0

        try:
            with pytest.raises(IdleTimeoutError, match="Agent idle for 1s"):
                await asyncio.wait_for(
                    execute_prompts(
                        client,  # type: ignore[arg-type]
                        session,
                        ["solve"],
                        timeout=30,
                        idle_timeout=1,
                    ),
                    timeout=hang_guard_sec,
                )
            # The watchdog returned while the stuck prompt task is still wedged on
            # release.wait(): proves it bounded the drain instead of waiting for a
            # cancellation that never completes. Load-independent — no clock math.
            assert client.task is not None
            assert not client.task.done()
        finally:
            client.release.set()
            if client.task is not None:
                with pytest.raises(asyncio.CancelledError):
                    await client.task


class TestIdleTimeoutDiagnostics:
    """Guards ENG-149: idle timeouts must carry structured diagnostics."""

    @pytest.mark.asyncio
    async def test_idle_timeout_raises_with_structured_info(self) -> None:
        """Guards ENG-149: IdleTimeoutError carries idle_timeout diagnostic."""
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
        info = exc_info.value.diagnostic.to_dict()
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
        info = exc_info.value.diagnostic.to_dict()
        assert info["n_tool_calls"] == 1

    @pytest.mark.asyncio
    async def test_idle_timeout_last_activity_reports_progress_time(self) -> None:
        """Guards issue #525: last_activity_at is last progress, not timeout fire."""
        from benchflow.acp.runtime import IdleTimeoutError, execute_prompts

        class OneMessageThenHang:
            def __init__(self, session):
                self._session = session
                self._called = False

            async def prompt(self, _prompt: str):
                if not self._called:
                    self._called = True
                    self._session.message_chunks.append("working")
                    await asyncio.sleep(0.1)
                await asyncio.Future()

        session = ACPSession("diag-last-activity")
        client = OneMessageThenHang(session)
        with pytest.raises(IdleTimeoutError) as exc_info:
            await execute_prompts(
                client,  # type: ignore[arg-type]
                session,
                ["solve"],
                timeout=30,
                idle_timeout=1,
            )
        finished_at = datetime.now(UTC)
        info = exc_info.value.diagnostic.to_dict()
        last_activity_at = datetime.fromisoformat(info["last_activity_at"])

        assert info["n_message_chunks"] == 1
        assert (finished_at - last_activity_at).total_seconds() >= 0.75

    @pytest.mark.asyncio
    async def test_in_flight_tool_call_defers_idle_to_wall_clock(self) -> None:
        """A pending/in-progress tool call (agent running a long shell command)
        must NOT trip the idle watchdog: such tools emit no ACP updates until
        they return, so the idle path would otherwise false-kill real work. The
        agent is alive (executing a tool), so it defers to the wall-clock
        backstop. Contrast test_idle_timeout_info_reflects_activity_counts, where
        a *completed* tool that then hangs still idles out."""
        from benchflow.acp.runtime import AgentPromptTimeoutError, execute_prompts

        class PendingToolThenHang:
            def __init__(self, session: ACPSession):
                self._session = session

            async def prompt(self, _prompt: str):
                self._session.handle_update(
                    {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tc_long",
                        "title": "long running build",
                        "kind": "bash",
                    }
                )
                await asyncio.Future()  # hang while the "tool" runs

        session = ACPSession("pending-idle-session")
        # idle_timeout(1) < wall-clock(3): without the in-flight exemption this
        # would raise IdleTimeoutError at ~1s. With it, the pending tool defers
        # idle and the wall-clock backstop fires (AgentPromptTimeoutError) instead.
        with pytest.raises(AgentPromptTimeoutError):
            await execute_prompts(
                PendingToolThenHang(session),  # type: ignore[arg-type]
                session,
                ["solve"],
                timeout=3,
                idle_timeout=1,
            )
        assert session.pending_tool_call_ids() == ["tc_long"]

    @pytest.mark.asyncio
    async def test_wall_clock_timeout_records_terminal_diagnostic(self) -> None:
        """Wall-clock timeouts record runner-owned terminal timeout evidence."""
        from benchflow.acp.runtime import AgentPromptTimeoutError, execute_prompts
        from benchflow.diagnostics import AgentPromptTimeoutDiagnostic

        class SlowClient:
            async def prompt(self, _prompt: str):
                await asyncio.Future()

        session = ACPSession("wall-clock-session")
        # Add continuous activity to prevent idle timeout
        session.tool_calls.append(MagicMock(status=ToolCallStatus.COMPLETED))
        with pytest.raises(AgentPromptTimeoutError) as exc_info:
            await execute_prompts(
                SlowClient(),  # type: ignore[arg-type]
                session,
                ["solve"],
                timeout=2,
                idle_timeout=None,
            )
        assert isinstance(exc_info.value.diagnostic, AgentPromptTimeoutDiagnostic)
        assert exc_info.value.diagnostic.timeout_sec == 2
        assert exc_info.value.diagnostic.pending_tool_call_ids == []
        assert exc_info.value.terminal_trajectory_complete is True
        assert exc_info.value.n_tool_calls == 1
        assert [event["type"] for event in exc_info.value.trajectory] == [
            "user_message",
            "agent_timeout",
        ]
        assert exc_info.value.trajectory[-1]["terminal_trajectory_complete"] is True

    @pytest.mark.asyncio
    async def test_wall_clock_timeout_with_pending_tool_stays_partial(self) -> None:
        """Guards PR #640: pending tool calls cannot become healthy timeouts."""
        from benchflow.acp.runtime import AgentPromptTimeoutError, execute_prompts

        class PendingToolThenHang:
            def __init__(self, session: ACPSession):
                self._session = session

            async def prompt(self, _prompt: str):
                self._session.handle_update(
                    {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "tc_pending",
                        "title": "long running command",
                        "kind": "bash",
                    }
                )
                await asyncio.Future()

        session = ACPSession("pending-wall-clock-session")
        with pytest.raises(AgentPromptTimeoutError) as exc_info:
            await execute_prompts(
                PendingToolThenHang(session),  # type: ignore[arg-type]
                session,
                ["solve"],
                timeout=2,
                idle_timeout=None,
            )

        assert exc_info.value.terminal_trajectory_complete is False
        assert exc_info.value.diagnostic.pending_tool_call_ids == ["tc_pending"]
        assert [event["type"] for event in exc_info.value.trajectory] == [
            "user_message",
            "tool_call",
            "agent_timeout",
        ]
        assert exc_info.value.trajectory[1]["status"] == ToolCallStatus.PENDING.value


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
        mock_acp.set_config_option = AsyncMock()
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
    async def test_openhands_direct_execution_patches_before_privilege_drop(
        self, tmp_path
    ):
        """Guards the PR #921 follow-up for OpenHands tasks rooted outside /root."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = AsyncMock()
        mock_env.exec.return_value = MagicMock(return_code=0, stdout="", stderr="")

        with (
            patch(
                "benchflow.acp.runtime.DockerProcess.from_sandbox_env",
                return_value=MagicMock(),
            ),
            patch(
                "benchflow.acp.runtime.ContainerTransport",
                return_value=MagicMock(),
            ) as mock_transport,
            patch("benchflow.acp.runtime.ACPClient", return_value=mock_acp),
        ):
            await connect_acp(
                env=mock_env,
                agent="openhands",
                agent_launch="openhands acp --always-approve --override-with-envs",
                agent_env={"BENCHFLOW_OPENHANDS_DISABLE_SUBAGENTS": "1"},
                sandbox_user="agent",
                model=None,
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        patch_call = mock_env.exec.await_args_list[0]
        assert "openhands_cli.utils" in patch_call.args[0]
        assert patch_call.kwargs == {"timeout_sec": 30}
        transport_env = mock_transport.call_args.kwargs["env"]
        assert transport_env["BENCHFLOW_OPENHANDS_DISABLE_SUBAGENTS"] == "0"
        transport_command = mock_transport.call_args.kwargs["command"]
        assert "--reuid=agent" in transport_command

    @pytest.mark.asyncio
    async def test_codex_uses_session_advertised_model_id(self, tmp_path):
        """Guards commit 81ff286 against codex-acp rejecting bare set_model IDs."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_acp.session_new.return_value.model_state = {
            "availableModels": [
                {"modelId": "gpt-5.5[low]"},
                {"modelId": "gpt-5.5[medium]"},
                {"modelId": "gpt-5.5[high]"},
            ],
            "currentModelId": "gpt-5[medium]",
        }
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
                agent="codex-acp",
                agent_launch="codex-acp",
                agent_env={},
                sandbox_user=None,
                model="azure-foundry-openai/gpt-5.5",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_awaited_once_with("gpt-5.5[medium]")

    @pytest.mark.asyncio
    async def test_claude_uses_config_options_for_model_and_effort(self, tmp_path):
        """Guards PR #825 repro: latest claude-agent-acp removed session/set_model."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_acp.session_new.return_value.config_options = [
            {"id": "model"},
            {"id": "effort"},
        ]
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
                agent_env={},
                sandbox_user=None,
                model="claude-opus-4-8",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
                reasoning_effort="max",
            )

        mock_acp.set_model.assert_not_awaited()
        assert mock_acp.set_config_option.await_args_list == [
            call("model", "claude-opus-4-8"),
            call("effort", "max"),
        ]

    @pytest.mark.asyncio
    async def test_claude_litellm_env_owns_model_selection(self, tmp_path):
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_acp.session_new.return_value.config_options = [{"id": "model"}]
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
                    "BENCHFLOW_LITELLM_MODEL_ALIAS": "benchflow-bedrock-sonnet",
                    "BENCHFLOW_LITELLM_MODEL_VIA_ENV": "1",
                    "ANTHROPIC_MODEL": "benchflow-bedrock-sonnet",
                },
                sandbox_user=None,
                model="aws-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                rollout_dir=tmp_path,
                environment="docker",
                agent_cwd="/app",
            )

        mock_acp.set_model.assert_not_awaited()
        # LiteLLM VIA_ENV owns model selection -> no ACP set_model or config option.
        mock_acp.set_config_option.assert_not_awaited()

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

    @pytest.mark.asyncio
    async def test_daytona_direct_uses_pty_transport(self, tmp_path):
        """Direct Daytona tasks also use PTY transport, not SSH pipes."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = MagicMock()
        mock_env.exec = AsyncMock(return_value=MagicMock(return_code=1, stdout=""))

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

    @pytest.mark.asyncio
    async def test_daytona_direct_can_opt_into_ssh_transport(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #921 fallback for PTY post-tool controller deadlocks."""
        from benchflow.acp.runtime import connect_acp

        monkeypatch.setenv("BENCHFLOW_DAYTONA_ACP_TRANSPORT", "ssh")
        mock_acp = self._make_mocks()
        mock_env = MagicMock()
        mock_env.exec = AsyncMock(return_value=MagicMock(return_code=1, stdout=""))

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
                agent="openhands",
                agent_launch="openhands acp",
                agent_env={},
                sandbox_user=None,
                model=None,
                rollout_dir=tmp_path,
                environment="daytona",
                agent_cwd="/app",
            )

        mock_ssh.assert_awaited_once_with(mock_env)
        mock_pty.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_daytona_transport_falls_back_to_pty(
        self, tmp_path, monkeypatch
    ):
        """Guards PR #921 against invalid transport config disabling Daytona."""
        from benchflow.acp.runtime import connect_acp

        monkeypatch.setenv("BENCHFLOW_DAYTONA_ACP_TRANSPORT", "invalid")
        mock_acp = self._make_mocks()
        mock_env = MagicMock()
        mock_env.exec = AsyncMock(return_value=MagicMock(return_code=1, stdout=""))

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
                agent="openhands",
                agent_launch="openhands acp",
                agent_env={},
                sandbox_user=None,
                model=None,
                rollout_dir=tmp_path,
                environment="daytona",
                agent_cwd="/app",
            )

        mock_pty.assert_awaited_once_with(mock_env)
        mock_ssh.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_daytona_gemini_uses_ssh_transport(self, tmp_path):
        """Guards the Gemini regression introduced by PR #896's PTY migration."""
        from benchflow.acp.runtime import connect_acp

        mock_acp = self._make_mocks()
        mock_env = MagicMock()
        mock_env.exec = AsyncMock(return_value=MagicMock(return_code=1, stdout=""))

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
                agent="gemini",
                agent_launch="gemini --acp --yolo",
                agent_env={"GEMINI_API_KEY": "test"},
                sandbox_user=None,
                model="gemini-3.1-flash-lite",
                rollout_dir=tmp_path,
                environment="daytona",
                agent_cwd="/app",
            )

        mock_ssh.assert_awaited_once_with(mock_env)
        mock_pty.assert_not_awaited()


class TestSandboxStartupDiagnostics:
    """Guards ENG-147: sandbox startup failures must carry structured diagnostics."""

    def test_sandbox_startup_error_has_info_dict(self) -> None:
        """Guards ENG-147: SandboxStartupError carries a SandboxStartupDiagnostic
        with all required fields for result.json."""
        from benchflow.sandbox.daytona import SandboxStartupError

        err = SandboxStartupError(
            "Sandbox creation failed after retries: timeout of 1200000ms exceeded",
            sandbox_id="e7d8ab0f-47da-40b1-b179-46e1363fe014",
            sandbox_state="creating",
            attempts=3,
            build_timeout_sec=600.0,
        )
        info = err.diagnostic.to_dict()
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
        from benchflow.diagnostics import RolloutDiagnostics, SandboxStartupDiagnostic
        from benchflow.rollout import _build_rollout_result

        diag = SandboxStartupDiagnostic(
            sandbox_id="abc123",
            sandbox_state="error",
            attempts=3,
            build_timeout_sec=600.0,
            raw_message="timeout",
        )
        diagnostics = RolloutDiagnostics()
        diagnostics.set(diag)
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
            diagnostics=diagnostics,
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["sandbox_startup_info"] == diag.to_dict()
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
        """Guards ENG-147: _create_sandbox retries 3 times, not 2.

        The retry contract is declared with the standard ``@retry(...)``
        decorator from tenacity. tenacity attaches the live ``stop`` /
        ``wait`` config to the wrapped function as ``fn.retry``; assert via
        that introspection hook rather than re-implementing scaffolding.
        """
        pytest.importorskip("tenacity")  # ``sandbox-daytona`` extra
        from benchflow.sandbox.daytona import DaytonaSandbox

        stop = DaytonaSandbox._create_sandbox.retry.stop  # type: ignore[attr-defined]
        assert stop.max_attempt_number == 3


class TestTransportErrorDiagnostics:
    """Guards ENG-148 / #504: ACP transport must carry structured diagnostics.

    The diagnostic is now raised at the *source* (``sandbox/process.py``) as
    a :class:`~benchflow.diagnostics.TransportClosedError` carrying a typed
    :class:`~benchflow.diagnostics.TransportClosedDiagnostic` — downstream
    code never regex-parses the error string back into fields.
    """

    @pytest.mark.asyncio
    async def test_live_process_raises_typed_transport_error_with_rc(self) -> None:
        """``LiveProcess.readline`` raises ``TransportClosedError`` with the
        structured ``process_exit_code`` filled in (issue #504)."""
        from benchflow.diagnostics import TransportClosedError
        from benchflow.sandbox.process import LiveProcess

        class _StubProcess:
            def __init__(self) -> None:
                self.returncode = 255
                self.pid = 4242
                self.stdout = self
                self.stderr = self

            async def readline(self) -> bytes:
                return b""

            async def read(self, _n: int) -> bytes:
                return b"Connection to sandbox lost"

        class _LP(LiveProcess):
            async def start(
                self, command, env=None, cwd=None
            ) -> None:  # pragma: no cover - unused
                pass

        lp = _LP()
        lp._process = _StubProcess()  # type: ignore[assignment]
        with pytest.raises(TransportClosedError) as exc_info:
            await lp.readline()
        diag = exc_info.value.diagnostic
        assert diag.process_exit_code == 255
        assert diag.transport_diagnosis == "process_exited"
        assert diag.stderr_snippet is not None
        assert "Connection to sandbox lost" in diag.stderr_snippet

    @pytest.mark.asyncio
    async def test_live_process_raises_typed_transport_error_when_remote_killed(
        self,
    ) -> None:
        """rc=None ⇒ remote session was killed; the diagnostic must carry
        the pid and the ``remote_session_killed`` diagnosis (issue #504)."""
        from benchflow.diagnostics import TransportClosedError
        from benchflow.sandbox.process import LiveProcess

        class _StubProcess:
            def __init__(self) -> None:
                self.returncode = None
                self.pid = 12345
                self.stdout = self
                self.stderr = None

            async def readline(self) -> bytes:
                return b""

        class _LP(LiveProcess):
            async def start(
                self, command, env=None, cwd=None
            ) -> None:  # pragma: no cover - unused
                pass

        lp = _LP()
        lp._process = _StubProcess()  # type: ignore[assignment]
        with pytest.raises(TransportClosedError) as exc_info:
            await lp.readline()
        diag = exc_info.value.diagnostic
        assert diag.process_exit_code is None
        assert diag.process_pid == 12345
        assert diag.transport_diagnosis == "remote_session_killed"

    def test_transport_closed_error_is_a_connection_error(self) -> None:
        """``TransportClosedError`` extends ``ConnectionError`` so existing
        ``except ConnectionError`` paths still catch it (issue #504)."""
        from benchflow.diagnostics import (
            TransportClosedDiagnostic,
            TransportClosedError,
        )

        err = TransportClosedError(
            "test", TransportClosedDiagnostic(raw_message="test")
        )
        assert isinstance(err, ConnectionError)
        assert err.diagnostic.transport_diagnosis == "unknown"

    def test_diagnostic_round_trips_through_result_json(self, tmp_path) -> None:
        """The structured diagnostic survives the dataclass → dict → JSON →
        dict roundtrip used by result.json (issue #503)."""
        from benchflow.diagnostics import (
            DIAGNOSTIC_BY_FIELD,
            RolloutDiagnostics,
            TransportClosedDiagnostic,
        )

        rd = RolloutDiagnostics()
        rd.set(
            TransportClosedDiagnostic(
                raw_message="boom",
                process_exit_code=255,
                process_pid=99,
                transport_diagnosis="process_exited",
                stderr_snippet="oops",
                sandbox_reachable=False,
            )
        )
        fields = rd.to_result_fields()
        assert fields["transport_error_info"] is not None
        roundtripped = fields["transport_error_info"]
        assert roundtripped["process_exit_code"] == 255
        assert roundtripped["transport_diagnosis"] == "process_exited"
        assert roundtripped["sandbox_reachable"] is False
        # The registry can rebuild a typed diagnostic from the dict.
        cls = DIAGNOSTIC_BY_FIELD["transport_error_info"]
        rebuilt = cls(
            **{
                k: v
                for k, v in roundtripped.items()
                if k in TransportClosedDiagnostic._init_fields()
            }
        )
        assert isinstance(rebuilt, TransportClosedDiagnostic)
        assert rebuilt.process_exit_code == 255

    def test_transport_error_info_in_result_json(self, tmp_path) -> None:
        """Guards ENG-148: transport_error_info is written to result.json."""
        from benchflow.diagnostics import RolloutDiagnostics, TransportClosedDiagnostic
        from benchflow.rollout import _build_rollout_result

        diag = TransportClosedDiagnostic(
            process_exit_code=255,
            transport_diagnosis="process_exited",
            sandbox_reachable=False,
        )
        diagnostics = RolloutDiagnostics()
        diagnostics.set(diag)
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
            diagnostics=diagnostics,
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["transport_error_info"] == diag.to_dict()
        assert rj["error_category"] == "pipe_closed"
        assert result.error_category == "pipe_closed"
        assert result.error is not None

    def test_transport_diagnostic_category_overrides_error_string(
        self, tmp_path
    ) -> None:
        """Guards PR #561: typed transport diagnostics own result categorization."""
        from benchflow.diagnostics import RolloutDiagnostics, TransportClosedDiagnostic
        from benchflow.rollout import _build_rollout_result

        diag = TransportClosedDiagnostic(
            raw_message="DaytonaPtyProcess: timeout waiting for agent start marker",
            transport_diagnosis="pty_startup_timeout",
        )
        diagnostics = RolloutDiagnostics()
        diagnostics.set(diag)

        result = _build_rollout_result(
            tmp_path,
            task_name="drone-planning-control",
            rollout_name="drone__abc123",
            agent="openhands",
            agent_name="",
            model="azure-foundry-openai/gpt-5.5",
            n_tool_calls=0,
            prompts=["solve"],
            error="DaytonaPtyProcess: timeout waiting for agent start marker",
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards=None,
            started_at=__import__("datetime").datetime.now(),
            timing={"environment_setup": 10.0},
            diagnostics=diagnostics,
        )

        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["transport_error_info"] == diag.to_dict()
        assert rj["error_category"] == "pipe_closed"
        assert result.error_category == "pipe_closed"

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

    def test_agent_timeout_diagnostic_round_trips_through_result_json(
        self, tmp_path
    ) -> None:
        """Guards PR #640: normal timeouts carry terminal trajectory evidence."""
        from benchflow.diagnostics import (
            AgentPromptTimeoutDiagnostic,
            RolloutDiagnostics,
        )
        from benchflow.rollout import _build_rollout_result

        diagnostics = RolloutDiagnostics()
        diag = AgentPromptTimeoutDiagnostic(
            timeout_sec=900.0,
            n_tool_calls=2,
            terminal_event_recorded=True,
            terminal_trajectory_complete=True,
        )
        diagnostics.set(diag)

        result = _build_rollout_result(
            tmp_path,
            task_name="hello-world",
            rollout_name="hello__timeout",
            agent="openhands",
            agent_name="openhands",
            model="test-model",
            n_tool_calls=2,
            prompts=["solve"],
            error="Agent prompt exceeded wall-clock budget 900s",
            verifier_error=None,
            trajectory=[
                {"type": "user_message", "text": "solve"},
                {
                    "type": "agent_timeout",
                    "reason": "wall_clock_timeout",
                    "timeout_sec": 900.0,
                    "pending_tool_call_ids": [],
                    "terminal_trajectory_complete": True,
                },
            ],
            partial_trajectory=False,
            trajectory_source="acp",
            rewards={"reward": 0.0},
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 900.0},
            diagnostics=diagnostics,
        )

        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["agent_timeout_info"] == diag.to_dict()
        assert rj["trajectory_summary"]["event_type_counts"]["agent_timeout"] == 1
        assert rj["trajectory_summary"]["partial_trajectory"] is False
        assert rj["error_category"] == "timeout"
        assert result.error_category == "timeout"


class TestDiagnosticRegistry:
    """Guards #503: the diagnostic registry is the single source of truth.

    Every diagnostic shape — result.json field name, summary warning,
    check_results invalidation line — flows from the same registry. Adding
    a new diagnostic should require adding ONE class, not coordinated
    edits across rollout/evaluation/check_results.
    """

    def test_every_diagnostic_round_trips_through_result_fields(self) -> None:
        """Every registered Diagnostic serializes under its own field name."""
        from benchflow.diagnostics import (
            DIAGNOSTIC_BY_FIELD,
            DIAGNOSTIC_REGISTRY,
            RolloutDiagnostics,
        )

        # Field name → class lookup must mirror the registry.
        assert set(DIAGNOSTIC_BY_FIELD.keys()) == {d.field for d in DIAGNOSTIC_REGISTRY}
        # An empty collector renders every field as None.
        empty = RolloutDiagnostics().to_result_fields()
        for diag_cls in DIAGNOSTIC_REGISTRY:
            assert empty[diag_cls.field] is None

    def test_diagnostic_reason_constants_share_scoring_source(self) -> None:
        """Guards the fix from PR #858 against diagnostic reasons drifting off the shared scoring source."""
        from typing import get_args, get_type_hints

        from benchflow._utils.scoring import IDLE_TIMEOUT
        from benchflow.diagnostics import (
            DIAGNOSTIC_REASON_IDLE_TIMEOUT,
            DIAGNOSTIC_REASON_SANDBOX_STARTUP_FAILED,
            DIAGNOSTIC_REASON_TRANSPORT_CLOSED,
            DIAGNOSTIC_REASON_WALL_CLOCK_TIMEOUT,
            AgentPromptTimeoutDiagnostic,
            DiagnosticReason,
            IdleTimeoutDiagnostic,
            SandboxStartupDiagnostic,
            TransportClosedDiagnostic,
        )

        assert IDLE_TIMEOUT == DIAGNOSTIC_REASON_IDLE_TIMEOUT
        assert set(get_args(DiagnosticReason)) == {
            DIAGNOSTIC_REASON_IDLE_TIMEOUT,
            DIAGNOSTIC_REASON_WALL_CLOCK_TIMEOUT,
            DIAGNOSTIC_REASON_SANDBOX_STARTUP_FAILED,
            DIAGNOSTIC_REASON_TRANSPORT_CLOSED,
        }

        reason_hints = {
            diag_cls.__name__: get_type_hints(diag_cls)["reason"]
            for diag_cls in (
                IdleTimeoutDiagnostic,
                AgentPromptTimeoutDiagnostic,
                SandboxStartupDiagnostic,
                TransportClosedDiagnostic,
            )
        }
        assert get_args(reason_hints["IdleTimeoutDiagnostic"]) == (
            DIAGNOSTIC_REASON_IDLE_TIMEOUT,
        )
        assert get_args(reason_hints["AgentPromptTimeoutDiagnostic"]) == (
            DIAGNOSTIC_REASON_WALL_CLOCK_TIMEOUT,
        )
        assert get_args(reason_hints["SandboxStartupDiagnostic"]) == (
            DIAGNOSTIC_REASON_SANDBOX_STARTUP_FAILED,
        )
        assert get_args(reason_hints["TransportClosedDiagnostic"]) == (
            DIAGNOSTIC_REASON_TRANSPORT_CLOSED,
        )

    def test_summary_warning_uses_registry_metadata(self) -> None:
        """Summary warning text comes from the registry's
        ``summary_description``, not a per-call f-string in evaluation.py."""
        from benchflow.diagnostics import (
            IdleTimeoutDiagnostic,
            TransportClosedDiagnostic,
            summary_warning,
        )

        msg = summary_warning(IdleTimeoutDiagnostic, count=3, total=10)
        assert "3 tasks (30%) hit idle timeout" in msg
        assert "idle_timeout_info" in msg
        msg = summary_warning(TransportClosedDiagnostic, count=2, total=10)
        assert "lost transport" in msg
        assert "transport_error_info" in msg

    def test_format_issue_for_field_dispatches_via_registry(self) -> None:
        """check_results renders per-task invalidation lines through the
        registry — no per-diagnostic format string lives in check_results.py."""
        from benchflow.diagnostics import format_issue_for_field

        line = format_issue_for_field(
            "idle_timeout_info",
            "task-1",
            {
                "idle_duration_sec": 602,
                "n_tool_calls": 3,
                "wall_clock_elapsed_sec": 605,
            },
        )
        assert "task-1: idle timeout after 602s idle (3 tool calls" in line

        line = format_issue_for_field(
            "transport_error_info",
            "task-2",
            {
                "process_exit_code": 255,
                "transport_diagnosis": "process_exited",
                "sandbox_reachable": False,
            },
        )
        assert "task-2: transport closed (rc=255, diagnosis=process_exited" in line

    def test_sandbox_startup_error_diagnostic_view_is_consistent(self) -> None:
        """``SandboxStartupError.diagnostic.to_dict()`` reflects the same
        dict the registry serializer produces — one schema, one source."""
        from benchflow.diagnostics import RolloutDiagnostics
        from benchflow.sandbox.protocol import SandboxStartupError

        err = SandboxStartupError(
            "boom",
            sandbox_id="sb-1",
            sandbox_state="error",
            attempts=2,
            build_timeout_sec=600.0,
        )
        rd = RolloutDiagnostics()
        rd.set(err.diagnostic)
        fields = rd.to_result_fields()
        assert fields["sandbox_startup_info"] == err.diagnostic.to_dict()


class TestVerifierDepInstallDiagnostics:
    """Guards ENG-151: verifier dep install failures must be classified distinctly."""

    def test_classify_verifier_dep_install_error(self) -> None:
        """Guards ENG-151: classify_verifier_error detects dependency install
        patterns and returns VERIFIER_DEP_INSTALL."""
        from benchflow._utils.scoring import (
            VERIFIER_DEP_INSTALL,
            classify_verifier_error,
        )

        assert (
            classify_verifier_error(
                "verifier crashed: verifier exited with rc=1; dependency install failed"
            )
            == VERIFIER_DEP_INSTALL
        )
        assert (
            classify_verifier_error(
                "verifier crashed: No solution found when resolving dependencies"
            )
            == VERIFIER_DEP_INSTALL
        )
        assert (
            classify_verifier_error(
                "verifier crashed: Could not find a version that satisfies "
                "the requirement torch==2.1.2+cpu"
            )
            == VERIFIER_DEP_INSTALL
        )

    def test_classify_verifier_dep_install_not_false_positive(self) -> None:
        """Guards ENG-151: normal verifier crashes are NOT classified as
        dep install failures."""
        from benchflow._utils.scoring import (
            VERIFIER_DEP_INSTALL,
            classify_verifier_error,
        )

        assert (
            classify_verifier_error("verifier crashed: assert False")
            != VERIFIER_DEP_INSTALL
        )
        assert (
            classify_verifier_error("verifier timed out after 900s")
            != VERIFIER_DEP_INSTALL
        )
        assert classify_verifier_error(None) is None

    def test_verifier_error_category_in_result_json(self, tmp_path: Path) -> None:
        """Guards ENG-151: result.json includes verifier_error_category field."""
        from benchflow.rollout import _build_rollout_result

        result = _build_rollout_result(
            tmp_path,
            task_name="simpo-code-reproduction",
            rollout_name="simpo__abc123",
            agent="gemini",
            agent_name="gemini-cli",
            model="gemini-2.0-flash-lite",
            n_tool_calls=0,
            prompts=["solve"],
            error=None,
            verifier_error=(
                "verifier crashed: verifier exited with rc=1; dependency install failed"
            ),
            trajectory=[],
            partial_trajectory=False,
            rewards=None,
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 0.0},
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["verifier_error_category"] == "verifier_dep_install"
        assert result.verifier_error is not None

    def test_verifier_error_category_null_when_no_verifier_error(
        self, tmp_path: Path
    ) -> None:
        """Guards ENG-151: verifier_error_category is null for successful runs."""
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
        assert rj["verifier_error_category"] is None
        assert result.rewards == {"reward": 1.0}


class TestVerifierTimeoutDiagnostics:
    """Guards ENG-152: verifier timeouts must produce structured diagnostics."""

    def test_classify_verifier_timeout(self) -> None:
        """Guards ENG-152: classify_verifier_error returns VERIFIER_TIMEOUT
        for timeout messages."""
        from benchflow._utils.scoring import (
            VERIFIER_TIMEOUT,
            classify_verifier_error,
        )

        assert (
            classify_verifier_error("verifier timed out after 240s") == VERIFIER_TIMEOUT
        )
        assert (
            classify_verifier_error("verifier timed out after 600.0s")
            == VERIFIER_TIMEOUT
        )

    def test_classify_verifier_timeout_not_false_positive(self) -> None:
        """Guards ENG-152: non-timeout verifier errors are NOT classified as
        timeout."""
        from benchflow._utils.scoring import (
            VERIFIER_TIMEOUT,
            classify_verifier_error,
        )

        assert (
            classify_verifier_error("verifier crashed: assert False")
            != VERIFIER_TIMEOUT
        )
        assert (
            classify_verifier_error("verifier crashed: dependency install failed")
            != VERIFIER_TIMEOUT
        )

    def test_verifier_timeout_info_in_result_json(self, tmp_path: Path) -> None:
        """Guards ENG-152: result.json includes verifier_timeout_info when
        verifier times out."""
        from benchflow.diagnostics import RolloutDiagnostics, VerifierTimeoutDiagnostic
        from benchflow.rollout import _build_rollout_result

        diagnostics = RolloutDiagnostics()
        diagnostics.set(
            VerifierTimeoutDiagnostic(
                timeout_budget_sec=240.0,
                elapsed_sec=240.1,
                task_name="quantum-numerical-simulation",
            )
        )
        _build_rollout_result(
            tmp_path,
            task_name="quantum-numerical-simulation",
            rollout_name="quantum__abc123",
            agent="gemini",
            agent_name="gemini-cli",
            model="gemini-2.0-flash-lite",
            n_tool_calls=0,
            prompts=["solve"],
            error=None,
            verifier_error="verifier timed out after 240s",
            trajectory=[],
            partial_trajectory=False,
            rewards=None,
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 0.0},
            diagnostics=diagnostics,
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["verifier_error_category"] == "verifier_timeout"
        vti = rj["verifier_timeout_info"]
        assert vti is not None
        assert vti["timeout_budget_sec"] == 240.0
        assert vti["elapsed_sec"] == 240.1
        assert vti["task_name"] == "quantum-numerical-simulation"

    def test_verifier_timeout_info_null_when_no_timeout(self, tmp_path: Path) -> None:
        """Guards ENG-152: verifier_timeout_info is null for non-timeout runs."""
        from benchflow.rollout import _build_rollout_result

        _build_rollout_result(
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
        assert rj["verifier_timeout_info"] is None

    def test_verifier_timeout_info_null_for_crash(self, tmp_path: Path) -> None:
        """Guards ENG-152: verifier_timeout_info is null when verifier crashes
        (not a timeout)."""
        from benchflow.rollout import _build_rollout_result

        _build_rollout_result(
            tmp_path,
            task_name="some-task",
            rollout_name="some__abc",
            agent="gemini",
            agent_name="gemini-cli",
            model="gemini-2.0-flash-lite",
            n_tool_calls=0,
            prompts=["solve"],
            error=None,
            verifier_error="verifier crashed: assert False",
            trajectory=[],
            partial_trajectory=False,
            rewards=None,
            started_at=__import__("datetime").datetime.now(),
            timing={"agent": 0.0},
        )
        rj = __import__("json").loads((tmp_path / "result.json").read_text())
        assert rj["verifier_error_category"] == "verifier_failure"
        assert rj["verifier_timeout_info"] is None
