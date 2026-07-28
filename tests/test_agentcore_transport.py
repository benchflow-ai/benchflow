"""AgentCore shell transport: liveness and channel separation.

The transport carries ACP JSON-RPC. Two failure modes matter more than
throughput: a dead reader that nobody notices (the run hangs until a 15-minute
read timeout instead of reporting an infrastructure disconnect), and side
channels leaking into the protocol stream (a diagnostic line that the JSON-RPC
parser sees as traffic).
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.diagnostics import TransportClosedError
from benchflow.sandbox.agentcore import AgentCoreSandbox
from benchflow.sandbox.process.agentcore import AgentCoreProcess
from benchflow.task.config import SandboxConfig


def _frame(payload: bytes, channel: str = "STDOUT"):
    return SimpleNamespace(
        payload=payload, channel=SimpleNamespace(name=channel), raw_channel_byte=1
    )


class _FakeShell:
    """Async-iterable stand-in for the SDK's ShellSession."""

    def __init__(self, frames, *, error: Exception | None = None):
        self._frames = list(frames)
        self._error = error
        self.sent: list[str] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._frames:
            return self._frames.pop(0)
        if self._error:
            raise self._error
        raise StopAsyncIteration

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        return None


class _MarkerShell:
    """Emit the nonce from the first send, then stay open or cleanly EOF."""

    def __init__(self, *, eof_after_marker: bool, fail_launch: bool = False):
        self.eof_after_marker = eof_after_marker
        self.fail_launch = fail_launch
        self.sent: list[str] = []
        self._marker_ready = asyncio.Event()
        self._marker_emitted = False
        self._closed = asyncio.Event()
        self.shell_id = "fake-shell"

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._marker_emitted:
            await self._marker_ready.wait()
            self._marker_emitted = True
            marker = re.search(r"(__BENCHFLOW_ACP_[0-9a-f]+__)", self.sent[0])
            assert marker is not None
            return _frame((marker.group(1) + "\n").encode())
        if self.eof_after_marker:
            raise StopAsyncIteration
        await self._closed.wait()
        raise StopAsyncIteration

    async def send(self, data):
        if self._marker_emitted and self.fail_launch:
            raise ConnectionResetError("launch socket closed")
        self.sent.append(data)
        self._marker_ready.set()

    async def close(self):
        self._closed.set()


class _ShellContext:
    def __init__(self, shell):
        self.shell = shell

    async def __aenter__(self):
        return self.shell


def _runtime_client(shell):
    client = MagicMock()
    client.open_shell.return_value = _ShellContext(shell)
    return client


def _runtime_sdk(shell):
    """Install the optional AgentCore SDK modules for one transport test."""
    package = ModuleType("bedrock_agentcore")
    runtime = ModuleType("bedrock_agentcore.runtime")
    runtime.AgentCoreRuntimeClient = MagicMock(return_value=_runtime_client(shell))
    package.runtime = runtime
    return patch.dict(
        sys.modules,
        {
            "bedrock_agentcore": package,
            "bedrock_agentcore.runtime": runtime,
        },
    )


def _process(shell):
    proc = AgentCoreProcess(MagicMock(), "arn:rt", "s" * 40, "us-west-2")
    proc._shell = shell
    return proc


class TestReaderLiveness:
    @pytest.mark.asyncio
    async def test_reader_error_wakes_readline_immediately(self):
        """A disconnect must not wait out the 900s read timeout."""
        proc = _process(_FakeShell([], error=ConnectionResetError("socket died")))
        proc._reader_task = asyncio.create_task(proc._drain_frames())

        with pytest.raises(TransportClosedError) as excinfo:
            await asyncio.wait_for(proc.readline(), timeout=2)

        assert "transport failed" in str(excinfo.value)
        assert excinfo.value.diagnostic.transport_diagnosis == "remote_session_killed"

    @pytest.mark.asyncio
    async def test_clean_eof_also_wakes_readline(self):
        """A clean iterator end records no failure but is still fatal."""
        proc = _process(_FakeShell([]))
        proc._reader_task = asyncio.create_task(proc._drain_frames())

        with pytest.raises(TransportClosedError) as excinfo:
            await asyncio.wait_for(proc.readline(), timeout=2)

        assert "closed by the remote session" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_buffered_output_is_delivered_before_the_end_sentinel(self):
        """Ending the reader must not discard already-framed lines."""
        proc = _process(_FakeShell([_frame(b'{"jsonrpc":"2.0"}\n')]))
        proc._reader_task = asyncio.create_task(proc._drain_frames())

        line = await asyncio.wait_for(proc.readline(), timeout=2)

        assert line == b'{"jsonrpc":"2.0"}\n'

    @pytest.mark.asyncio
    async def test_is_running_is_false_once_the_reader_ends(self):
        """Liveness that ignores the reader reports a dead pipe as healthy."""
        proc = _process(_FakeShell([]))
        assert proc.is_running is True

        await proc._drain_frames()

        assert proc.is_running is False


class TestChannelSeparation:
    @pytest.mark.asyncio
    async def test_stderr_never_enters_the_acp_stream(self):
        """A diagnostic must not be readable as JSON-RPC traffic."""
        proc = _process(
            _FakeShell(
                [
                    _frame(b"traceback: something failed\n", channel="STDERR"),
                    _frame(b'{"jsonrpc":"2.0","id":1}\n'),
                ]
            )
        )
        proc._reader_task = asyncio.create_task(proc._drain_frames())

        line = await asyncio.wait_for(proc.readline(), timeout=2)

        assert line == b'{"jsonrpc":"2.0","id":1}\n'

    @pytest.mark.asyncio
    async def test_frames_without_channel_info_are_treated_as_stdout(self):
        """An SDK emitting one undifferentiated stream must not be muted.

        Requiring a recognized STDOUT would turn a naming mismatch into total
        transport failure rather than a cosmetic one.
        """
        untyped = SimpleNamespace(payload=b"hello\n", channel=None)
        proc = _process(_FakeShell([untyped]))
        proc._reader_task = asyncio.create_task(proc._drain_frames())

        assert await asyncio.wait_for(proc.readline(), timeout=2) == b"hello\n"

    @pytest.mark.asyncio
    async def test_a_typed_unknown_channel_never_reaches_the_acp_stream(self):
        """A typed channel this code does not know is still a side channel.

        Admitting it would put non-protocol bytes into JSON-RPC input, which is
        how a diagnostic gets parsed as — or impersonates — protocol traffic.
        """
        proc = _process(
            _FakeShell(
                [
                    _frame(b"telemetry blob\n", channel="METRICS"),
                    _frame(b'{"jsonrpc":"2.0","id":7}\n'),
                ]
            )
        )
        proc._reader_task = asyncio.create_task(proc._drain_frames())

        assert (
            await asyncio.wait_for(proc.readline(), timeout=2)
            == b'{"jsonrpc":"2.0","id":7}\n'
        )

    @pytest.mark.asyncio
    async def test_shell_death_during_startup_does_not_hang(self):
        """Startup drain must not swallow the end sentinel.

        Discarding it strands the next readline for the full 900s timeout even
        though the transport is already known to be gone.
        """
        proc = _process(_FakeShell([_frame(b"noise\n")]))
        proc._reader_task = asyncio.create_task(proc._drain_frames())
        await asyncio.sleep(0)
        while not proc._reader_done:
            await asyncio.sleep(0.01)
        proc._clear_buffered_output()

        with pytest.raises(TransportClosedError):
            await asyncio.wait_for(proc.readline(), timeout=2)


class TestEnvHandling:
    @pytest.mark.asyncio
    async def test_secrets_are_staged_not_typed_into_the_terminal(self):
        """The PTY echoes input, so a typed secret lands in the agent log."""
        proc = _process(_FakeShell([]))
        proc._sandbox.write_text_file = AsyncMock(return_value=True)

        path = await proc._write_env_file({"API_KEY": "hunter2"})

        assert path.startswith("/tmp/")
        body = proc._sandbox.write_text_file.call_args.args[1]
        assert "hunter2" in body
        assert proc._sandbox.write_text_file.call_args.kwargs["mode"] == "600"

    @pytest.mark.asyncio
    async def test_invalid_env_names_are_refused_before_staging(self):
        """Guards PR #937: invalid exports fail and can strand secret files."""
        proc = _process(_FakeShell([]))
        proc._sandbox.write_text_file = AsyncMock(return_value=True)

        with pytest.raises(ValueError, match=r"BAD\.KEY"):
            await proc._write_env_file({"BAD.KEY": "secret"})

        proc._sandbox.write_text_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_env_cleanup_is_armed_before_cwd_and_removed_before_exec(self):
        """Guards PR #937: every pre-launch failure must delete staged secrets."""
        shell = _MarkerShell(eof_after_marker=False)
        proc = _process(shell)
        proc._sandbox.write_text_file = AsyncMock(return_value=True)

        with _runtime_sdk(shell):
            await proc.start("agent --serve", env={"API_KEY": "hunter2"}, cwd="/work")

        launch = shell.sent[1]
        assert launch.startswith("( ") and launch.endswith(" )\n")
        assert launch.index("trap 'rm -f") < launch.index("cd /work")
        source = launch.index(". /tmp/.benchflow_agent_env_")
        remove_after_source = launch.index("rm -f", source)
        assert source < remove_after_source < launch.index("trap - EXIT")
        assert "hunter2" not in launch
        await proc.close()

    @pytest.mark.asyncio
    async def test_echoed_marker_command_does_not_complete_startup(self):
        """Guards PR #937: PTY command echo is not the nonce response."""
        proc = _process(_FakeShell([]))
        marker = "__BENCHFLOW_ACP_deadbeef__"
        await proc._line_buffer.put(
            f"root@host:/# stty raw -echo; echo '{marker}'\n".encode()
        )
        await proc._line_buffer.put(f"\x1b[?2004l{marker}\r\n".encode())

        await proc._await_marker(marker)

        assert proc._line_buffer.empty()

    @pytest.mark.asyncio
    async def test_terminal_only_line_after_marker_never_reaches_acp(self):
        """Guards PR #937: bracketed-paste state is not JSON-RPC input."""
        proc = _process(
            _FakeShell(
                [
                    _frame(b"\x1b[?2004l\r\n"),
                    _frame(b'{"jsonrpc":"2.0","id":7}\r\n'),
                ]
            )
        )
        proc._reader_task = asyncio.create_task(proc._drain_frames())

        assert await proc.readline() == b'{"jsonrpc":"2.0","id":7}\n'

    @pytest.mark.asyncio
    async def test_failed_staging_best_effort_deletes_the_partial_file(self):
        """Guards PR #937 when write succeeds but chmod reports failure."""
        proc = _process(_FakeShell([]))
        proc._sandbox.write_text_file = AsyncMock(return_value=False)
        proc._sandbox.exec = AsyncMock()

        with pytest.raises(RuntimeError, match="Failed to stage"):
            await proc._write_env_file({"API_KEY": "hunter2"})

        cleanup = proc._sandbox.exec.call_args.args[0]
        assert cleanup.startswith("rm -f /tmp/.benchflow_agent_env_")
        assert "hunter2" not in cleanup

    @pytest.mark.asyncio
    async def test_failed_launch_deletes_the_staged_env_file(self):
        """Guards PR #937 when the shell dies after env staging."""
        shell = _MarkerShell(eof_after_marker=False, fail_launch=True)
        proc = _process(shell)
        proc._sandbox.write_text_file = AsyncMock(return_value=True)
        proc._sandbox.exec = AsyncMock()

        with (
            _runtime_sdk(shell),
            pytest.raises(ConnectionResetError, match="launch socket"),
        ):
            await proc.start("agent --serve", env={"API_KEY": "hunter2"})

        cleanup = proc._sandbox.exec.call_args.args[0]
        assert cleanup.startswith("rm -f /tmp/.benchflow_agent_env_")
        assert "hunter2" not in cleanup


class TestAdversarialChannelNames:
    """Guards PR #937: typed channels are matched exactly, not by substring."""

    @pytest.mark.parametrize("name", ["NOT_STDOUT", "STDOUT_METADATA", "METRICS"])
    def test_colliding_typed_names_are_not_stdout(self, name):
        """Guards PR #937: substring collisions must not become ACP input."""
        assert AgentCoreProcess._is_stdout(_frame(b"x\n", channel=name)) is False

    @pytest.mark.parametrize("name", ["STDOUT", "ShellChannel.STDOUT", " stdout "])
    def test_genuine_stdout_is_accepted(self, name):
        """Guards PR #937: genuine enum/name variants must still deliver."""
        assert AgentCoreProcess._is_stdout(_frame(b"x\n", channel=name)) is True

    @pytest.mark.asyncio
    async def test_a_colliding_name_never_reaches_readline(self):
        """Guards PR #937 end to end against typed channel impersonation."""
        proc = _process(
            _FakeShell(
                [
                    _frame(b"impersonating\n", channel="NOT_STDOUT"),
                    _frame(b'{"jsonrpc":"2.0","id":9}\n'),
                ]
            )
        )
        proc._reader_task = asyncio.create_task(proc._drain_frames())

        assert (
            await asyncio.wait_for(proc.readline(), timeout=2)
            == b'{"jsonrpc":"2.0","id":9}\n'
        )


class TestHalfCloseGuard:
    """Guards PR #937: no writes to a transport whose read side is gone."""

    @pytest.mark.asyncio
    async def test_writeline_refuses_after_clean_eof(self):
        """Guards PR #937 because ContainerTransport does not pre-check liveness.

        Without this guard an ACP request is handed to a dead socket and its
        reply can never arrive, so the run stalls instead of failing.
        """
        shell = _FakeShell([])
        proc = _process(shell)
        await proc._drain_frames()

        with pytest.raises(TransportClosedError, match="not running") as excinfo:
            await proc.writeline("lost-message")

        assert shell.sent == []
        assert excinfo.value.diagnostic.transport_diagnosis == "remote_session_killed"

    @pytest.mark.asyncio
    async def test_writeline_refuses_after_reader_error(self):
        """Guards PR #937 when the reader ends with an exception."""
        shell = _FakeShell([], error=ConnectionResetError("socket died"))
        proc = _process(shell)
        await proc._drain_frames()

        with pytest.raises(TransportClosedError, match="not running"):
            await proc.writeline("lost-message")

        assert shell.sent == []

    @pytest.mark.asyncio
    async def test_writeline_works_while_the_reader_is_alive(self):
        """Guards PR #937: the half-close fix must preserve normal writes."""
        shell = _FakeShell([_frame(b"hi\n")])
        proc = _process(shell)

        await proc.writeline("ping")

        assert shell.sent == ["ping\n"]

    @pytest.mark.asyncio
    async def test_marker_then_eof_refuses_the_agent_launch(self):
        """Guards PR #937: startup must use the half-close checked send path."""
        shell = _MarkerShell(eof_after_marker=True)
        proc = _process(shell)

        with (
            _runtime_sdk(shell),
            pytest.raises(TransportClosedError) as excinfo,
        ):
            await proc.start("agent --serve")

        assert len(shell.sent) == 1
        assert excinfo.value.diagnostic.transport_diagnosis == "remote_session_killed"


@pytest.mark.skipif(
    os.environ.get("BENCHFLOW_AGENTCORE_LIVE_TEST") != "1",
    reason="live AWS test; set BENCHFLOW_AGENTCORE_LIVE_TEST=1 to run",
)
@pytest.mark.asyncio
async def test_real_agentcore_shell_transport(tmp_path):
    """Guards PR #937 end to end: real WebSocket stdin/stdout/env/cwd/cleanup."""
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN echo baked > /baked.txt\n"
    )
    env = AgentCoreSandbox(
        environment_dir=tmp_path,
        environment_name="live-canary",
        session_id="live-transport",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    await env.start(force_build=False)
    process = await env.live_process()
    program = (
        "import os,sys;"
        "print('READY:'+os.getcwd()+':'+os.environ['BF_LIVE_SECRET'], flush=True);"
        "[(print('ECHO:'+line.rstrip('\\\\n'), flush=True)) for line in sys.stdin]"
    )
    try:
        await process.start(
            f"python3 -u -c {shlex.quote(program)}",
            env={"BF_LIVE_SECRET": "value with spaces"},
            cwd="/tmp",
        )
        assert await process.readline() == b"READY:/tmp:value with spaces\n"
        await process.writeline("hello-agentcore")
        assert await process.readline() == b"ECHO:hello-agentcore\n"

        staged = await env.exec(
            "find /tmp -maxdepth 1 -name '.benchflow_agent_env_*' -print"
        )
        assert staged.return_code == 0
        assert not (staged.stdout or "").strip()
    finally:
        await process.close()
        await env.stop(delete=True)
