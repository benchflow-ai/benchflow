"""AgentCore shell transport: liveness and channel separation.

The transport carries ACP JSON-RPC. Two failure modes matter more than
throughput: a dead reader that nobody notices (the run hangs until a 15-minute
read timeout instead of reporting an infrastructure disconnect), and side
channels leaking into the protocol stream (a diagnostic line that the JSON-RPC
parser sees as traffic).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.diagnostics import TransportClosedError
from benchflow.sandbox.process.agentcore import AgentCoreProcess


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
