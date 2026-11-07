"""Daytona command frames, delivery uncertainty and lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.sandbox._base import KIT_PROBE_ROUTE, ExecResult
from benchflow.sandbox._exec_channel import (
    HELPER_SOURCE,
    ChannelLost,
    ChannelUnavailable,
    ChannelUnconfirmed,
)
from benchflow.sandbox.daytona import _ExecCounts
from benchflow.sandbox.daytona_channel import (
    _PTY_PRELUDE,
    CHUNK_SIZE,
    DaytonaChannel,
    DaytonaChannels,
    eligible,
)

pytestmark = pytest.mark.timeout(30)


def output(kind, ident, payload=b""):
    return kind + struct.pack(">II", ident, len(payload)) + payload


def channel():
    made = DaytonaChannel(SimpleNamespace(), _ExecCounts())
    made.pty = SimpleNamespace(send_input=AsyncMock())
    made.ready.set_result(None)
    made.synchronized = True
    return made


async def test_fragmented_binary_frames_keep_streams_and_status_separate():
    """Raw frames preserve NUL, CR, LF and all byte values."""
    made = channel()
    task = asyncio.create_task(made.run("true", "/bin/sh", 10))
    await asyncio.sleep(0.01)
    binary = bytes(range(256))
    data = (
        output(b"S", 1)
        + output(b"O", 1, binary)
        + output(b"E", 1, b"err\r\n")
        + output(b"X", 1, b"7")
    )
    for n in range(0, len(data), 3):
        await made.on_data(data[n : n + 3])
    result = await task
    assert result.return_code == 7
    assert result.stream(1) == binary
    assert result.stream(2) == b"err\r\n"


async def test_chunked_command_is_one_frame_without_interleaving():
    """Large concurrent requests use bounded WebSocket messages."""
    made = channel()
    large = "x" * (CHUNK_SIZE * 6)
    tasks = [asyncio.create_task(made.run(c, "/bin/sh", 10)) for c in (large, "second")]
    await asyncio.sleep(0.05)
    writes = [call.args[0] for call in made.pty.send_input.call_args_list]
    assert all(len(b) <= CHUNK_SIZE for b in writes)
    data = b"".join(writes)
    heads = []
    while data:
        size = struct.unpack(">I", data[:4])[0]
        heads.append(json.loads(data[4 : 4 + size]))
        data = data[4 + size :]
    assert [h["a"][-1] for h in heads] == [large, "second"]
    for i in (2, 1):
        await made.on_data(output(b"X", i, str(i).encode()))
    assert [r.return_code for r in await asyncio.gather(*tasks)] == [1, 2]


async def test_death_before_any_send_allows_fallback():
    """A command rejected before writing is proven unstarted."""
    made = channel()
    made.die("before request")
    with pytest.raises(ChannelUnavailable):
        await made.run("side-effect", "/bin/sh", 10)
    made.pty.send_input.assert_not_awaited()


@pytest.mark.parametrize("started", [False, True])
async def test_death_after_delivery_never_reports_unstarted(started):
    """A lost start acknowledgement cannot authorize replay."""
    made = channel()
    task = asyncio.create_task(made.run("side-effect", "/bin/sh", 10))
    await asyncio.sleep(0.01)
    if started:
        await made.on_data(output(b"S", 1))
    made.die("lost response")
    with pytest.raises(ChannelLost) as error:
        await task
    assert isinstance(error.value, ChannelUnconfirmed) is (not started)


async def test_failed_send_is_uncertain_even_if_the_sdk_raised():
    """An SDK send may deliver all bytes before reporting failure."""
    made = channel()
    made.pty.send_input.side_effect = ConnectionResetError("after acceptance")
    with pytest.raises(ChannelUnconfirmed):
        await made.run("side-effect", "/bin/sh", 10)
    assert not made.alive


async def test_timeout_during_partial_send_invalidates_framing():
    """Another request cannot complete a timed-out partial frame."""
    made = channel()

    async def blocked(data):
        await asyncio.Event().wait()

    made.pty.send_input.side_effect = blocked
    with pytest.raises(ChannelUnconfirmed):
        await made.run("x" * CHUNK_SIZE * 2, "/bin/sh", 0.01)
    with pytest.raises(ChannelUnavailable):
        await made.run("next", "/bin/sh", 10)


async def test_completed_result_survives_a_subsequent_channel_death():
    """A whole response remains authoritative after disconnect."""
    made = channel()
    task = asyncio.create_task(made.run("true", "/bin/sh", 10))
    await asyncio.sleep(0.01)
    await made.on_data(output(b"X", 1, b"0"))
    made.die("after result")
    assert (await task).return_code == 0


async def test_startup_echo_is_discarded_until_the_private_marker():
    """Terminal prompts cannot be parsed as response frames."""
    made = DaytonaChannel(SimpleNamespace(), _ExecCounts())
    stream = b"prompt\r\n" + made.marker + output(b"R", 0, b"123")
    for b in stream:
        await made.on_data(bytes([b]))
    assert made.alive and made.pid == 123


@pytest.mark.parametrize(
    "data", [b"Z" + b"\0" * 8, b"O" + struct.pack(">II", 1, 1000000)]
)
async def test_malformed_frame_fails_pending_commands(data):
    """Corrupt or oversized responses fail closed without replay."""
    made = channel()
    task = asyncio.create_task(made.run("true", "/bin/sh", 10))
    await asyncio.sleep(0.01)
    await made.on_data(data)
    try:
        assert made.dead
    finally:
        made.die("test finished")
    with pytest.raises(ChannelLost):
        await task


@pytest.mark.parametrize(
    "change",
    [
        dict(kit=False),
        dict(root=False),
        dict(pin=None),
        dict(task_root=True),
        dict(own_vm=True),
    ],
)
def test_only_protected_root_plumbing_is_eligible(change):
    """Task, agent, DinD and unprotected routes keep session execs."""
    box = SimpleNamespace(
        exec_channel=True,
        _sandbox=SimpleNamespace(
            process=SimpleNamespace(create_pty_session=AsyncMock())
        ),
    )
    assert eligible(box, KIT_PROBE_ROUTE)
    assert not eligible(box, replace(KIT_PROBE_ROUTE, **change))


async def test_close_waits_for_startup_and_keeps_the_original_provider():
    """A late accepted create is closed before lifecycle completes."""
    original = object()
    pool = object.__new__(DaytonaChannels)
    pool.sandbox = SimpleNamespace(_sandbox=original, _exec_generation=1)
    pool.provider, pool.generation, pool.closed = original, 1, False
    made = SimpleNamespace(close=AsyncMock())
    pool.channels = []
    release = asyncio.Event()

    async def start():
        await release.wait()
        pool.channels.append(made)

    pool.starting = asyncio.create_task(start())
    closing = asyncio.create_task(pool.close())
    await asyncio.sleep(0)
    assert not pool.current() and not closing.done()
    pool.sandbox._sandbox = object()
    release.set()
    await closing
    assert pool.provider is original
    made.close.assert_awaited_once()


@pytest.mark.parametrize(
    "journal,expected",
    [("J", True), ("Jstarted 1\n", False), ("Jstarted ", False), (None, False)],
)
async def test_fallback_requires_a_final_readable_journal(journal, expected):
    """Only a killed helper's complete journal can prove no start."""
    from benchflow.diagnostics import TransientSandboxTransportError

    made = channel()
    made.pid, made.journal = 123, "/private/journal"
    made.verdict = AsyncMock(return_value=ExecResult(return_code=0, stdout=journal))
    made.run = AsyncMock(side_effect=ChannelUnconfirmed("lost", 1))
    pool = object.__new__(DaytonaChannels)
    counts = _ExecCounts()
    pool.sandbox = SimpleNamespace(
        _sandbox=object(), _exec_generation=1, _exec_counts=lambda: counts
    )
    pool.provider, pool.generation, pool.closed = pool.sandbox._sandbox, 1, False
    pool.starting = asyncio.get_running_loop().create_future()
    pool.starting.set_result(made)
    if expected:
        assert await pool.run("side-effect", KIT_PROBE_ROUTE, 30) is None
    else:
        with pytest.raises(TransientSandboxTransportError):
            await pool.run("side-effect", KIT_PROBE_ROUTE, 30)
    made.run.assert_awaited_once()
    made.verdict.assert_awaited_once()


@pytest.fixture
async def local_pty(tmp_path):
    """Exercise the actual protected-helper source on a real terminal."""
    if sys.platform != "linux":
        pytest.skip("Linux PTY and /proc required")
    import fcntl
    import pty
    import termios

    made = DaytonaChannel(SimpleNamespace(), _ExecCounts())
    base = str(tmp_path / "context")
    given = {b"PATH": b"/usr/bin:/bin", b"LANG": b"C", b"BF_DATA": b"unchanged"}
    Path(base + ".env").write_bytes(
        b"\0".join(k + b"=" + v for k, v in given.items()) + b"\0"
    )
    Path(base + ".cwd").write_text(str(tmp_path) + "\n")
    Path(base + ".mask").write_text("0027\n")
    source = tmp_path / "helper.py"
    source.write_text(
        f"BF_READY_MARK = {made.marker!r}\n" + _PTY_PRELUDE + HELPER_SOURCE
    )
    master, slave = pty.openpty()

    def session():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(source),
        str(tmp_path / "journal"),
        base,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        preexec_fn=session,
    )
    os.close(slave)
    os.set_blocking(master, False)
    readable = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_reader(master, readable.set)

    async def pump():
        while True:
            await readable.wait()
            readable.clear()
            try:
                data = os.read(master, 65536)
            except BlockingIOError:
                continue
            except OSError:
                return
            if not data:
                return
            await made.on_data(data)

    reader = asyncio.create_task(pump())

    async def send(data):
        while data:
            try:
                size = os.write(master, data)
                data = data[size:]
            except BlockingIOError:
                await asyncio.sleep(0.001)

    made.pty = SimpleNamespace(send_input=send)
    await asyncio.wait_for(made.ready, 5)
    try:
        yield made, process
    finally:
        if process.returncode is None:
            process.kill()
        await process.wait()
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        loop.remove_reader(master)
        os.close(master)


async def test_real_pty_binary_output_environment_cwd_and_umask(local_pty, tmp_path):
    """Terminal state never reaches children; binary streams stay exact."""
    made, _ = local_pty
    script = 'env | sort; pwd; umask; printf "out\\r\\n"; printf "err\\r\\n" >&2'
    result = await made.run(script, "/bin/sh", 10)
    assert result.return_code == 0
    assert (
        result.stream(1)
        == f"BF_DATA=unchanged\nLANG=C\nPATH=/usr/bin:/bin\nPWD={tmp_path}\n{tmp_path}\n0027\nout\r\n".encode()
    )
    assert result.stream(2) == b"err\r\n"
    binary = await made.run("/usr/bin/head -c 3000000 /dev/zero", "/bin/sh", 10)
    assert binary.stream(1) == b"\0" * 3000000


async def test_real_pty_timeout_leaves_child_running_and_channel_usable(
    local_pty, tmp_path
):
    """Host timeout stops waiting while its session-leader child continues."""
    made, _ = local_pty
    mark = tmp_path / "after"
    release = tmp_path / "release"
    with pytest.raises(TimeoutError):
        await made.run(
            f"while test ! -f {release}; do sleep .01; done; echo done > {mark}",
            "/bin/sh",
            0.05,
        )
    release.touch()
    while not mark.exists():
        await asyncio.sleep(0.01)
    assert mark.read_text() == "done\n"
    assert (await made.run("echo cleanup", "/bin/sh", 10)).stream(1) == b"cleanup\n"


async def test_real_pty_waits_for_background_stream_eof(local_pty):
    """Preserve Daytona's EOF rule instead of Docker's two-second grace."""
    made, _ = local_pty
    result = await made.run("(sleep 2.2; echo late) & echo now", "/bin/sh", 5)
    assert result.stream(1) == b"now\nlate\n"


async def test_real_pty_concurrency_preserves_independent_results(local_pty):
    """Concurrent children retain their frame IDs, status and streams."""
    made, _ = local_pty
    results = await asyncio.gather(
        *(
            made.run(f"sleep .1; echo {i}; echo e{i} >&2; exit {i}", "/bin/sh", 5)
            for i in range(8)
        )
    )
    assert [r.return_code for r in results] == list(range(8))
    assert [r.stream(1) for r in results] == [f"{i}\n".encode() for i in range(8)]
    assert [r.stream(2) for r in results] == [f"e{i}\n".encode() for i in range(8)]


async def test_real_pty_lost_start_acknowledgement_is_in_journal(local_pty, tmp_path):
    """Side effects with a lost response are listed before they can run."""
    made, _ = local_pty
    mark = tmp_path / "ran"
    task = asyncio.create_task(made.run(f"echo once >> {mark}; sleep .2", "/bin/sh", 5))
    for _ in range(100):
        if mark.exists():
            break
        await asyncio.sleep(0.01)
    made.die("response lost")
    with pytest.raises(ChannelLost):
        await task
    assert mark.read_text() == "once\n"
    assert (tmp_path / "journal").read_text() == "started 1\n"


async def test_deadline_queries_before_reporting_timeout():
    """A shell timeout remains exit 124 after the final status query."""
    made = channel()

    async def send(data):
        head = json.loads(data[4:])
        if "q" in head:
            await made.on_data(output(b"X", head["q"], b"124"))
        else:
            await made.on_data(output(b"S", head["i"]))

    made.pty.send_input.side_effect = send
    result = await made.run("/usr/bin/timeout 1 sleep 2", "/bin/sh", 0.01)
    assert result.return_code == 124
    assert made.pty.send_input.await_count == 2


async def test_cancelled_queued_step_is_withdrawn_before_the_helper_resumes(
    local_pty, tmp_path
):
    """A cancelled caller cannot leave a queued side effect behind."""
    import signal

    made, process = local_pty
    os.kill(process.pid, signal.SIGSTOP)
    mark = tmp_path / "late"
    try:
        task = asyncio.create_task(made.run(f"echo late > {mark}", "/bin/sh", 10))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        if made.withdrawals:
            await asyncio.gather(*made.withdrawals)
    finally:
        os.kill(process.pid, signal.SIGCONT)
    assert (await made.run("echo alive", "/bin/sh", 5)).stream(1) == b"alive\n"
    assert not mark.exists()


async def test_cancelling_close_still_drains_the_accepted_start():
    """Cancelling lifecycle cleanup cannot orphan a late channel."""
    pool = object.__new__(DaytonaChannels)
    pool.closed = False
    pool.channels = []
    release = asyncio.Event()
    made = SimpleNamespace(close=AsyncMock())

    async def start():
        await release.wait()
        pool.channels.append(made)

    pool.starting = asyncio.create_task(start())
    closing = asyncio.create_task(pool.close())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done() and not pool.starting.cancelled()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    made.close.assert_awaited_once()


async def test_stop_deletes_provider_even_when_channel_close_is_cancelled(monkeypatch):
    """Channel teardown cancellation cannot skip sandbox deletion."""
    from benchflow.sandbox.daytona import DaytonaSandbox

    box = object.__new__(DaytonaSandbox)
    box._sandbox = object()
    box._strategy = SimpleNamespace(stop=AsyncMock())
    channels = box._command_channels = SimpleNamespace(
        provider=box._sandbox,
        retire=Mock(),
        finalize=Mock(),
        join=AsyncMock(),
        finalized=False,
        close=AsyncMock(side_effect=asyncio.CancelledError()),
        cleanup_task=None,
    )
    box._drain_create_attempts = AsyncMock()
    box._write_exec_counts = lambda: None
    # A surviving provider's cleanup can already be running at stop.
    box._retire_command_channels()
    box._queue_command_channel_cleanup()
    await box.stop(True)
    with pytest.raises(asyncio.CancelledError):
        await channels.cleanup_task
    box._strategy.stop.assert_awaited_once_with(True)
    channels.retire.assert_called_once()
    channels.finalize.assert_called_once()


async def test_default_stdin_stays_open_like_a_daytona_session(local_pty):
    """An unstaged stdin read waits on a pipe instead of seeing EOF."""
    made, _ = local_pty
    result = await made.run(
        "test -p /proc/self/fd/0 && /usr/bin/timeout .1 head -c 1", "/bin/sh", 5
    )
    assert result.return_code == 124
    assert result.stream(1) == b""


@pytest.mark.parametrize(
    "data,expected",
    [
        (b"", ""),
        (b"\0", ""),
        (b"a\0b\0c\0", "abc\n"),
        (b"\0\n\0", "\n"),
        (b"ab", "ab\n"),
        (b"\r", "\r\n"),
        (b"\xff\xfe", "\ufffd\ufffd\n"),
        (b"\xe2\x82", "\ufffd\ufffd\n"),  # replacement per byte.
        (b"a\rb\r\nc", "a\rb\r\nc\n"),
        (b"\n\n", "\n\n"),
        (b"x" * 200000, "x" * 200000 + "\n"),
    ],
)
def test_session_output_matches_the_vm15_provider_probe(data, expected):
    """Retain measured SDK log records, including unterminated lines."""
    from benchflow.sandbox.daytona_channel import session_output

    assert session_output(data) == expected


async def test_waiting_for_another_frame_obeys_its_own_budget():
    """An unsent command retains proof permitting fallback."""

    made = channel()
    await made.write_lock.acquire()
    try:
        with pytest.raises(ChannelUnavailable):
            await made.run("side-effect", "/bin/sh", 0.01)
    finally:
        made.write_lock.release()
    made.pty.send_input.assert_not_awaited()


async def test_close_prevents_the_rest_of_a_partial_frame_from_being_sent():
    """Lifecycle invalidation cannot deliver the end of an old request."""
    made = channel()

    async def send(data):
        made.die("lifecycle changed")

    made.pty.send_input.side_effect = send
    with pytest.raises(ChannelUnconfirmed):
        await made.run("x" * CHUNK_SIZE * 2, "/bin/sh", 1)
    assert made.pty.send_input.await_count == 1


def test_missing_pty_support_falls_back_without_staging_a_helper():
    """Older SDKs and session-only adapters cannot start a channel."""
    box = SimpleNamespace(
        exec_channel=True, _sandbox=SimpleNamespace(process=SimpleNamespace())
    )
    assert not eligible(box, KIT_PROBE_ROUTE)


async def test_close_cancels_a_watcher_that_disconnect_did_not_finish():
    """A stuck SDK watcher cannot keep lifecycle cleanup waiting."""
    made = channel()
    made.pty.kill = AsyncMock()
    made.pty.disconnect = AsyncMock()
    made.gone = asyncio.create_task(asyncio.Event().wait())
    await asyncio.wait_for(made.close(), 1)
    assert made.gone.cancelled()
    made.pty.kill.assert_awaited_once()
    made.pty.disconnect.assert_awaited_once()
