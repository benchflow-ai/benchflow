"""The sandbox's command channel (PERF-B, sandbox/_exec_channel.py).

Root's steps on a Docker sandbox with the protected kit go through one
long-lived helper instead of a ``docker exec`` each. The first half drives
the helper itself as a local process: exit statuses, stdin, large output,
timeouts, concurrent commands, a channel that dies, Ctrl-C and Docker's 2 s
wait for a background writer. The second half runs BenchFlow's real
``DockerSandbox.exec`` against a container that holds the kit and checks
that a command through the channel runs exactly as its own ``docker exec``
would (C8's route, K1's barrier, the environment and its transport), and
that root's steps no longer start a docker client each.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest

from benchflow.sandbox._base import ExecResult

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the helper reads /proc"
)


def _bfx() -> str:
    """The helper: the pinned bfx for this machine (``make bfx``)."""
    from benchflow._bfx.locate import BfxUnavailable, verified_binary

    try:
        return str(verified_binary(platform.machine()))
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx for this machine: {exc}")


#: The channel's two helpers: bfx where it is delivered, and the kit's
#: Python running HELPER_SOURCE where it is not (TRUSTED-RUNTIME.md v3
#: scope: a backend without bfx keeps today's route). Here the Python is
#: this test's own, started as docker.py starts the kit's (-I -S -B).
HELPERS = ("bfx", "python")


@pytest.fixture(params=HELPERS)
def helper(request) -> str:
    if request.param == "bfx":
        _bfx()
    return request.param


def _helper(kind: str, *journal: str, prelude: str = "") -> list[str]:
    from benchflow.sandbox._exec_channel import HELPER_SOURCE, helper_argv

    if kind == "python":
        return [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            prelude + HELPER_SOURCE,
            *journal,
        ]
    return helper_argv(_bfx(), *journal)


def _verdict(kind: str) -> list[str]:
    from benchflow.sandbox._exec_channel import VERDICT_SOURCE, verdict_argv

    if kind == "python":
        return [sys.executable, "-I", "-S", "-B", "-c", VERDICT_SOURCE]
    return verdict_argv(_bfx())


def _local_channel(kind: str, env: dict[str, str] | None = None):
    from benchflow.sandbox._exec_channel import ExecChannel

    return ExecChannel(
        _helper(kind),
        env,
        workdir="/",
        logger=logging.getLogger("test"),
    )


SH = ["/bin/sh", "-c"]


def _journal_ids(kind: str, channel) -> frozenset[int] | None:
    """What the sandbox learns after *channel* died: the helper's verdict run
    locally with the helper's journal and pid (it kills a helper that lives
    on, then reads the journal), as docker.py runs it in the container."""
    from benchflow.sandbox._exec_channel import parse_journal

    done = subprocess.run(
        [
            *_verdict(kind),
            channel.journal, str(channel.helper_pid),
        ],
        capture_output=True,
        timeout=30,
    )  # fmt: skip
    return parse_journal(done.stdout) if done.returncode == 0 else None


# Stands in for a transport that dies while the helper lives (a killed
# docker client, a remote daemon's dropped connection; Docker keeps the
# exec process running): it relays stdin to the helper and the helper's
# output up to and including the ready frame, drops the next frame (the
# first step's S), closes the host side and keeps the helper running.
# Review R3-M1's relay, reading the ready frame by its length (its payload
# is now the helper's pid).
RELAY = r"""
import os, struct, subprocess, sys, threading
helper = subprocess.Popen(sys.argv[1:], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
def pump_in():
    while True:
        b = os.read(0, 65536)
        if not b:
            break
        try:
            helper.stdin.write(b); helper.stdin.flush()
        except OSError:
            break
threading.Thread(target=pump_in, daemon=True).start()
fd = helper.stdout.fileno()
def exactly(n):
    data = b''
    while len(data) < n:
        b = os.read(fd, n - len(data))
        if not b:
            sys.exit(0)
        data += b
    return data
head = exactly(9)
os.write(1, head + exactly(struct.unpack('>II', head[1:])[1]))
exactly(9)                              # the step's S frame: dropped
os.close(1)                             # the host side ends here
while os.read(fd, 65536):               # the helper lives on
    pass
helper.wait()
"""


@linux_only
@pytest.mark.asyncio
async def test_exit_statuses_are_the_commands_and_128_plus_a_signal(helper):
    channel = _local_channel(helper)
    await channel.start()
    try:
        for script, want in (
            ("exit 0", 0),
            ("exit 3", 3),
            ("exit 255", 255),
            ("kill -9 $$", 137),
            ("kill -15 $$", 143),
        ):
            out = await channel.run([*SH, script], cwd="/", stdin_data=None)
            assert out.return_code == want, script
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_stdin_is_the_commands_input_or_dev_null(tmp_path, helper):
    channel = _local_channel(helper)
    await channel.start()
    try:
        out = await channel.run(
            [*SH, "cat; echo; readlink /proc/self/fd/0"],
            cwd="/",
            stdin_data=b"line one\nline two",
        )
        assert out.stream(1).startswith(b"line one\nline two\n")
        assert out.stream(1).split(b"\n")[2].startswith(b"pipe:")
        none = await channel.run(
            [*SH, "readlink /proc/self/fd/0; wc -c"], cwd="/", stdin_data=None
        )
        assert none.stream(1) == b"/dev/null\n0\n"
        empty = await channel.run([*SH, "wc -c"], cwd="/", stdin_data=b"")
        assert empty.stream(1).strip() == b"0"
        big = os.urandom(3 * 1024 * 1024)
        echoed = await channel.run([*SH, "cat"], cwd="/", stdin_data=big)
        assert echoed.stream(1) == big
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_large_output_on_both_streams_arrives_whole_and_apart(helper):
    channel = _local_channel(helper)
    await channel.start()
    try:
        out = await channel.run(
            [
                *SH,
                "head -c 12000000 /dev/zero | tr '\\0' a; "
                "head -c 5000000 /dev/zero | tr '\\0' b >&2; echo end",
            ],
            cwd="/",
            stdin_data=None,
        )
        assert out.return_code == 0
        assert out.stream(1) == b"a" * 12_000_000 + b"end\n"
        assert out.stream(2) == b"b" * 5_000_000
        assert len(out.merged()) == 17_000_004
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_a_timeout_stops_the_wait_and_the_command_goes_on(tmp_path, helper):
    """As when BenchFlow kills a ``docker exec`` client: the exec itself is
    not stopped. The channel keeps serving."""
    mark = tmp_path / "after"
    channel = _local_channel(helper)
    await channel.start()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                channel.run([*SH, f"sleep 1; : > {mark}"], cwd="/", stdin_data=None),
                timeout=0.2,
            )
        assert not mark.exists()
        await asyncio.sleep(1.5)
        assert mark.exists()
        out = await channel.run([*SH, "echo still"], cwd="/", stdin_data=None)
        assert out.stream(1) == b"still\n"
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_concurrent_commands_run_at_once_and_keep_their_own_output(helper):
    channel = _local_channel(helper)
    await channel.start()
    try:
        started = time.monotonic()
        outs = await asyncio.gather(
            *(
                channel.run(
                    [*SH, f"sleep 0.5; echo out{i}; echo err{i} >&2; exit {i % 7}"],
                    cwd="/",
                    stdin_data=f"in{i}".encode(),
                )
                for i in range(30)
            )
        )
        assert time.monotonic() - started < 3
        for i, out in enumerate(outs):
            assert out.stream(1) == f"out{i}\n".encode()
            assert out.stream(2) == f"err{i}\n".encode()
            assert out.return_code == i % 7
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_a_dead_channel_fails_the_running_command_and_refuses_the_next(helper):
    from benchflow.sandbox._exec_channel import ChannelLost, ChannelUnavailable

    channel = _local_channel(helper)
    await channel.start()
    try:
        running = asyncio.ensure_future(
            channel.run([*SH, "sleep 5"], cwd="/", stdin_data=None)
        )
        await asyncio.sleep(0.3)
        assert channel._process is not None
        channel._process.kill()
        with pytest.raises(ChannelLost, match="command channel stopped"):
            await asyncio.wait_for(running, timeout=5)
        assert not channel.alive
        with pytest.raises(ChannelUnavailable):
            await channel.run([*SH, "true"], cwd="/", stdin_data=None)
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_ctrl_c_cancels_the_wait_not_the_channel(helper):
    """A cancelled command (Ctrl-C) leaves the channel to the clean-up, and
    the channel's client is in its own session, so the terminal's SIGINT
    never reaches it."""
    channel = _local_channel(helper)
    await channel.start()
    try:
        assert channel._process is not None
        assert os.getsid(channel._process.pid) != os.getsid(0)
        task = asyncio.ensure_future(
            channel.run([*SH, "sleep 3"], cwd="/", stdin_data=None)
        )
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert channel.alive
        out = await channel.run([*SH, "echo cleanup"], cwd="/", stdin_data=None)
        assert out.stream(1) == b"cleanup\n"
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_a_background_writer_holds_the_result_for_dockers_2_seconds(helper):
    """Docker returns an exec once it exits, after reading what a process it
    left behind still writes for up to 2 s; the channel does the same."""
    channel = _local_channel(helper)
    await channel.start()
    try:
        started = time.monotonic()
        out = await channel.run(
            [*SH, "(sleep 0.5; echo late; sleep 5; echo too-late) & echo now"],
            cwd="/",
            stdin_data=None,
        )
        took = time.monotonic() - started
        assert out.stream(1) == b"now\nlate\n"
        assert 1.9 < took < 3.5
        quick = time.monotonic()
        await channel.run(
            [*SH, "(sleep 5 >/dev/null 2>&1 &); echo ok"], cwd="/", stdin_data=None
        )
        assert time.monotonic() - quick < 1
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_a_step_sent_to_a_helper_that_already_died_is_not_lost(tmp_path, helper):
    """Guards R2-M1 (review of 73a53f94): once the helper dies, the docker
    client still holds its pipes for a while, so the next frame is written
    and then failed as "may or may not have run". Now the
    helper lists a step in its journal before the step can exist; the
    journal of the dead helper does not list this one, so exec() gives it
    its own docker exec, and lists the step that killed it. A shell around
    the helper keeps its pipes open after it dies, as the docker client
    does."""
    from benchflow.sandbox._exec_channel import (
        ChannelLost,
        ChannelUnconfirmed,
        ExecChannel,
    )

    journal = str(tmp_path / "journal")
    channel = ExecChannel(
        [
            "/bin/sh",
            "-c",
            '"$@"; sleep 3',
            "sh",
            *_helper(helper, journal),
        ],
        None,
        workdir="/",
        logger=logging.getLogger("test"),
        journal=journal,
    )
    await channel.start()
    try:
        # A step that kills its parent, the helper: it did start.
        first = asyncio.ensure_future(
            channel.run([*SH, "kill -9 $PPID"], cwd="/", stdin_data=None)
        )
        await asyncio.sleep(0.5)
        # Sent while the dead helper's pipes are still open: never started.
        mark = tmp_path / "ran"
        with pytest.raises(ChannelUnconfirmed) as late:
            await channel.run([*SH, f": > {mark}"], cwd="/", stdin_data=None)
        assert not mark.exists()
        with pytest.raises(ChannelLost):
            await first
        started = _journal_ids(helper, channel)
        assert started is not None
        assert late.value.ident not in started  # so it may fall back
        assert len(started) == 1  # the step that killed the helper
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_a_step_whose_s_frame_the_transport_lost_is_not_reported_not_started(
    tmp_path,
    helper,
):
    """Guards R3-M1 (review of 1eb1e5d6): when the transport dies but the
    helper lives, a step can start while its S frame is lost. Round 3 then
    reported it not started (ChannelUnavailable) and exec() ran it a second
    time. The channel now reports it unconfirmed, and the dead helper's
    journal, read after the helper is killed, lists it: it may have run, so
    it is never started again. The relay plays the transport (the review's
    test, with the ready frame read by its length)."""
    from benchflow.sandbox._exec_channel import (
        ChannelLost,
        ChannelUnavailable,
        ExecChannel,
    )

    journal = str(tmp_path / "journal")
    mark = tmp_path / "ran"
    channel = ExecChannel(
        [
            sys.executable, "-c", RELAY,
            *_helper(helper, journal),
        ],
        None,
        workdir="/",
        logger=logging.getLogger("test"),
        journal=journal,
    )  # fmt: skip
    await channel.start()
    try:
        outcome = None
        try:
            await channel.run(
                [*SH, f"echo ran >> {mark}; sleep 1"], cwd="/", stdin_data=None
            )
            outcome = "ok"
        except ChannelUnavailable:
            outcome = "not-started"
        except ChannelLost:
            outcome = "lost"
        await asyncio.sleep(1.5)
        # The review's invariant: "not started" must mean the step never ran.
        assert outcome == "lost", outcome
        assert mark.read_text() == "ran\n"
        started = _journal_ids(helper, channel)
        assert started is not None and len(started) == 1
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_a_step_given_up_while_the_helper_was_busy_never_starts(tmp_path, helper):
    """Guards PERF-B-L2: a helper that reads a step only after its caller
    gave up (here it was stopped) withdraws it on the cancel frame that
    caller sent, and never starts it."""
    import signal as signals

    from benchflow.sandbox._exec_channel import ChannelDeadline

    mark = tmp_path / "late"
    channel = _local_channel(helper)
    await channel.start()
    try:
        assert channel._process is not None
        os.kill(channel._process.pid, signals.SIGSTOP)
        try:
            with pytest.raises((TimeoutError, ChannelDeadline)):
                await asyncio.wait_for(
                    channel.run(
                        [*SH, f": > {mark}"],
                        cwd="/",
                        stdin_data=None,
                        deadline=time.monotonic() + 0.5,
                    ),
                    timeout=0.5,
                )
        finally:
            os.kill(channel._process.pid, signals.SIGCONT)
        await asyncio.sleep(1.0)
        assert not mark.exists()
        out = await channel.run([*SH, "echo alive"], cwd="/", stdin_data=None)
        assert out.stream(1) == b"alive\n"
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_a_wall_clock_step_does_not_refuse_later_steps(helper):
    """Guards R2-L1 (review of 73a53f94): one clock offset taken when the
    channel started turned every later step into a refused "timeout" once
    the helper's wall clock stepped (NTP, a resumed VM, a remote daemon).
    The step carries its time left, counted on the helper's monotonic
    clock. Here the Python helper's wall clock jumps 600 s after its ready
    frame; bfx reads no wall clock at all (its deadlines are monotonic
    instants), so no jump can be injected into it. The steps run."""
    from benchflow.sandbox._exec_channel import ExecChannel

    jump = (
        "import time\n"
        "_wall = time.time\n"
        "_calls = [0]\n"
        "def _jumped():\n"
        "    _calls[0] += 1\n"
        "    return _wall() + (600 if _calls[0] > 1 else 0)\n"
        "time.time = _jumped\n"
    )
    channel = ExecChannel(
        _helper(helper, prelude=jump),
        None,
        workdir="/",
        logger=logging.getLogger("test"),
    )
    await channel.start()
    try:
        for _ in range(3):
            out = await channel.run(
                [*SH, "echo ran"],
                cwd="/",
                stdin_data=None,
                deadline=time.monotonic() + 30,
            )
            assert out.return_code == 0 and out.stream(1) == b"ran\n"
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_a_command_the_helper_cannot_start_is_not_started(tmp_path, helper):
    """A missing folder or program: the helper says so before anything runs,
    so exec() gives the command its own docker exec (and Docker's error)."""
    from benchflow.sandbox._exec_channel import ChannelUnavailable

    channel = _local_channel(helper)
    await channel.start()
    try:
        with pytest.raises(ChannelUnavailable):
            await channel.run(
                [*SH, "true"], cwd=str(tmp_path / "missing"), stdin_data=None
            )
        with pytest.raises(ChannelUnavailable):
            await channel.run(
                [str(tmp_path / "no-shell"), "-c", "true"], cwd="/", stdin_data=None
            )
        assert channel.alive
    finally:
        await channel.close()


@linux_only
@pytest.mark.asyncio
async def test_commands_get_the_helpers_environment_unchanged_and_lead_a_session(
    tmp_path,
    helper,
):
    """Under a C locale Python adds LC_CTYPE to its own environment (PEP 538,
    even with -I); a command still gets exactly what the helper was given."""
    env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LD_PRELOAD": "", "BF_X": "a b"}
    channel = _local_channel(helper, env)
    await channel.start()
    try:
        out = await channel.run(
            [*SH, "cat /proc/$$/environ; echo; cut -d' ' -f6 /proc/$$/stat; echo $$"],
            cwd=str(tmp_path),
            stdin_data=None,
        )
        environ, sid, pid = out.stream(1).decode().rsplit("\n", 3)[:3]
        given = dict(item.split("=", 1) for item in environ.split("\0") if item)
        assert given == env
        assert sid == pid
    finally:
        await channel.close()


def test_only_the_kits_own_python_wrappers_are_started_directly():
    from benchflow.sandbox._exec_channel import helper_python_argv

    loader = (
        "#!/opt/benchflow/rt/bin/sh\nexec /lib64/ld-linux-x86-64.so.2 --library-path "
        '/opt/benchflow/rt/lib /opt/benchflow/rt/libexec/python3 -I -S -B "$@"\n'
    )
    direct = (
        "#!/opt/benchflow/rt/bin/sh\n"
        'exec /opt/benchflow/rt/libexec/python3 -I -S -B "$@"\n'
    )
    assert helper_python_argv(loader) == [
        "/lib64/ld-linux-x86-64.so.2",
        "--library-path",
        "/opt/benchflow/rt/lib",
        "/opt/benchflow/rt/libexec/python3",
        "-I",
        "-S",
        "-B",
    ]
    assert helper_python_argv(direct) == [
        "/opt/benchflow/rt/libexec/python3",
        "-I",
        "-S",
        "-B",
    ]
    for bad in (
        direct.replace("-I -S -B", "-S -B"),
        direct.replace("/opt/benchflow/rt/libexec", "/srv/bin"),
        loader.replace("/lib64/ld-linux-x86-64.so.2", "/srv/ld-linux-x86-64.so.2"),
        direct + "echo extra\n",
        "",
    ):
        assert helper_python_argv(bad) is None


def _gnu_head() -> str | None:
    head = shutil.which("head")
    if head is None:
        return None
    version = subprocess.run([head, "--version"], capture_output=True, text=True)
    return head if version.stdout.startswith("head (GNU coreutils)") else None


def _local_pool(size: int = 3):
    from benchflow.sandbox._exec_channel import WarmExecPool

    head = _gnu_head()
    if head is None:
        pytest.skip("needs GNU head")
    return WarmExecPool([], None, shell="/bin/sh", head=head, size=size)


@pytest.mark.asyncio
async def test_a_pre_started_exec_cut_off_while_its_stdin_was_sent_is_lost():
    """R3-M1's class in the pre-started execs: a client that passed on the
    command and its "OK" line and then died with stdin bytes still pending
    may have started the step, so the step is lost (never re-run through
    docker exec). Cut off with no stdin, "OK" is the frame's last line and
    nothing can have started: it falls back as before."""
    from benchflow.sandbox._exec_channel import (
        ChannelLost,
        ChannelUnavailable,
        WarmExecPool,
    )

    class CutOff:
        returncode = 0
        pid = 0

        def __init__(self):
            self.stdin = SimpleNamespace(
                write=lambda data: None, drain=self._drain, close=lambda: None
            )

        async def _drain(self):
            raise ConnectionResetError("the client died")

        async def wait(self):
            return 0

    pool = WarmExecPool([], None, shell="/bin/sh", head="/bin/head", size=1)
    pool._take = lambda: CutOff()
    with pytest.raises(ChannelLost):
        await pool.run([*SH, "cat"], cwd=None, stdin_data=b"input")
    with pytest.raises(ChannelUnavailable):
        await pool.run([*SH, "true"], cwd=None, stdin_data=None)


async def _pool_ready(pool) -> None:
    pool.fill()
    for _ in range(100):
        if len(pool._ready) >= pool._size:
            return
        await asyncio.sleep(0.02)


@linux_only
@pytest.mark.asyncio
async def test_a_pre_started_exec_runs_the_command_it_reads(tmp_path):
    """The pool's reader becomes ``/bin/sh -c <command>``: its status, its
    output, its input (or /dev/null), its own command line."""
    pool = _local_pool()
    try:
        await _pool_ready(pool)
        out = await pool.run(
            [*SH, "cat; echo; tr '\\0' ' ' </proc/$$/cmdline; exit 7"],
            cwd=None,
            stdin_data=b"input",
        )
        assert out.return_code == 7
        assert (
            out.stream(1)
            == b"input\n/bin/sh -c cat; echo; tr '\\0' ' ' </proc/$$/cmdline; exit 7 "
        )
        await _pool_ready(pool)
        none = await pool.run(
            [*SH, "readlink /proc/self/fd/0"], cwd=None, stdin_data=None
        )
        assert none.stream(1) == b"/dev/null\n"
        await _pool_ready(pool)
        big = "x=1\n" * 20000 + "head -c 4000000 /dev/zero | tr '\\0' a; kill -9 $$"
        large = await pool.run([*SH, big], cwd=None, stdin_data=None)
        assert (
            large.stream(1) == b"a" * 4_000_000 and large.return_code == -9
        )  # Docker's 137; here no Docker maps it
    finally:
        await pool.close()


@linux_only
@pytest.mark.asyncio
async def test_a_pre_started_exec_never_runs_a_cut_off_command(tmp_path):
    """A write that stops short of the command's last byte, or of the OK
    line after it, runs nothing (exit 125)."""
    from benchflow.sandbox._exec_channel import warm_reader

    head = _gnu_head()
    if head is None:
        pytest.skip("needs GNU head")
    mark = tmp_path / "ran"
    command = f": > {mark}".encode()
    for sent in (
        b"%d 0\n" % (len(command) + 5) + command,
        b"%d 0\n" % len(command) + command,
        b"%d 0\n" % len(command) + command + b"O",
    ):
        done = subprocess.run(
            ["/bin/sh", "-c", warm_reader("/bin/sh", head)], input=sent
        )
        assert done.returncode == 125
        assert not mark.exists()


@linux_only
@pytest.mark.asyncio
async def test_pre_started_execs_refuse_what_they_cannot_run():
    """Another shell, a -w folder or an empty pool: not started, so exec()
    gives the step its own docker exec."""
    from benchflow.sandbox._exec_channel import ChannelUnavailable

    pool = _local_pool(size=1)
    try:
        await _pool_ready(pool)
        for argv, cwd in (
            (["/bin/bash", "-c", "true"], None),
            ([*SH, "true"], "/tmp"),
        ):
            with pytest.raises(ChannelUnavailable):
                await pool.run(argv, cwd=cwd, stdin_data=None)
        first = asyncio.ensure_future(
            pool.run([*SH, "sleep 0.3"], cwd=None, stdin_data=None)
        )
        await asyncio.sleep(0)
        with pytest.raises(ChannelUnavailable):
            await pool.run([*SH, "true"], cwd=None, stdin_data=None)
        assert (await first).return_code == 0
    finally:
        await pool.close()


# --- BenchFlow's DockerSandbox.exec on a container with the kit -------------

# (image, bfx at /.bfx as DockerSandbox's read-only bind): bfx's helper; the
# same image without bfx, whose kit's Python is the helper (today's route,
# kept where bfx is not delivered); an image whose kit has no Python (the
# pre-started execs; Ubuntu's head is GNU's).
BOXES = (
    ("python:3.12-slim", True),
    ("python:3.12-slim", False),
    ("ubuntu:24.04", False),
)
BOX_IDS = ("python-bfx", "python-kit", "ubuntu-pool")
# How each helper shows in its process's command line.
HELPER_MARKS = {"bfx": "--frames", "python": "GRACE = 2.0"}
WORKSPACE = "/srv"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str, user: str = "0") -> subprocess.CompletedProcess:
    """A command in the container as the test's harness (not BenchFlow)."""
    return subprocess.run(
        ["docker", "exec", "-u", user, "-e", "LD_PRELOAD=", name, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )  # fmt: skip


def _sandbox(name: str, calls: list[list[str]]):
    """A DockerSandbox whose ``main`` container is *name*, its docker calls
    run with the real CLI and counted in *calls*."""
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = object.__new__(DockerSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._main_container = name
    sandbox.logger = logging.getLogger("test")
    sandbox._launch = SimpleNamespace(docker_args=[], env=dict(os.environ))

    async def run(argv, check=False, timeout_sec=None, **kwargs):
        calls.append(list(argv))
        raw = kwargs.get("raw_output", False)
        done = await asyncio.to_thread(
            subprocess.run,
            ["docker", *argv],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if raw else subprocess.STDOUT,
            input=kwargs.get("stdin_data"),
            timeout=timeout_sec,
        )
        # As _run_docker_process: stderr on stdout unless raw, empty is None.
        result = ExecResult(
            return_code=done.returncode,
            stdout=done.stdout.decode(errors="replace") or None,
            stderr=(done.stderr or b"").decode(errors="replace") or None,
        )
        if check and result.return_code != 0:
            raise RuntimeError(result.stderr)
        return result

    sandbox._run_docker_container_command = run
    sandbox._docker_cli = run
    sandbox._docker_client_env = lambda: None
    return sandbox


@pytest.fixture(params=BOXES, ids=BOX_IDS)
def kit_box(request, monkeypatch):
    image, with_bfx = request.param
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    from benchflow.sandbox.docker import DockerSandbox

    monkeypatch.setattr(DockerSandbox, "exec_channel", True, raising=False)
    name = "PERF-B-exec-channel-" + uuid.uuid4().hex[:8]
    helper = []
    if with_bfx:
        from benchflow._bfx import delivery

        folder = delivery.host_dir(platform.machine()) if _bfx() else None
        helper = delivery.run_args(delivery.BIND, source=str(folder))
    # A C locale (Python adds LC_CTYPE to its own environment there), an
    # image preload that does not exist (any root step that kept the image's
    # loader values says "cannot be preloaded"), K1's barrier as BenchFlow
    # creates a sandbox, and the agent's folder as WORKDIR.
    run = subprocess.run(
        [
            "docker", "run", "-d", "--name", name, "--network", "none",
            "--security-opt", "no-new-privileges:true",
            "-e", "LANG=C", "-e", "LD_PRELOAD=/usr/local/lib/bf-missing.so",
            *helper,
            "-w", WORKSPACE, image, "sleep", "1200",
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {image}: {run.stderr.strip()}")
    made = _dx(name, f"mkdir -p {WORKSPACE}")
    assert made.returncode == 0, made.stderr
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
async def kit_sandbox(kit_box):
    from benchflow.sandbox.lockdown import setup_sandbox_user

    calls: list[list[str]] = []
    sandbox = _sandbox(kit_box, calls)
    sandbox.calls = calls
    if _dx(kit_box, "test -x /.bfx/bfx").returncode == 0:
        sandbox.bfx_delivery = "bind"
    try:
        await setup_sandbox_user(sandbox, "agent", WORKSPACE, guard_daemon=False)
        assert sandbox.runtime_kit is not None
        yield sandbox
    finally:
        close = getattr(sandbox, "_close_exec_channels", None)
        if close is not None:  # absent on a build without the channel
            await close()


def _helper_kind(sandbox) -> str | None:
    """Which helper the sandbox's channel runs: "bfx", "python" (the kit's),
    or None for the pre-started execs."""
    if sandbox.bfx_delivery:
        return "bfx"
    return "python" if sandbox.runtime_kit.python else None


def _helper_pids(sandbox, name: str) -> list[str]:
    mark = HELPER_MARKS[_helper_kind(sandbox)]
    return _dx(
        name,
        f"for p in /proc/[0-9]*; do grep -q -- '{mark}' $p/cmdline 2>/dev/null "
        "&& echo ${p#/proc/}; done",
    ).stdout.split()


def _channel(sandbox, user="root"):
    from benchflow.sandbox._exec_channel import ExecChannel, WarmExecPool

    for key, state in (sandbox._exec_channels or {}).items():
        if isinstance(state, asyncio.Future) and state.done():
            state = None if state.cancelled() or state.exception() else state.result()
        if key[2] == str(user) and isinstance(state, (ExecChannel, WarmExecPool)):
            return state
    return None


async def _channel_ready(sandbox, user="root") -> None:
    """Start the channel for *user* and wait until it runs (for the pool:
    until every pre-started exec is up)."""
    await sandbox.exec("true", user=user)
    if not hasattr(sandbox, "_exec_channels"):
        # A build without the channel (the fails-before run): nothing to wait for.
        await asyncio.sleep(2)
        return
    from benchflow.sandbox._exec_channel import WarmExecPool

    for _ in range(300):
        channel = _channel(sandbox, user)
        if (
            channel is not None
            and channel.alive
            and (not isinstance(channel, WarmExecPool) or len(channel._ready) >= 3)
        ):
            return
        states = list((sandbox._exec_channels or {}).values())
        if states and all(
            s is None or (isinstance(s, asyncio.Future) and s.done() and not s.result())
            for s in states
        ):
            break
        await asyncio.sleep(0.05)
    raise AssertionError(f"the channel did not start: {sandbox._exec_channels}")


# What a root step can see of its own process: credentials, K1's barrier,
# capabilities, seccomp, its folder, umask, open descriptors, session, the
# shell it runs on and its command line.
PROBE = (
    "for k in Uid Gid Groups NoNewPrivs CapInh CapPrm CapEff CapBnd CapAmb "
    'Seccomp; do grep "^$k:" /proc/$$/status; done; '
    'echo "cwd=$(pwd)"; umask; '
    'for f in /proc/$$/fd/*; do t=$(readlink "$f"); echo "fd ${f##*/} ${t%%:*}"; done; '
    "set -- $(cat /proc/$$/stat); "
    '[ "$6" = "$$" ] && echo session-leader; '
    'echo "exe=$(readlink /proc/$$/exe)"; '
    "tr '\\0' ' ' </proc/$$/cmdline | cut -c1-60"
)


@pytest.mark.asyncio
async def test_root_steps_stop_starting_a_docker_client_each(kit_sandbox, kit_box):
    """The speed claim, by behaviour: once the channel runs, root's steps on
    the kit start no ``docker exec`` of their own (each costs 0.05-0.1 s)."""
    sandbox = kit_sandbox
    calls = sandbox.calls
    await _channel_ready(sandbox)
    before = len(calls)
    started = time.monotonic()
    for i in range(40):
        result = await sandbox.exec(f"echo {i}", user="root")
        assert result.stdout == f"{i}\n"
    took = time.monotonic() - started
    assert calls[before:] == []
    assert took / 40 < 0.05


@pytest.mark.asyncio
async def test_a_root_step_through_the_channel_runs_as_its_own_docker_exec(
    kit_sandbox, kit_box
):
    """C8's route and K1's barrier hold for a command the channel carries:
    the same user, groups, capabilities, no_new_privs, seccomp, folder,
    umask, descriptors, session, kit shell and command line as its own
    ``docker exec``; the task route's environment is the same too, with no
    image preload reaching a root step."""
    from benchflow.sandbox.docker import DockerSandbox

    sandbox = kit_sandbox
    calls = sandbox.calls

    def same(channel, direct):
        assert (channel.stdout or "").splitlines() == (direct.stdout or "").splitlines()
        assert (channel.stderr, channel.return_code) == (
            direct.stderr,
            direct.return_code,
        )

    pool = _helper_kind(sandbox) is None

    async def both(**kwargs):
        # Streams apart: Docker merges them in no fixed order.
        kwargs.setdefault("raw_output", True)
        # The pre-started execs take root's plumbing in the image's folder.
        via = not sandbox._in_task_command and (not pool or "cwd" not in kwargs)
        DockerSandbox.exec_channel = False
        try:
            direct = await sandbox.exec(**kwargs)
        finally:
            DockerSandbox.exec_channel = True
        await _channel_ready(sandbox, kwargs.get("user", "root"))
        n = len(calls)
        channel = await sandbox.exec(**kwargs)
        assert (calls[n:] == []) is via
        return direct, channel

    for user in ("root", 0):
        direct, channel = await both(command=PROBE, user=user)
        assert direct.return_code == 0, direct.stdout
        same(channel, direct)
        assert "NoNewPrivs:\t1" in channel.stdout
        assert "session-leader" in channel.stdout
        assert "exe=/opt/benchflow/rt/" in channel.stdout
        assert f"cwd={WORKSPACE}" in channel.stdout
        assert channel.stderr is None  # no image preload in root's plumbing

    direct, channel = await both(command=PROBE, user="root", cwd="/tmp")
    same(channel, direct)
    assert "cwd=/tmp" in channel.stdout

    env_probe = "env | LC_ALL=C sort; " + PROBE
    with sandbox.task_command():
        direct, channel = await both(command=env_probe, user="root")
    same(channel, direct)
    # The task's own command gets the image's values back (C8's task
    # route), its preload included; never the helper's LC_CTYPE.
    assert "LD_PRELOAD=/usr/local/lib/bf-missing.so" in channel.stdout
    assert "LC_CTYPE" not in channel.stdout

    direct, channel = await both(
        command="echo out; echo err >&2; exit 5", user="root", raw_output=True
    )
    assert (
        channel == direct == ExecResult(return_code=5, stdout="out\n", stderr="err\n")
    )


@pytest.mark.asyncio
async def test_env_values_reach_the_command_and_no_command_line(kit_sandbox, kit_box):
    """Env values through the channel: the value is on the command's
    stdin, never in any argv in the container or on the host."""
    sandbox = kit_sandbox
    calls = sandbox.calls
    await _channel_ready(sandbox)
    secret = "bf-chan-" + uuid.uuid4().hex
    running = asyncio.ensure_future(
        sandbox.exec(
            'sleep 2; printf %s "$BF_CHANNEL_SECRET"',
            user="root",
            env={"BF_CHANNEL_SECRET": secret},
        )
    )
    await asyncio.sleep(1)
    inside = _dx(kit_box, "cat /proc/[0-9]*/cmdline 2>/dev/null | tr '\\0' ' '")
    host = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
    result = await running
    assert result.stdout == secret
    assert secret not in inside.stdout
    assert secret not in host.stdout
    # The channel was there to see.
    assert any(
        mark in inside.stdout for mark in (*HELPER_MARKS.values(), "__bf_warm_n")
    )
    assert not any(secret in " ".join(argv) for argv in calls)


@pytest.mark.asyncio
async def test_timeout_dead_channel_and_ctrl_c_on_a_real_sandbox(kit_sandbox, kit_box):
    """A timeout says what it always said and leaves the command running; a
    helper that dies fails its running command clearly and later commands
    take docker exec; a cancelled command leaves the channel working."""
    sandbox = kit_sandbox
    calls = sandbox.calls
    await _channel_ready(sandbox)

    with pytest.raises(RuntimeError, match="Command timed out after 1 seconds"):
        await sandbox.exec("sleep 2; : > /tmp/went-on", user="root", timeout_sec=1)
    await asyncio.sleep(1.5)
    assert _dx(kit_box, "test -e /tmp/went-on").returncode == 0

    task = asyncio.ensure_future(sandbox.exec("sleep 3", user="root"))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    n = len(calls)
    assert (await sandbox.exec("echo after", user="root")).stdout == "after\n"
    assert calls[n:] == []

    if _helper_kind(sandbox) is None:
        # The pre-started execs: each is its own docker exec, and none is
        # left in the sandbox once the channel closes.
        await sandbox._close_exec_channels()
        for _ in range(40):
            left = _dx(
                kit_box, "grep -l __bf_warm_[n] /proc/[0-9]*/cmdline 2>/dev/null"
            )
            if not left.stdout.strip():
                break
            await asyncio.sleep(0.1)
        assert not left.stdout.strip()
        return

    running = asyncio.ensure_future(sandbox.exec("sleep 5", user="root"))
    await asyncio.sleep(0.5)
    helper = _helper_pids(sandbox, kit_box)
    assert helper
    _dx(kit_box, f"kill -9 {' '.join(helper)}")
    with pytest.raises(RuntimeError, match="may or may not have run"):
        await asyncio.wait_for(running, timeout=10)
    n = len(calls)
    assert (await sandbox.exec("echo fallback", user="root")).stdout == "fallback\n"
    # That step took its own docker exec; a new helper starts beside it.
    assert any(c[0] == "exec" and "fallback" in c[-1] for c in calls[n:])


# What a task's own root step may do to clean up: kill every process whose
# command line says python (as ``pkill -f python`` does, skipping itself).
# The pattern is built in pieces so this step's own command line lacks it.
PATTERN_KILL = (
    "p=pyt; p=${p}hon; for d in /proc/[0-9]*; do n=${d#/proc/}; "
    '[ "$n" = $$ ] && continue; '
    "c=$(tr '\\000' ' ' < $d/cmdline 2>/dev/null) || continue; "
    'case "$c" in *"$p"*) kill -9 "$n" 2>/dev/null;; esac; done; echo done'
)


@pytest.mark.asyncio
async def test_a_root_step_runs_once_when_the_channel_transport_dies(
    kit_sandbox, kit_box, monkeypatch
):
    """Guards R3-M1 (review of 1eb1e5d6) on a real sandbox: the channel's
    transport dies (the relay drops the step's S frame and closes the host
    side) while the helper lives and runs the step. 1eb1e5d6 reported the
    step not started and ran it again through docker exec: two runs. The
    sandbox now kills the old helper, reads its journal, finds the step
    listed and fails it as "may or may not have run": one run. Plumbing
    then goes on, on a new helper."""
    sandbox = kit_sandbox
    if _helper_kind(sandbox) is None:
        pytest.skip("the pre-started execs have no helper")
    import benchflow.sandbox.docker as docker_module
    from benchflow.sandbox._exec_channel import ExecChannel

    relayed: list[bool] = []

    class FirstChannelRelayed(ExecChannel):
        def __init__(self, argv, *args, **kwargs):
            if not relayed:  # only the first helper; its restart is direct
                relayed.append(True)
                argv = [sys.executable, "-c", RELAY, *argv]
            super().__init__(argv, *args, **kwargs)

    monkeypatch.setattr(docker_module, "ExecChannel", FirstChannelRelayed)
    await _channel_ready(sandbox)
    with pytest.raises(RuntimeError, match="may or may not have run"):
        await sandbox.exec("echo ran >> /tmp/runs; sleep 1", user="root")
    await asyncio.sleep(2.5)
    assert _dx(kit_box, "cat /tmp/runs").stdout == "ran\n"
    after = await sandbox.exec("echo after", user="root")
    assert after.return_code == 0 and after.stdout == "after\n"


@pytest.mark.asyncio
async def test_a_task_root_step_that_kills_by_pattern_keeps_its_result(
    kit_sandbox, kit_box
):
    """Guards PERF-B-M1 (review of fc2563ec): the task's own commands as
    root ran on the channel, whose helper is a python process in the
    container; a test.sh that kills python processes by pattern killed it
    and lost its own result (helper exit 137 against docker exec's rc 0).
    Task steps now take their own docker exec, as on e91a5791, and the
    channel keeps serving BenchFlow's plumbing.

    Guards R2-M1 (review of 73a53f94): the kill still takes the helper, and
    the very next plumbing step, sent before the host saw the helper's end,
    was failed as "may or may not have run". A step the helper never said it
    started now takes its own docker exec. Checked right after the kill,
    every round, past the restart limit."""
    sandbox = kit_sandbox
    for round in range(5):
        if round <= 3:  # the first helper, then its three restarts
            await _channel_ready(sandbox)
        with sandbox.task_command():
            result = await sandbox.exec(PATTERN_KILL, user="root")
        assert result.return_code == 0, result.stdout
        assert (result.stdout or "").strip().endswith("done")
        # No sleep: the next plumbing step goes out while the helper's end
        # may not have reached the host yet.
        after = await sandbox.exec(f"echo plumbing-{round}", user="root")
        assert after.return_code == 0 and after.stdout == f"plumbing-{round}\n"


@pytest.mark.asyncio
async def test_a_step_the_helper_gets_after_its_deadline_never_runs(
    kit_sandbox, kit_box
):
    """Guards PERF-B-L2 (review of fc2563ec): a step whose caller had timed
    out could still be read and started by a busy helper later; a per-call
    docker exec that timed out before it began never runs. The step now
    carries its deadline and the helper refuses it after that."""
    sandbox = kit_sandbox
    if _helper_kind(sandbox) is None:
        pytest.skip("the pre-started execs have no helper")
    await _channel_ready(sandbox)
    helper = _helper_pids(sandbox, kit_box)
    assert helper
    _dx(kit_box, f"kill -STOP {' '.join(helper)}")  # a helper busy elsewhere
    try:
        with pytest.raises(RuntimeError, match="Command timed out after 1 seconds"):
            await sandbox.exec(": > /tmp/late-step", user="root", timeout_sec=1)
    finally:
        _dx(kit_box, f"kill -CONT {' '.join(helper)}")
    await asyncio.sleep(1.5)
    assert _dx(kit_box, "test -e /tmp/late-step").returncode != 0
    assert (await sandbox.exec("echo after", user="root")).stdout == "after\n"
