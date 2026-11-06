"""SIGTERM tears a run down instead of killing it.

CI cancel and job timeouts send SIGTERM. `bench eval run` died at once with
exit 143 and no teardown, leaving a STARTED Daytona sandbox that the cleanup
command could not reclaim. run_until_terminated turns the first SIGTERM into
the cancellation Ctrl-C gives: the running coroutine's finally blocks (the
rollouts' sandbox cleanup) run, then the process exits 143.
"""

from __future__ import annotations

import contextlib
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

SCRIPT = """
import asyncio, sys
from pathlib import Path
from benchflow.cli._termination import run_until_terminated

marker = Path(sys.argv[1])

async def job():
    try:
        print("ready", flush=True)
        await asyncio.sleep(60)
    finally:
        await asyncio.sleep(0.2)  # async cleanup, like a sandbox delete
        marker.write_text("cleaned")

run_until_terminated(job())
print("not reached", flush=True)
"""


def _start(tmp_path: Path) -> tuple[subprocess.Popen, Path]:
    marker = tmp_path / "cleanup.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", SCRIPT, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc, marker


def test_sigterm_runs_cleanup_and_exits_143(tmp_path: Path) -> None:
    proc, marker = _start(tmp_path)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 143, err
    assert marker.read_text() == "cleaned"
    assert "not reached" not in out
    assert "Terminated" in err


def test_sigterm_is_honoured_even_when_sigint_is_ignored(tmp_path: Path) -> None:
    """A background job in a non-interactive shell starts with SIGINT ignored;
    SIGTERM must still tear down."""
    marker = tmp_path / "cleanup.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", SCRIPT, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=30)
    assert proc.returncode == 143
    assert marker.read_text() == "cleaned"


# SIGHUP-1: SIGHUP is handled as SIGTERM, with 129; an inherited
# ignored SIGHUP (nohup) is respected. The real-Docker check is
# tests/test_sighup_container.py.


def _default_signals() -> None:
    # The child starts as from an interactive shell, however the suite was
    # launched (an & job ignores SIGINT, nohup SIGHUP).
    for sig in (signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, signal.SIG_DFL)


def _start_with(
    tmp_path: Path, preexec_fn: Callable[[], object] = _default_signals
) -> tuple[subprocess.Popen[str], Path]:
    marker = tmp_path / "cleanup.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", SCRIPT, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=preexec_fn,
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    time.sleep(0.1)
    return proc, marker


def test_sighup_runs_cleanup_and_exits_129(tmp_path: Path) -> None:
    """On 5653de15 SIGHUP killed the process at once: no cleanup, -1."""
    proc, marker = _start_with(tmp_path)
    proc.send_signal(signal.SIGHUP)
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 129, err
    assert marker.read_text() == "cleaned"
    assert "not reached" not in out
    assert "Hung up (SIGHUP)" in err


def test_an_inherited_ignored_sighup_keeps_the_run_going(tmp_path: Path) -> None:
    """nohup: the run survives the hangup and still stops cleanly on SIGTERM."""
    proc, marker = _start_with(
        tmp_path, preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN)
    )
    proc.send_signal(signal.SIGHUP)
    time.sleep(1.0)
    assert proc.poll() is None
    assert not marker.exists()
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=30)
    assert proc.returncode == 143
    assert marker.read_text() == "cleaned"


SLOW_CLEANUP = """
import asyncio, sys
from pathlib import Path
from benchflow.cli._termination import run_until_terminated

marker = Path(sys.argv[1])

async def job():
    try:
        print("ready", flush=True)
        await asyncio.sleep(60)
    finally:
        await asyncio.sleep(1.0)  # a slow sandbox delete
        marker.write_text("cleaned")

run_until_terminated(job())
"""


def test_a_sigterm_during_ctrl_c_cleanup_does_not_cut_it_short(tmp_path: Path) -> None:
    """A sharded run's worker gets Ctrl-C from the terminal and then the
    parent's SIGTERM. On 5653de15 that SIGTERM raised out of the event loop
    and cancelled the cleanup half way: no marker."""
    marker = tmp_path / "cleanup.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", SLOW_CLEANUP, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_default_signals,
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    time.sleep(0.1)
    proc.send_signal(signal.SIGINT)
    time.sleep(0.3)
    proc.send_signal(signal.SIGTERM)
    _, err = proc.communicate(timeout=30)
    assert marker.read_text() == "cleaned", err
    assert proc.returncode == 130, err


HUNG_UP_TERMINAL = """
import asyncio, sys
from pathlib import Path
from benchflow.cli._termination import run_until_terminated

marker = Path(sys.argv[1])

async def job():
    try:
        print("ready", flush=True)
        await asyncio.sleep(60)
    finally:
        for i in range(200):  # cleanup that reports, as the live display does
            print("removing sandbox", i, flush=True)
            print("removing sandbox", i, file=sys.stderr, flush=True)
        await asyncio.sleep(0.2)
        marker.write_text("cleaned")

run_until_terminated(job())
"""


def test_a_closed_terminal_stops_the_run_cleanly(tmp_path: Path) -> None:
    """The terminal window closes: the kernel hangs the tty up and sends
    SIGHUP. The cleanup's writes to the hung-up terminal must not fail it."""
    import os
    import pty
    import select

    marker = tmp_path / "cleanup.txt"
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child
        _default_signals()
        os.execv(sys.executable, [sys.executable, "-c", HUNG_UP_TERMINAL, str(marker)])
    seen = b""
    deadline = time.monotonic() + 30
    while b"ready" not in seen and time.monotonic() < deadline:
        if select.select([fd], [], [], 0.5)[0]:
            seen += os.read(fd, 4096)
    assert b"ready" in seen, seen
    os.close(fd)  # the terminal goes away
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 129
    assert marker.read_text() == "cleaned"


SLOW_REPORTING_CLEANUP = """
import asyncio, sys
from pathlib import Path
from benchflow.cli._termination import run_until_terminated

marker = Path(sys.argv[1])

async def job():
    try:
        print("ready", flush=True)
        await asyncio.sleep(60)
    finally:
        for i in range(40):  # reports while it removes sandboxes, for 2 s
            print("removing sandbox", i)
            print("removing sandbox", i, file=sys.stderr)
            await asyncio.sleep(0.05)
        marker.write_text("cleaned")
        print("done")  # buffered: flushed at exit

run_until_terminated(job())
"""


def test_a_terminal_that_closes_after_the_hangup_signal_still_stops_cleanly(
    tmp_path: Path,
) -> None:
    """The shell dies first (the kernel sends SIGHUP to the foreground group
    while the terminal is still open), then the window closes during the
    cleanup. Checking only for a hung-up terminal at the signal left the
    streams on it, and the final flush failed: exit 120 (SIGHUP-1, runs 2, 3
    and 6 of the first real-Docker loop). The same sequence as
    tests/test_sighup_container.py, with a plain job instead of bench."""
    import json
    import os
    import pty
    import select

    from tests.test_sighup_container import _LEADER, _RECORDER

    marker = tmp_path / "cleanup.txt"
    rc_path = tmp_path / "rc.json"
    argv = [
        sys.executable,
        "-c",
        _LEADER,
        _RECORDER,
        str(rc_path),
        sys.executable,
        "-c",
        SLOW_REPORTING_CLEANUP,
        str(marker),
    ]
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the child, the session's leader
        _default_signals()
        os.execv(argv[0], argv)
    seen = b""
    deadline = time.monotonic() + 30
    while b"ready" not in seen and time.monotonic() < deadline:
        if select.select([fd], [], [], 0.5)[0]:
            seen += os.read(fd, 4096)
    assert b"ready" in seen, seen
    os.kill(pid, signal.SIGKILL)  # the shell dies: SIGHUP, terminal still open
    os.waitpid(pid, 0)
    time.sleep(0.5)
    os.close(fd)  # then the window closes, mid-cleanup
    deadline = time.monotonic() + 30
    while not rc_path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    time.sleep(0.2)
    assert json.loads(rc_path.read_text())["rc"] == 129
    assert marker.read_text() == "cleaned"


FAKE_WORKER = """
import os, signal, sys, time
from pathlib import Path

root = Path(sys.argv[1])
(root / "pid").write_text(str(os.getpid()))
stop = []
signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
print("ready", flush=True)
while not stop:
    time.sleep(0.05)
for i in range(4000):  # more than a pipe holds, while it cleans up
    print("cleanup line", i, flush=True)
time.sleep(0.3)
(root / "cleaned").write_text("yes")
print("worker done", flush=True)
sys.exit(143)
"""


async def test_a_stopped_sharded_run_stops_its_worker_and_waits(
    tmp_path: Path, monkeypatch
) -> None:
    """On 5653de15 the cancelled parent left its worker running, so the
    worker's sandboxes outlived the run."""
    import asyncio
    import os

    from benchflow import eval_sharding

    real = asyncio.create_subprocess_exec

    async def fake_exec(*args, **kwargs):
        return await real(sys.executable, "-c", FAKE_WORKER, str(tmp_path), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    log = tmp_path / "worker.log"
    task = asyncio.ensure_future(
        eval_sharding._run_worker_once(tmp_path / "payload.json", log)
    )
    try:
        for _ in range(200):
            if log.exists() and "ready" in log.read_text():
                break
            await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)
        assert (tmp_path / "cleaned").read_text() == "yes"
        assert "worker done" in log.read_text()
    finally:
        pid_file = tmp_path / "pid"
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)


# Round 2: which repeated signals
# force the exit.


def _slow(tmp_path: Path, cleanup_sec: float) -> tuple[subprocess.Popen[str], Path]:
    marker = tmp_path / "cleanup.txt"
    script = SLOW_CLEANUP.replace("asyncio.sleep(1.0)", f"asyncio.sleep({cleanup_sec})")
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=_default_signals,
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    time.sleep(0.1)
    return proc, marker


def _send(proc: subprocess.Popen[str], *signals: int) -> None:
    for sig in signals:
        proc.send_signal(sig)
        time.sleep(0.2)


def test_a_hangup_signalled_twice_and_a_sigterm_still_clean_up(tmp_path: Path) -> None:
    """M1: a closed terminal sends the foreground job SIGHUP twice (the
    kernel's and the shell's), and a sharded worker then gets its parent's
    SIGTERM: on 06cd177e the third signal forced the exit, no cleanup."""
    proc, marker = _slow(tmp_path, 1.5)
    _send(proc, signal.SIGHUP, signal.SIGHUP, signal.SIGTERM)
    _, err = proc.communicate(timeout=30)
    assert marker.read_text() == "cleaned", err
    assert proc.returncode == 129, err


def test_any_number_of_sighups_never_force_the_exit(tmp_path: Path) -> None:
    proc, marker = _slow(tmp_path, 1.5)
    _send(proc, *[signal.SIGHUP] * 5)
    _, err = proc.communicate(timeout=30)
    assert marker.read_text() == "cleaned", err
    assert proc.returncode == 129, err


def test_a_ctrl_c_during_a_sigterm_stop_does_not_cut_the_cleanup(
    tmp_path: Path,
) -> None:
    """L1: on 06cd177e asyncio's own Ctrl-C handler cancelled the cleanup
    again (exit 143, no marker)."""
    proc, marker = _slow(tmp_path, 1.5)
    _send(proc, signal.SIGTERM, signal.SIGINT)
    _, err = proc.communicate(timeout=30)
    assert marker.read_text() == "cleaned", err
    assert proc.returncode == 143, err


def test_ctrl_c_twice_still_stops_at_once(tmp_path: Path) -> None:
    """The second Ctrl-C forces the exit without waiting for the cleanup, as
    asyncio does."""
    proc, marker = _slow(tmp_path, 20)
    started = time.monotonic()
    _send(proc, signal.SIGINT, signal.SIGINT)
    _, err = proc.communicate(timeout=30)
    assert time.monotonic() - started < 10, err
    assert proc.returncode == 130, err
    assert not marker.exists()


def test_a_sigterm_twice_more_still_forces_the_exit(tmp_path: Path) -> None:
    proc, marker = _slow(tmp_path, 20)
    started = time.monotonic()
    _send(proc, signal.SIGTERM, signal.SIGTERM, signal.SIGTERM)
    proc.communicate(timeout=30)
    assert time.monotonic() - started < 10
    assert proc.returncode == 143
    assert not marker.exists()


PIPED_OUTPUT = """
import asyncio, sys
from pathlib import Path
from benchflow.cli._termination import run_until_terminated

marker = Path(sys.argv[1])

async def job():
    try:
        print("ready", flush=True)
        await asyncio.sleep(60)
    finally:
        await asyncio.sleep(0.5)
        marker.write_text("cleaned")
        print("removed the sandboxes")  # buffered: flushed at exit

run_until_terminated(job())
"""


def test_a_hangup_whose_tee_died_exits_129_not_120(tmp_path: Path) -> None:
    """L2: ``bench … | tee log`` in a terminal that closes: tee dies with the
    terminal, and on 06cd177e the final flush into its pipe failed (exit
    120). The cleanup ran either way."""
    marker = tmp_path / "cleanup.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", PIPED_OUTPUT, str(marker)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        preexec_fn=_default_signals,
    )
    assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    proc.stdout.close()  # tee is gone
    proc.send_signal(signal.SIGHUP)
    proc.wait(timeout=30)
    assert marker.read_text() == "cleaned"
    assert proc.returncode == 129
