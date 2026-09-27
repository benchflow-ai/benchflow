"""SIGTERM tears a run down instead of killing it.

CI cancel and job timeouts send SIGTERM. `bench eval run` died at once with
exit 143 and no teardown, leaving a STARTED Daytona sandbox that the cleanup
command could not reclaim. run_until_terminated turns the first SIGTERM into
the cancellation Ctrl-C gives: the running coroutine's finally blocks (the
rollouts' sandbox cleanup) run, then the process exits 143.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
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
