"""The host-service teardown reaps the whole process group (uvicorn children)."""

from __future__ import annotations

import os
import signal
import subprocess
import time

import pytest

from benchflow.arena.bootstrap import _kill_service_group


def test_kill_service_group_reaps_backgrounded_children():
    # A launcher that forks a child into the same session (like `uv run` → uvicorn
    # workers). Signalling only the launcher would orphan the child — the bug.
    proc = subprocess.Popen(
        ["bash", "-c", "sleep 30 & echo $! ; wait"],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    child_pid = int(proc.stdout.readline())

    _kill_service_group(proc, signal.SIGKILL)
    proc.wait(timeout=5)
    time.sleep(0.2)

    with pytest.raises(ProcessLookupError):  # the child died with the group
        os.kill(child_pid, 0)


def test_kill_service_group_on_dead_proc_is_noop():
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait(timeout=5)
    _kill_service_group(proc, signal.SIGTERM)  # must not raise
