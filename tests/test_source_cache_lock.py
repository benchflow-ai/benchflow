"""A source cache lock whose owner is dead is taken over (PLAN P8).

The lock on a cached checkout was a file's existence, so a run stopped before
it removed the file (Ctrl-C during a clone, a kill, a crash) left a lock with
no owner and every later run on that cache waited 300 s and failed (seen in
the Rust round's parity run, rust-break-1011 E4-breadth section 4). It is an
``flock`` now, which the kernel releases when the holder exits.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos

# A child that takes the lock through the module, says so, then either exits
# without letting go (os._exit skips every finally, as a kill would) or waits
# for its stdin to close and lets go normally.
_CHILD = r"""
import os, sys
from pathlib import Path
from benchflow._utils import benchmark_repos
benchmark_repos._cache_dir = lambda: Path(sys.argv[1])
with benchmark_repos._repo_cache_lock("acme", "tasks"):
    print("held", flush=True)
    if sys.argv[2] == "vanish":
        os._exit(0)
    sys.stdin.read()
"""


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_repos, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(benchmark_repos, "_LOCK_TIMEOUT_SEC", 2.0)
    return tmp_path


def _lock_file(cache: Path) -> Path:
    return cache / "acme" / ".tasks.lock"


def _child(cache: Path, mode: str) -> subprocess.Popen:
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(cache), mode],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "held"
    return child


def test_a_lock_file_left_by_a_dead_run_is_taken_over(cache):
    # What an older BenchFlow leaves when Ctrl-C stops it before its clean-up:
    # the file, naming a pid that no longer runs.
    gone = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    lock = _lock_file(cache)
    lock.parent.mkdir(parents=True)
    lock.write_text(gone.stdout.strip())
    started = time.monotonic()
    with benchmark_repos._repo_cache_lock("acme", "tasks"):
        assert time.monotonic() - started < 1.0
    assert not lock.exists()


def test_a_run_that_vanished_holding_the_lock_does_not_block_the_next(cache):
    child = _child(cache, "vanish")
    assert child.wait(timeout=30) == 0
    assert _lock_file(cache).exists()  # its clean-up never ran
    started = time.monotonic()
    with benchmark_repos._repo_cache_lock("acme", "tasks"):
        assert time.monotonic() - started < 1.0
    assert not _lock_file(cache).exists()


def test_a_live_holder_still_excludes_and_is_named(cache, monkeypatch):
    monkeypatch.setattr(benchmark_repos, "_LOCK_TIMEOUT_SEC", 0.5)
    child = _child(cache, "wait")
    try:
        with (
            pytest.raises(TimeoutError, match=rf"held by pid {child.pid}\)"),
            benchmark_repos._repo_cache_lock("acme", "tasks"),
        ):
            pass
    finally:
        assert child.stdin is not None
        child.stdin.close()
        assert child.wait(timeout=30) == 0
    with benchmark_repos._repo_cache_lock("acme", "tasks"):
        pass


def test_threads_of_one_run_take_turns(cache, monkeypatch):
    monkeypatch.setattr(benchmark_repos, "_LOCK_TIMEOUT_SEC", 60.0)
    inside = 0
    most = 0
    guard = threading.Lock()

    def work() -> None:
        nonlocal inside, most
        for _ in range(5):
            with benchmark_repos._repo_cache_lock("acme", "tasks"):
                with guard:
                    inside += 1
                    most = max(most, inside)
                time.sleep(0.01)
                with guard:
                    inside -= 1

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert most == 1
    assert not _lock_file(cache).exists()
