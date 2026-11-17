"""A job lock a forked child still holds for a moment is not a running job.

``test_budget_scoring_resume``'s external-cancel cases failed about 1 run
in 6 (on the base as on the branch) with "Job ... is already running;
wait for its lock" when the test resumed the job right after cancelling it.
Instrumented: at the refusal no JobLock was held, and the only other holder
of the lock file was a child of the test process that had not exec'd yet;
it became ``docker info --format {{.ID}}``, which
``sandbox._docker_subnets._read_daemon_id`` runs with ``subprocess.run`` on
the event loop's executor thread (the subnet prefetch every Docker job
starts). A fork copies the lock's open file, so the child held the flock
from its fork until its exec, milliseconds after the cancelled run's
``release()``; the resumed run's ``flock`` landed in that gap. Releasing
now unlocks the open file before closing it, which frees the lock whatever
copies a child holds; a refused lock is still tried again for
:data:`TRANSIENT_HOLD_SEC`, for a holder that died between its fork and its
unlock.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from benchflow._utils.job_lock import TRANSIENT_HOLD_SEC, JobLock, JobLockedError

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="fork and flock are POSIX"
)


def test_a_released_lock_is_free_while_a_forked_child_still_holds_it(tmp_path):
    """The child keeps its copy of the lock's open file for a second, far
    past any retry: the release unlocks the open file itself, so the lock
    is free at once (the review of 0a3a5fab8: closing the fd alone left it
    held until the child exited)."""
    job = tmp_path / "jobs" / "2026-11-13__12-15-03"
    job.mkdir(parents=True)
    held = JobLock.acquire(job / ".evaluation.lock")
    pid = os.fork()
    if pid == 0:  # a child between its fork and its exec, slow to exec
        time.sleep(1.0)
        os._exit(0)
    try:
        held.release()
        started = time.monotonic()
        again = JobLock.acquire(job / ".evaluation.lock")
        assert time.monotonic() - started < TRANSIENT_HOLD_SEC
        again.release()
    finally:
        os.waitpid(pid, 0)


def test_a_job_resumed_beside_a_thread_starting_programs_is_never_refused(tmp_path):
    """The reviewer's reproduction, bounded: a worker thread keeps starting
    programs (as the Docker subnet check does), and the job is released and
    taken again; no take may be refused (28 of 3000 were before)."""
    import subprocess
    import threading

    job = tmp_path / "jobs" / "2026-11-13__12-15-03"
    job.mkdir(parents=True)
    stop = threading.Event()

    def spawn() -> None:
        while not stop.is_set():
            subprocess.run(["true"], check=False)

    worker = threading.Thread(target=spawn, daemon=True)
    worker.start()
    refused = 0
    try:
        lock = JobLock.acquire(job / ".evaluation.lock")
        for _ in range(300):
            lock.release()
            try:
                lock = JobLock.acquire(job / ".evaluation.lock")
            except JobLockedError:
                refused += 1
                lock = JobLock.acquire(job / ".evaluation.lock")
        lock.release()
    finally:
        stop.set()
        worker.join(5)
    assert refused == 0


def test_a_running_job_is_still_refused(tmp_path):
    job = tmp_path / "jobs" / "2026-11-13__12-15-03"
    job.mkdir(parents=True)
    held = JobLock.acquire(job / ".evaluation.lock")
    try:
        started = time.monotonic()
        with pytest.raises(JobLockedError, match="already running"):
            JobLock.acquire(job / ".evaluation.lock")
        assert time.monotonic() - started >= TRANSIENT_HOLD_SEC
    finally:
        held.release()
