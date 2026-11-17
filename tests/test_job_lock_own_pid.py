"""A run can resume a job whose stale lock names the run's own pid.

Guards the audit fix on branch ``audit-job-lock-own-pid``: a run
killed with SIGKILL leaves ``.evaluation.lock`` with its pid. In a container
the next run often gets the same pid, and the lock check then refused the
resume as "already running in process <its own pid>", although the run already
held the job's flock.
"""

from __future__ import annotations

import json
import os
import socket

import pytest

from benchflow._utils.job_lock import JobLockedError
from benchflow.evaluation import JOB_LOCK, _acquire_job_lock


def _stale_lock(job, pid):
    job.mkdir(parents=True)
    (job / JOB_LOCK).write_text(
        json.dumps({"pid": pid, "host": socket.gethostname(), "started_at": "x"})
    )


def test_a_lock_left_under_this_runs_own_pid_is_stale(tmp_path):
    job = tmp_path / "jobs" / "job"
    _stale_lock(job, os.getpid())

    lock = _acquire_job_lock(job)
    try:
        assert json.loads((job / JOB_LOCK).read_text())["pid"] == os.getpid()
    finally:
        lock.release()


def test_a_lock_held_twice_in_one_process_is_still_refused(tmp_path):
    job = tmp_path / "jobs" / "job"
    first = _acquire_job_lock(job)
    try:
        with pytest.raises(JobLockedError, match="already running"):
            _acquire_job_lock(job)
    finally:
        first.release()


def test_another_live_process_on_this_host_still_holds_the_job(tmp_path):
    job = tmp_path / "jobs" / "job"
    _stale_lock(job, os.getppid())

    with pytest.raises(JobLockedError, match=f"process {os.getppid()}"):
        _acquire_job_lock(job)
