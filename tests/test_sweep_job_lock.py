"""Guards FF-sw-evaluation scheduling and lock races at 363d815b."""

import json
import multiprocessing
import socket
from pathlib import Path

import pytest

from benchflow.evaluation import Evaluation


def _job(root):
    job = object.__new__(Evaluation)
    job._jobs_dir = Path(root)
    job._job_name = "job"
    return job


def _holder(root, ready, finish):
    job = _job(root)
    job._acquire_job_lock()
    # Reproduce the publication window: metadata is empty while ownership lives.
    (Path(root) / "job" / ".evaluation.lock").write_text("")
    ready.set()
    finish.wait(15)


@pytest.mark.parametrize("stale", [False, True])
def test_incomplete_claim_never_allows_concurrent_owner(tmp_path, stale):
    if stale:
        path = tmp_path / "job" / ".evaluation.lock"
        path.parent.mkdir()
        path.write_text(json.dumps({"pid": 999_999_999, "host": socket.gethostname()}))
    ctx = multiprocessing.get_context("fork")
    ready, finish = ctx.Event(), ctx.Event()
    process = ctx.Process(target=_holder, args=(str(tmp_path), ready, finish))
    process.start()
    try:
        assert ready.wait(10)
        with pytest.raises(RuntimeError, match="already running"):
            _job(tmp_path)._acquire_job_lock()
    finally:
        finish.set()
        process.join(10)
    assert process.exitcode == 0


@pytest.mark.parametrize("claim", ["", "{", "null", "[]"])
def test_unreadable_legacy_claim_is_not_takeover_authority(tmp_path, claim):
    path = tmp_path / "job" / ".evaluation.lock"
    path.parent.mkdir()
    path.write_text(claim)
    with pytest.raises(RuntimeError, match="lock"):
        _job(tmp_path)._acquire_job_lock()
    assert path.read_text() == claim
