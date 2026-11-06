"""Guards the cfca0513/362be116 merge: scoring shares owned job locks."""

import json
import socket
from contextlib import contextmanager
from pathlib import Path

import pytest

from benchflow.evaluation import JOB_LOCK, Evaluation
from benchflow.review.resume import ReviewResumeError, _scoring_budget


def _record(config):
    """A job record with the keys every one has (a record without them marks
    no job)."""
    return json.dumps(
        {"schema_version": 1, "tasks_dir": "/t", "job_name": "job", "config": config}
    )


@contextmanager
def _holding(kind, job):
    if kind == "scoring":
        with _scoring_budget(job._jobs_dir / job._job_name, ...):
            yield
    else:
        ownership = job._acquire_job_lock()
        try:
            yield
        finally:
            # Let the regression run behaviorally on the pre-merge Path API.
            if isinstance(ownership, Path):
                ownership.unlink(missing_ok=True)
            else:
                ownership.release()


@pytest.fixture
def job(tmp_path):
    job = object.__new__(Evaluation)
    job._jobs_dir = tmp_path
    job._job_name = "job"
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "evaluation.json").write_text(_record({}))
    return job


@pytest.mark.parametrize("owner", ["evaluation", "scoring"])
@pytest.mark.parametrize("contender", ["evaluation", "scoring"])
@pytest.mark.parametrize("claim", ["", "stale", "missing"])
def test_owned_lock_excludes_both_entry_points(job, owner, contender, claim):
    """cfca0513 ownership survives absent or incomplete process metadata."""
    path = job._jobs_dir / job._job_name / JOB_LOCK
    with _holding(owner, job):
        if claim == "missing":
            path.unlink()
        else:
            path.write_text(
                ""
                if claim == ""
                else json.dumps({"pid": 999_999_999, "host": socket.gethostname()})
            )
        with (
            pytest.raises(RuntimeError, match="already running"),
            _holding(contender, job),
        ):
            pytest.fail("a live job acquired a second owner")
        # A failed contender must not release the original owner's guard.
        with (
            pytest.raises(RuntimeError, match="already running"),
            _holding(contender, job),
        ):
            pytest.fail("a refused contender dropped the live owner's guard")
    assert not path.exists()
    with _holding(contender, job):
        assert path.is_file()
    assert not path.exists()


@pytest.mark.parametrize("owner", ["evaluation", "scoring"])
@pytest.mark.parametrize("contender", ["evaluation", "scoring"])
@pytest.mark.parametrize("fail", [False, True])
def test_job_lock_released_after_success_or_exception(job, owner, contender, fail):
    """The 362be116 scoring boundary must close cfca0513's owned descriptor."""
    path = job._jobs_dir / job._job_name / JOB_LOCK
    try:
        with _holding(owner, job):
            if fail:
                raise ValueError("scoring failed")
    except ValueError:
        assert fail
    assert not path.exists()
    with _holding(contender, job):
        assert path.is_file()
    assert not path.exists()


@pytest.mark.parametrize("config", [None, {"budget": {"max_tokens": 0}}])
def test_scoring_policy_failure_releases_owned_lock(job, config):
    """Admission failure in 362be116 must leave cfca0513's lock available."""
    directory = job._jobs_dir / job._job_name
    record = directory / "evaluation.json"
    record.write_text(_record(config))
    with pytest.raises(ReviewResumeError), _holding("scoring", job):
        pytest.fail("invalid saved policy was accepted")
    assert not (directory / JOB_LOCK).exists()
    record.write_text(_record({}))
    with _holding("evaluation", job):
        assert (directory / JOB_LOCK).is_file()
