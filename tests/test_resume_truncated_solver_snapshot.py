"""A resume refuses a job whose solver snapshot cannot be read.

Guards the audit fix on branch ``audit-truncated-solver-snapshot``:
``_get_completed_tasks`` read ``solver-complete.json`` and ``solver.json`` with
a bare ``json.loads``, so a truncated file (a partial copy or restore, a disk
error) ended the resume with a JSONDecodeError traceback that named no file.
The refusal now names the file. It is not skipped: a skipped snapshot would run
that trial's solver again.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.errors import UserError
from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    UnreadableSolverSnapshotError,
)
from tests.test_resume_refuses_changed_settings import _fake_rollouts
from tests.test_resume_settings_match import _fake_runner, _first_job

SNAPSHOT = {
    "task_name": "task",
    "purpose": "task",
    "rewards": None,
    "verifier_error": "[solver-preserved] verifier pending",
    "execution_stage": "solver_complete",
}


def _job(tmp_path, name: str, text: str) -> Evaluation:
    trial = tmp_path / "jobs" / "job" / "task__a"
    trial.mkdir(parents=True)
    (trial / name).write_text(text)
    job = Evaluation.__new__(Evaluation)
    job._jobs_dir = tmp_path / "jobs"
    job._job_name = "job"
    job._config = EvaluationConfig()
    return job


@pytest.mark.parametrize("name", ["solver-complete.json", "solver.json"])
@pytest.mark.parametrize("text", ['{"task_name": "ta', "", "[]"])
def test_a_truncated_snapshot_stops_the_resume_and_names_the_file(tmp_path, name, text):
    job = _job(tmp_path, name, text)

    with pytest.raises(UnreadableSolverSnapshotError) as refused:
        job._get_completed_tasks()

    assert isinstance(refused.value, UserError)
    message = str(refused.value)
    assert str(tmp_path / "jobs" / "job" / "task__a" / name) in message
    assert "truncated or unreadable" in message
    assert "task__a" in refused.value.hint


def test_a_readable_snapshot_still_keeps_its_solver(tmp_path):
    job = _job(tmp_path, "solver-complete.json", json.dumps(SNAPSHOT))

    assert "task" in job._get_completed_tasks()


# --- through Evaluation.resume() and `bench eval resume` -----------------------

TRUNCATED = '{"task_name": "ta'


def _resumable_job(tmp_path: Path, name: str, *, keep_result: bool = False) -> Path:
    """A two-task job whose task-0 trial holds a truncated ``name`` (beside its
    result.json, or with none), and whose task-1 trial waits for its review."""
    job_dir = _first_job(tmp_path, job_name="j")
    broken = job_dir / "task-0__abcd1234"
    if not keep_result:
        (broken / "result.json").unlink()
    (broken / name).write_text(TRUNCATED)
    pending = job_dir / "task-1__abcd1234"
    (pending / "result.json").unlink()
    (pending / "solver.json").write_text(
        json.dumps({**SNAPSHOT, "task_name": "task-1"})
    )
    config = json.loads((pending / "config.json").read_text())
    config["review"] = {"rubric_sha256": "0" * 64}
    (pending / "config.json").write_text(json.dumps(config))
    return job_dir


def _spy_reviews(monkeypatch) -> list[str]:
    reviewed: list[str] = []

    async def spy(path, **_kwargs):
        reviewed.append(Path(path).name)
        return {}

    monkeypatch.setattr("benchflow.review.resume.resume_review", spy)
    return reviewed


@pytest.mark.parametrize(
    ("name", "keep_result"),
    [("solver.json", False), ("solver-complete.json", False), ("solver.json", True)],
    ids=["solver", "solver-complete", "solver-beside-result"],
)
def test_evaluation_resume_refuses_a_truncated_snapshot_before_any_review(
    tmp_path, monkeypatch, name, keep_result
):
    """Review of PR #131, 2026-11-13: a real resume runs resume_pending_reviews
    before _get_completed_tasks, and it reads every trial's solver.json, so a
    truncated one ended the resume there as a ReviewResumeError (no UserError,
    no next step), and a truncated solver-complete.json was refused only after
    another trial's pending review had started. Evaluation.resume() now refuses
    each before any review or solver runs."""
    job_dir = _resumable_job(tmp_path, name, keep_result=keep_result)
    reviewed = _spy_reviews(monkeypatch)
    job = Evaluation.resume(job_dir)
    job._run_single_task = AsyncMock(side_effect=_fake_runner(job))

    with pytest.raises(UnreadableSolverSnapshotError) as refused:
        asyncio.run(job.run())

    assert isinstance(refused.value, UserError)
    message = str(refused.value)
    assert (
        f"{job_dir / 'task-0__abcd1234' / name} is truncated or unreadable" in message
    )
    assert "task-0__abcd1234" in refused.value.hint
    assert reviewed == []
    job._run_single_task.assert_not_called()


def test_bench_eval_resume_refuses_a_truncated_snapshot_without_a_crash_log(
    tmp_path, monkeypatch, capsys
):
    """Review of PR #131, 2026-11-13: `bench eval resume` on a job with a
    truncated solver.json printed the ReviewResumeError's text with the next
    step of a failed run (`bench check <task folder>` validates the tasks). It
    now prints the refusal with its own next step, exits 1 as for any
    UserError, and writes no crash log."""
    from benchflow.cli._errors import run_cli
    from benchflow.cli.main import app

    job_dir = _resumable_job(tmp_path, "solver.json")
    ran = _fake_rollouts(monkeypatch)
    reviewed = _spy_reviews(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BENCHFLOW_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exited:
        run_cli(app, ["eval", "resume", str(job_dir)])

    err = " ".join(capsys.readouterr().err.split())
    assert exited.value.code == 1
    snapshot = job_dir / "task-0__abcd1234" / "solver.json"
    assert f"{snapshot} is truncated or unreadable" in err
    assert (
        "next: restore solver.json from a copy of the job, or move the trial "
        "folder task-0__abcd1234 out of the job to run its task again"
    ) in err
    assert "unexpected error" not in err
    assert not list((tmp_path / "logs").glob("*"))
    assert ran == [] and reviewed == []
