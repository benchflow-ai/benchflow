"""``bench eval regrade <job>`` on a batch job run with ``--tasks-dir``.

Guards two defects of regrading a batch job: a batch trial records only the task folder's name in
``config.json`` (``task_path: "fix-git"``), so without ``--tasks-dir`` every
trial was "not regradable: task folder unknown" although the job's own
``evaluation.json`` records the tasks folder; and attempts a retry replaced
(a sandbox that failed to start, then a scored retry) were listed as extra
trials, more rows than the job's summary, ``inspect`` and ``load_job`` count.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from benchflow import load_job
from benchflow._utils.attempt_decision import FINAL, RETRIED, record_decision
from benchflow.eval_regrade import find_trials, resolve_task


def _task(tasks: Path, name: str) -> Path:
    task = tasks / name
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Do it.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")
    return task


def _trial(job: Path, name: str, task: str, rewards, *, mtime: float) -> Path:
    trial = job / name
    trial.mkdir(parents=True)
    (trial / "config.json").write_text(json.dumps({"task_path": task}))
    result = trial / "result.json"
    result.write_text(
        json.dumps(
            {
                "task_name": task,
                "rewards": rewards,
                "error": None if rewards else "Sandbox startup failed",
                "error_category": None if rewards else "sandbox_setup",
            }
        )
    )
    os.utime(result, (mtime, mtime))
    return trial


def test_a_batch_trial_finds_its_task_through_the_jobs_evaluation(tmp_path: Path):
    tasks = tmp_path / "tasks"
    task = _task(tasks, "fix-git")
    job = tmp_path / "jobs" / "batch"
    trial = _trial(job, "fix-git__1", "fix-git", {"reward": 1.0}, mtime=1000)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "tasks_dir": str(tasks)})
    )

    found, why = resolve_task(trial, None)

    assert why is None
    assert found == task


def test_attempts_a_retry_replaced_are_not_listed_as_trials(tmp_path: Path):
    job = tmp_path / "jobs" / "batch"
    _trial(job, "db__a", "db", None, mtime=1000)
    _trial(job, "db__b", "db", None, mtime=1001)
    scored = _trial(job, "db__c", "db", {"reward": 1.0}, mtime=1002)
    other = _trial(job, "git__d", "git", {"reward": 0.0}, mtime=1003)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "tasks_dir": str(tmp_path / "tasks")})
    )

    assert find_trials(job) == [scored, other]


def test_every_sample_of_a_folder_that_is_no_job_is_listed(tmp_path: Path):
    """Re-review of PR #139, 2026-11-13: ``bf.run`` calls sharing a job name
    leave rollouts in a folder with no evaluation.json or summary.json, each
    a sample of its own, and ``bf.load_job`` counts every one. Regrade's
    rule for a job from before the attempt records dropped the unscored ones
    as attempts a scored retry replaced."""
    job = tmp_path / "jobs" / "shared"
    _trial(job, "db__a", "db", None, mtime=1000)
    _trial(job, "db__b", "db", None, mtime=1001)
    _trial(job, "db__c", "db", {"reward": 1.0}, mtime=1002)
    _trial(job, "git__d", "git", {"reward": 0.0}, mtime=1003)

    listed = [t.name for t in find_trials(job)]

    assert listed == ["db__a", "db__b", "db__c", "git__d"]
    assert sorted(t.path.name for t in load_job(job).trials) == listed


def test_the_runs_records_decide_in_a_folder_that_is_no_job(tmp_path: Path):
    """Re-review of PR #139, 2026-11-13: a folder that has the run's attempt
    records but no evaluation.json (trial folders copied out of their job)
    still lists the attempt the run kept, never one it retried."""
    job = tmp_path / "jobs" / "copied"
    retried = _trial(job, "db__a", "db", {"reward": 1.0}, mtime=1000)
    kept = _trial(job, "db__b", "db", None, mtime=1001)
    record_decision(retried, RETRIED, task="db", attempt=1)
    record_decision(kept, FINAL, task="db", attempt=2)

    assert find_trials(job) == [kept]
    assert [t.path for t in load_job(job).trials] == [kept]


def test_regrade_and_load_job_read_a_copied_job_alike(tmp_path: Path):
    """Re-review of PR #139, 2026-11-13: in a job folder copied without its
    evaluation.json, a task whose attempts have no record (the run stopped
    before it decided) is read by the rule for a job from before the
    records, by regrade as by ``bf.load_job``: the scored attempt."""
    job = tmp_path / "jobs" / "copied"
    retried = _trial(job, "db__a", "db", {"reward": 1.0}, mtime=1000)
    kept = _trial(job, "db__b", "db", None, mtime=1001)
    record_decision(retried, RETRIED, task="db", attempt=1)
    record_decision(kept, FINAL, task="db", attempt=2)
    _trial(job, "git__c", "git", None, mtime=1002)
    scored = _trial(job, "git__d", "git", {"reward": 0.0}, mtime=1003)

    assert find_trials(job) == [kept, scored]
    assert sorted(t.path for t in load_job(job).trials) == [kept, scored]
