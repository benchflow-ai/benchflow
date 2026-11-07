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

    assert find_trials(job) == [scored, other]
