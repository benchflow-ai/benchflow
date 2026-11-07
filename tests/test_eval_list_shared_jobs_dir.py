"""``bench eval list jobs/`` must list every job in a shared jobs folder.

Every finished job also writes a backward-compatible copy of its summary to the
jobs folder root (``Evaluation`` since the v0.5 merge 546aca59). ``eval list``
treated that root copy as "this folder is one job" (the single-job layout) and
printed a single row named after the folder, showing whichever job finished
last, so every other job in ``jobs/`` was invisible.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from benchflow.cli.main import app


def _summary(path: Path, total: int, passed: int) -> None:
    path.write_text(json.dumps({"total": total, "passed": passed, "score": "x"}))


def test_every_job_in_a_shared_jobs_folder_is_listed(tmp_path: Path):
    jobs = tmp_path / "jobs"
    for name, total, passed in (("alpha-oracle", 16, 16), ("beta-nop", 14, 0)):
        (jobs / name / "task-a__1234").mkdir(parents=True)
        _summary(jobs / name / "summary.json", total, passed)
    # the backward-compatible root copy of the last job to finish
    _summary(jobs / "summary.json", 14, 0)

    result = CliRunner().invoke(
        app, ["eval", "list", str(jobs)], env={"COLUMNS": "200"}
    )

    assert result.exit_code == 0
    assert "alpha-oracle" in result.stdout
    assert "16/16" in result.stdout
    assert "beta-nop" in result.stdout
    assert "0/14" in result.stdout


def test_a_resumed_0_7_job_reads_as_one_job(tmp_path: Path):
    """0.9.0 journeys: after 0.9 resumes a 0.7 job it keeps the old
    files in <job>/.pre-0.8/, summary.json included, and `bench results <job>
    --json` took the job for a folder of jobs and exited 2."""
    from benchflow.cli.new_verbs import _jobs_folder

    job = tmp_path / "jobs" / "oracle" / "2026-11-04__02-14-35"
    trial = job / "task-a__1234"
    trial.mkdir(parents=True)
    (trial / "config.json").write_text("{}")
    (trial / "result.json").write_text(json.dumps({"rewards": {"reward": 1.0}}))
    _summary(job / "summary.json", 1, 1)
    (job / ".pre-0.8").mkdir()
    _summary(job / ".pre-0.8" / "summary.json", 1, 1)

    assert _jobs_folder(str(job)) is False
