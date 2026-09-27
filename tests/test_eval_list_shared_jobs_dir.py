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
