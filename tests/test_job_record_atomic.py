"""evaluation.json is replaced whole or left as it was.

Guards the audit fix on branch ``audit-atomic-job-record``: the
job record was written in place, twice per run (at the start and again once
the plan is known). A run killed during the second write left a truncated
evaluation.json, which ``Evaluation.resume`` cannot read and the settings
comparison of a named rerun silently ignores. summary.json, config.json and
result.json were already written through ``write_text_atomic``.
"""

from __future__ import annotations

import json
import os

import pytest

from benchflow.evaluation import EVALUATION_RECORD, Evaluation, EvaluationConfig


def test_a_failed_rewrite_leaves_the_earlier_job_record(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="job",
        config=EvaluationConfig(agent="oracle"),
    )
    evaluation._write_evaluation_record()
    path = evaluation.job_dir / EVALUATION_RECORD
    first = path.read_text()
    assert "planned_tasks" not in json.loads(first)

    def killed(*_args, **_kwargs):
        raise OSError("killed while the record was being replaced")

    evaluation._planned_tasks = {"task-a"}
    monkeypatch.setattr(os, "replace", killed)
    with pytest.raises(OSError, match="killed while"):
        evaluation._write_evaluation_record()

    assert path.read_text() == first
    assert sorted(p.name for p in evaluation.job_dir.iterdir()) == [EVALUATION_RECORD]


def test_the_rewrite_names_the_planned_tasks(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="job",
        config=EvaluationConfig(agent="oracle"),
    )
    evaluation._write_evaluation_record()
    evaluation._planned_tasks = {"task-b", "task-a"}
    evaluation._write_evaluation_record()

    record = json.loads((evaluation.job_dir / EVALUATION_RECORD).read_text())
    assert record["planned_tasks"] == ["task-a", "task-b"]
