"""Zero-task selection must exit non-zero and never write a misleading
summary.json (#407).

Before the fix, ``bench eval create --include not-a-real-task`` exited 0
and published a ``total: 0, passed: 0, score: "0.0%"`` artifact that
downstream dashboards could ingest as evidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import (
    EmptyTaskSelectionError,
    Evaluation,
    EvaluationConfig,
)


def _make_tasks(tmp_path: Path, names=("task-a", "task-b")) -> Path:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    for name in names:
        d = tasks_dir / name
        d.mkdir()
        (d / "task.toml").write_text('version = "1.0"\n')
    return tasks_dir


@pytest.mark.asyncio
async def test_evaluation_run_rejects_empty_selection(tmp_path):
    """Direct API: Evaluation.run() raises when include/exclude eliminate all tasks."""
    tasks_dir = _make_tasks(tmp_path)
    cfg = EvaluationConfig(include_tasks={"definitely-not-a-real-task"})
    job = Evaluation(
        tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg, job_name="empty"
    )

    with pytest.raises(EmptyTaskSelectionError) as excinfo:
        await job.run()

    msg = str(excinfo.value)
    assert "definitely-not-a-real-task" in msg
    # The guard must fail BEFORE writing any 0/0 summary.
    assert not (tmp_path / "jobs" / "empty" / "summary.json").exists()
    assert not (tmp_path / "jobs" / "summary.json").exists()


@pytest.mark.asyncio
async def test_evaluation_run_rejects_empty_selection_via_exclude(tmp_path):
    """Exclude-everything also trips the guard."""
    tasks_dir = _make_tasks(tmp_path)
    cfg = EvaluationConfig(exclude_tasks={"task-a", "task-b"})
    job = Evaluation(
        tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg, job_name="empty"
    )

    with pytest.raises(EmptyTaskSelectionError):
        await job.run()


@pytest.mark.asyncio
async def test_evaluation_run_rejects_empty_tasks_dir(tmp_path):
    """An empty tasks directory (no task.toml found) also trips the guard.

    Without this, a typo in --tasks-dir would silently publish a 0/0
    summary.json — same release-evidence footgun as #407.
    """
    tasks_dir = tmp_path / "empty"
    tasks_dir.mkdir()
    job = Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", job_name="empty")

    with pytest.raises(EmptyTaskSelectionError):
        await job.run()


def test_cli_zero_task_selection_exits_nonzero_no_summary(tmp_path):
    """Repro from #407: --include not-a-real-task exits non-zero, no 0/0 summary."""
    tasks_dir = _make_tasks(tmp_path)
    jobs_dir = tmp_path / "jobs"

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(tasks_dir),
            "--include",
            "definitely-not-a-real-task",
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(jobs_dir),
            "--concurrency",
            "1",
            "--agent-idle-timeout",
            "0",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "No tasks selected" in result.stderr
    # Critically: no 0/0 summary.json must exist.
    assert not (jobs_dir / "summary.json").exists(), (
        "zero-task selection must not publish a 0/0 summary.json — that "
        "would surface as a successful eval in downstream dashboards (#407)."
    )
    for child in jobs_dir.glob("*/summary.json") if jobs_dir.exists() else []:
        raise AssertionError(f"unexpected job summary written: {child}")
