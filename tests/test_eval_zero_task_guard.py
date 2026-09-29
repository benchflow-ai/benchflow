"""Zero-task selection must exit non-zero and never write a misleading
summary.json (#407).

Before the fix, ``bench eval run --include not-a-real-task`` exited 0
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


def _make_task_md_without_environment(tmp_path: Path) -> Path:
    """A task.md task whose environment/ directory is missing."""
    import shutil

    from benchflow.doctor_smoke import BUNDLED_TASK_DIR

    task = tmp_path / "no-env"
    shutil.copytree(BUNDLED_TASK_DIR, task)
    shutil.rmtree(task / "environment")
    return task


@pytest.mark.asyncio
async def test_single_unrunnable_task_names_the_reason(tmp_path):
    """A single task.md task that fails structural checks used to be reported
    as "No tasks selected after include/exclude filtering" although no filter
    was given, with no hint why. The message now names
    what `bench tasks check` finds."""
    task = _make_task_md_without_environment(tmp_path)
    job = Evaluation(tasks_dir=task, jobs_dir=tmp_path / "jobs", job_name="empty")

    with pytest.raises(EmptyTaskSelectionError) as excinfo:
        await job.run()

    msg = str(excinfo.value)
    assert "include/exclude" not in msg
    assert "not a runnable task" in msg
    assert "Missing required directory: environment/" in msg
    assert f"bench tasks check {task}" in msg
    assert not (tmp_path / "jobs" / "empty" / "summary.json").exists()


@pytest.mark.asyncio
async def test_batch_of_unrunnable_tasks_counts_them(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    _make_task_md_without_environment(tasks_dir)
    job = Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", job_name="e")

    with pytest.raises(EmptyTaskSelectionError) as excinfo:
        await job.run()

    msg = str(excinfo.value)
    assert "include/exclude" not in msg
    assert "1 subdirectory has a task file but is not a runnable task" in msg
    assert "no-env: Missing required directory: environment/" in msg


def test_cli_single_unrunnable_task_explains(tmp_path):
    task = _make_task_md_without_environment(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(task),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )
    assert result.exit_code == 1, result.output
    stderr = " ".join(result.stderr.split())  # Rich wraps at the test width
    assert "is not a runnable task" in stderr
    assert "Missing required directory: environment/" in stderr
