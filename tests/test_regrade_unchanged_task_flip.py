"""A regrade whose task did not change must not pass off a flip as a verdict.

Guards ``bench eval regrade``. Regrading an oracle job with the unchanged
tasks could report ``N changed (N pass->fail)``: the verifiers read state the
frozen workspace does not hold (a package the solver installed system-wide, a
running web server, a git server outside the workspace). The row now says the task is unchanged and why the
verdict can still differ, and the CLI marks it.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from benchflow._utils.task_authoring import task_digest
from benchflow.eval_regrade import aregrade
from tests.test_regrade import _runner, _task, _trial


@pytest.fixture
async def unchanged_job(tmp_path):
    tasks = tmp_path / "tasks"
    task = _task(tasks, "svc", "#!/bin/bash\nexit 0\n")
    job = tmp_path / "jobs" / "run1"
    trial = await _trial(job, "svc__a", "svc", 1.0, frozen=True, tmp=tmp_path)
    config = json.loads((trial / "config.json").read_text())
    config["task_digest"] = task_digest(task)
    (trial / "config.json").write_text(json.dumps(config))
    return job, tasks


@pytest.mark.asyncio
async def test_a_flip_with_an_unchanged_task_is_marked(unchanged_job):
    job, tasks = unchanged_job
    summary = await aregrade(
        job, tasks_dir=tasks, runner=_runner({"svc__a": {"reward": 0.0}}, [])
    )
    [row] = summary.trials
    assert row.change == "pass->fail"
    assert row.task_changed is False
    assert "task is unchanged" in (row.reason or "")
    assert "outside the frozen workspace" in row.reason
    written = json.loads((job / "regrade-summary.json").read_text())
    assert written["trials"][0]["task_changed"] is False


@pytest.mark.asyncio
async def test_the_same_verdict_with_an_unchanged_task_carries_no_warning(
    unchanged_job,
):
    job, tasks = unchanged_job
    summary = await aregrade(
        job, tasks_dir=tasks, runner=_runner({"svc__a": {"reward": 1.0}}, [])
    )
    [row] = summary.trials
    assert row.change == "same" and row.reason is None


def test_the_cli_marks_a_flip_with_an_unchanged_task(unchanged_job, monkeypatch):
    import asyncio as _asyncio

    from benchflow import eval_regrade
    from benchflow.cli.main import app

    job, tasks = unchanged_job
    real = eval_regrade.aregrade

    def fake_regrade(path, **kwargs):
        runner = _runner({"svc__a": {"reward": 0.0}}, [])
        return _asyncio.run(real(path, runner=runner, **kwargs))

    monkeypatch.setattr("benchflow.eval_regrade.regrade", fake_regrade)
    result = CliRunner().invoke(
        app,
        ["eval", "regrade", str(job), "--tasks-dir", str(tasks)],
        env={"COLUMNS": "250"},
    )
    assert result.exit_code == 0, result.output
    assert "task unchanged" in result.output
