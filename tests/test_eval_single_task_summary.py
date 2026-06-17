"""Single-task `bench eval run` must publish a summary.json (#400).

Single-task and batch must share the same orchestration path so that
``bench eval list`` can find a real score for both.  Before the fix the
single-task branch went directly through ``SDK().run`` and skipped the
summary writer.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RunResult


def _write_task(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
        "[agent]\ntimeout_sec = 60\n[environment]\n"
    )


@pytest.mark.asyncio
async def test_evaluation_run_writes_summary_for_single_task(tmp_path):
    """Direct API: Evaluation.run() over a single-task layout writes summary.json."""
    task_dir = tmp_path / "lone-task"
    _write_task(task_dir)
    jobs_dir = tmp_path / "jobs"

    cfg = EvaluationConfig(retry=RetryConfig(max_retries=0))
    job = Evaluation(
        tasks_dir=task_dir, jobs_dir=jobs_dir, config=cfg, job_name="single-run"
    )
    job._sdk = AsyncMock()
    job._sdk.run = AsyncMock(
        return_value=RunResult(task_name="lone-task", rewards={"reward": 1.0})
    )

    result = await job.run()

    job_summary = jobs_dir / "single-run" / "summary.json"
    root_summary = jobs_dir / "summary.json"
    assert job_summary.exists(), "summary.json missing in job dir"
    assert root_summary.exists(), "summary.json missing at jobs_dir root"

    payload = json.loads(job_summary.read_text())
    assert payload["total"] == 1
    assert payload["passed"] == 1
    assert payload["score"] == "100.0%"
    assert payload["score_ratio"] == 1.0
    assert result.total == 1
    assert result.passed == 1


def test_cli_single_task_publishes_summary_for_eval_list(tmp_path, monkeypatch):
    """CLI: `bench eval run --tasks-dir <task>` writes summary.json (#400).

    With the unified path the single-task CLI invocation produces the same
    summary.json layout that `bench eval list` reads — fixing the
    "Summary = no summary" regression.
    """
    from benchflow.models import RolloutResult

    task_dir = tmp_path / "pass-task"
    _write_task(task_dir)

    async def fake_rollout_create(config):
        trial = AsyncMock()
        trial.run = AsyncMock(
            return_value=RolloutResult(task_name="pass-task", rewards={"reward": 1.0})
        )
        return trial

    monkeypatch.setattr("benchflow.rollout.Rollout.create", fake_rollout_create)

    jobs_dir = tmp_path / "jobs"
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--tasks-dir",
            str(task_dir),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(jobs_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    root_summary = jobs_dir / "summary.json"
    assert root_summary.exists(), (
        "single-task eval did not write summary.json — `bench eval list` "
        "would show this run as 'no summary' (#400)."
    )
    payload = json.loads(root_summary.read_text())
    assert payload["total"] == 1
    assert payload["passed"] == 1


def test_cli_single_task_no_longer_special_cases_sdk_run(tmp_path, monkeypatch):
    """Unification guard: the single-task CLI path goes through Evaluation.run.

    Pins the collapsed single-task / batch paths so a future refactor cannot
    silently reintroduce the bypass that caused #400/#401/#407.
    """
    task_dir = tmp_path / "task-x"
    _write_task(task_dir)

    eval_run_called = {"count": 0}

    async def fake_eval_run(self):
        from types import SimpleNamespace

        eval_run_called["count"] += 1
        return SimpleNamespace(
            passed=1, total=1, score=1.0, errored=0, verifier_errored=0
        )

    sdk_run_called = {"count": 0}

    async def fake_sdk_run(self, **_kwargs):
        sdk_run_called["count"] += 1
        return RunResult(task_name="task-x", rewards={"reward": 1.0})

    with (
        patch.object(Evaluation, "run", new=fake_eval_run),
        patch("benchflow.sdk.SDK.run", new=fake_sdk_run),
    ):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "create",
                "--tasks-dir",
                str(task_dir),
                "--agent",
                "oracle",
                "--jobs-dir",
                str(tmp_path / "jobs"),
            ],
        )

    assert result.exit_code == 0, result.stdout
    assert eval_run_called["count"] == 1
    assert sdk_run_called["count"] == 0, (
        "single-task CLI must route through Evaluation.run, not SDK().run "
        "(the latter bypasses summary.json publishing — #400)."
    )
