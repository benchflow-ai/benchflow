"""Resuming a job BenchFlow 0.7 made (dogfood p4-upgrader, 2026-10-18).

DF-p4-upgrader-1: a resumed 0.7 job's mean_reward went from 1.00 to 0.40 with
nothing re-run, and nothing said why. DF-3: resuming it rewrote its files in
place and no file recorded a version. DF-4: ``bench eval resume`` refused it
with a Python snippet and a ``next:`` that contradicted it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from benchflow.models import RolloutResult

# --- DF-1: the end-of-run line names both means -------------------------------


def _report(result) -> str:
    from rich.console import Console

    import benchflow.cli._shared as shared

    rec = Console(record=True, width=400, force_terminal=False)
    original = shared.console
    shared.console = rec
    try:
        shared._report_eval_result(result)
    finally:
        shared.console = original
    return rec.export_text()


def test_the_score_line_says_which_mean_is_0_7s() -> None:
    out = _report(
        SimpleNamespace(
            passed=2,
            failed=0,
            total=5,
            errored=1,
            verifier_errored=2,
            score=0.4,
            job_name="j",
            mean_reward=0.4,
            mean_reward_scored_only=1.0,
        )
    )
    assert "mean reward 0.40 over all 5 trials (unscored as 0)" in out
    assert "scored only: 2/2 (100.0%), mean reward 1.00 (0.7's mean_reward)" in out


# --- DF-1, DF-3: resuming a job 0.7 made --------------------------------------


def _pre_08_job(tmp_path: Path) -> tuple[Path, Path]:
    """A 0.7.11 job: two finished trials, its summary, no evaluation.json."""
    tasks = tmp_path / "tasks"
    for name in ("hello-file", "verifier-crash"):
        (tasks / name).mkdir(parents=True)
        (tasks / name / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    job = tmp_path / "jobs711" / "2026-10-18__22-15-47"
    rows = {
        "hello-file": {"rewards": {"reward": 1.0}, "verifier_error": None},
        "verifier-crash": {"rewards": None, "verifier_error": "exit 3, no reward"},
    }
    for name, row in rows.items():
        trial = job / f"{name}__1234abcd"
        trial.mkdir(parents=True)
        (trial / "config.json").write_text(
            json.dumps(
                {
                    "task_path": name,
                    "agent": "oracle",
                    "model": None,
                    "environment": "docker",
                    "concurrency": 2,
                }
            )
        )
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": name,
                    "rollout_name": trial.name,
                    "error": None,
                    "agent": "oracle",
                    **row,
                }
            )
        )
    (job / "summary.json").write_text(
        json.dumps({"job_name": job.name, "total": 2, "passed": 1, "mean_reward": 1.0})
    )
    (job / "results.jsonl").write_text('{"task": "hello-file", "reward": 1.0}\n')
    return tasks, job


def _evaluation(tasks: Path, job: Path, **kwargs):
    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

    evaluation = Evaluation(
        tasks_dir=tasks,
        jobs_dir=job.parent,
        config=EvaluationConfig(
            agent="oracle",
            environment="docker",
            concurrency=2,
            retry=RetryConfig(max_retries=0),
        ),
        job_name=job.name,
        **kwargs,
    )

    async def fake_run(task_path, _cfg):
        # Anything rerun crashes its verifier again, as in 0.7.
        trial = job / f"{task_path.name}__rerun000"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_path.name,
                    "rollout_name": trial.name,
                    "rewards": None,
                    "error": None,
                    "verifier_error": "exit 3, no reward",
                    "agent": "oracle",
                }
            )
        )
        (trial / "config.json").write_text(json.dumps({"agent": "oracle"}))
        return RolloutResult(
            task_name=task_path.name,
            rollout_name=trial.name,
            rewards=None,
            verifier_error="exit 3, no reward",
            agent="oracle",
            rollout_dir=trial,
        )

    evaluation._run_single_task = AsyncMock(side_effect=fake_run)
    return evaluation


def test_resuming_a_0_7_job_keeps_its_files_and_says_so(tmp_path, caplog) -> None:
    tasks, job = _pre_08_job(tmp_path)
    old_summary = (job / "summary.json").read_text()
    old_results = (job / "results.jsonl").read_text()

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(_evaluation(tasks, job, preflight=False).run())

    kept = job / ".pre-0.8"
    assert (kept / "summary.json").read_text() == old_summary
    assert (kept / "results.jsonl").read_text() == old_results
    assert f"Job {job.name} was made by BenchFlow 0.7 or older" in caplog.text
    assert f"kept in {kept}" in caplog.text
    assert "Pass --fresh" in caplog.text
    # DF-1: the same trials, another figure, and the log says why.
    assert result.mean_reward == pytest.approx(0.5)
    assert result.mean_reward_scored_only == pytest.approx(1.0)
    assert "BenchFlow 0.7 summary said mean_reward=1.00" in caplog.text
    assert "mean_reward_scored_only is 0.7's figure: 1.00" in caplog.text
    # DF-3: the versions are on record.
    from benchflow import __version__

    record = json.loads((job / "evaluation.json").read_text())
    assert record["benchflow_version"] == __version__
    assert record["created_by_benchflow_version"] == "before 0.8"
    assert json.loads((job / "summary.json").read_text())["benchflow_version"] == (
        __version__
    )
    # A second resume keeps the first copy, which holds 0.7's own files.
    asyncio.run(_evaluation(tasks, job, preflight=False).run())
    assert (kept / "summary.json").read_text() == old_summary
    record = json.loads((job / "evaluation.json").read_text())
    assert record["created_by_benchflow_version"] == "before 0.8"


def test_a_trial_records_the_benchflow_that_ran_it(tmp_path) -> None:
    from datetime import datetime

    from benchflow import __version__
    from benchflow.rollout._results import _write_config
    from tests.test_config_redaction import _no_skill_policy

    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="oracle",
        model=None,
        environment="docker",
        skill_policy=_no_skill_policy(tmp_path / "task"),
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=datetime(2026, 1, 1),
        agent_env={},
    )
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["benchflow_version"] == __version__


# --- DF-4: bench eval resume on a 0.7 job ------------------------------------


def test_bench_eval_resume_prints_the_command_that_resumes_a_0_7_job(
    tmp_path,
) -> None:
    from benchflow.cli.main import app

    _, job = _pre_08_job(tmp_path)
    out = CliRunner().invoke(app, ["eval", "resume", str(job)], env={"COLUMNS": "400"})

    assert out.exit_code == 2, out.output
    text = " ".join(out.output.split())
    assert f"{job} was made by BenchFlow 0.7 or older" in text
    assert (
        "next: re-run the command that started it, which resumes this job and "
        "reuses its finished trials: `bench run <tasks folder> --agent oracle "
        f"--sandbox docker --concurrency 2 --jobs-dir {job.parent} --job {job.name}`"
    ) in text
    # The old next step that said to do what the user just did is gone.
    assert "pass the job folder of a run that stopped" not in text
    assert "Traceback" not in out.output


def test_bench_eval_resume_on_a_folder_that_is_no_job(tmp_path) -> None:
    from benchflow.cli.main import app

    (tmp_path / "empty").mkdir()
    out = CliRunner().invoke(
        app, ["eval", "resume", str(tmp_path / "empty")], env={"COLUMNS": "400"}
    )
    assert out.exit_code == 2, out.output
    text = " ".join(out.output.split())
    assert "is not a job folder: it has no evaluation.json and no trials" in text
    assert "next: pass a job folder, <jobs-dir>/<job name>" in text
