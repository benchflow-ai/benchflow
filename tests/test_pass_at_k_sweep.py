"""Guard the P4 Codex findings against the reviewed commit 20f9bcb8."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow._utils.result_paths import load_task_results
from benchflow._utils.scoring import score_summary_fields
from benchflow.cli import rescore
from benchflow.cli.main import app
from benchflow.pass_at_k import Sample, solve_rates
from benchflow.rollout import Rollout, RolloutConfig, _init_rollout
from tests.test_python_sdk_load_job import _trial


@pytest.mark.asyncio
async def test_external_cancellation_does_not_publish_a_terminal_deadline(
    tmp_path, monkeypatch
):
    """Guards the P4-C2 fix: interruption is not a terminal failed attempt."""
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Solve the task.")
    trial = Rollout(RolloutConfig(task_path=task, jobs_dir=tmp_path / "jobs"))
    entered = asyncio.Event()

    async def lifecycle():
        entered.set()
        await asyncio.sleep(3600)

    trial._run_lifecycle = lifecycle
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "60")
    running = asyncio.create_task(trial.run())
    await entered.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert not list((tmp_path / "jobs").rglob("result.json"))


def test_noop_scoring_refresh_keeps_terminal_agent_errors(tmp_path, monkeypatch):
    """Guards the fix at 20f9bcb8 without changing resume rules."""
    job = tmp_path / "job"
    passed = _trial(job, "pass", reward=1)
    _trial(job, "fail", reward=0)
    _trial(job, "crash", reward=None, error="agent crashed")
    (job / "summary.json").write_text(
        json.dumps(
            {
                "job_name": "job",
                "mean_reward": 1 / 3,
                "mean_reward_n": 3,
                "solve_rates": {"trials": 3, "solve_rate": 1 / 3},
            }
        )
    )
    # The already-admitted score returns unchanged. Only derived exports run.
    monkeypatch.setattr(
        rescore,
        "resume_review",
        AsyncMock(return_value=json.loads((passed / "result.json").read_text())),
    )
    result = CliRunner().invoke(
        app, ["eval", "score", str(passed), "--tasks-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((job / "summary.json").read_text())
    assert (summary["mean_reward"], summary["mean_reward_n"]) == (1 / 3, 3)
    assert summary["solve_rates"]["trials"] == 3
    assert summary["solve_rates"]["solve_rate"] == 1 / 3
    assert "crash" not in load_task_results(job), "resume still schedules agent errors"


def test_refresh_recomputes_solve_rates_from_the_same_final_attempts(tmp_path):
    """Guards 20f9bcb8: stale solve rates cannot survive scoring refresh."""
    job = tmp_path / "job"
    _trial(job, "task", reward=None, error="agent crashed", suffix="old")
    _trial(job, "task", reward=1, suffix="new")
    _trial(job, "crash", reward=None, error="agent crashed")
    (job / "summary.json").write_text(json.dumps({"solve_rates": {"trials": 99}}))
    rescore._refresh_job_artifacts(job)
    summary = json.loads((job / "summary.json").read_text())
    assert summary["mean_reward_n"] == 2
    assert summary["solve_rates"]["trials"] == 2
    assert summary["solve_rates"]["pass_at_k"]["1"] == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("initialized", [False, True])
async def test_setup_deadline_is_a_durable_unscored_trial(
    tmp_path, monkeypatch, initialized
):
    """Guards P4-C2 at 20f9bcb8: readback keeps the live denominator."""
    task = tmp_path / "deadline"
    task.mkdir()
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Solve the task.")
    cfg = RolloutConfig(
        task_path=task, jobs_dir=tmp_path, job_name="job", rollout_name="deadline__test"
    )
    trial = Rollout(cfg)
    if initialized:
        (
            trial._task,
            trial._rollout_dir,
            trial._rollout_paths,
            trial._started_at,
            trial._job_name,
            trial._rollout_name,
        ) = _init_rollout(task, "job", cfg.rollout_name, tmp_path)

    async def wedged_setup():
        await asyncio.sleep(3600)

    trial._run_lifecycle = wedged_setup
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "0.02")
    result = await trial.run()
    assert result.rewards is None and result.error_category == "infra_failure"
    _trial(tmp_path / "job", "pass", reward=1)
    job = bf.load_job(tmp_path / "job")
    assert len(job.trials) == 2
    assert job.solve_rates().solve_rate == 0.5
    saved = json.loads((result.rollout_dir / "result.json").read_text())
    assert saved["rewards"] is None
    assert saved["error_category"] == result.error_category
    assert saved["agent"] == cfg.primary_agent
    assert saved["rollout_name"] == "deadline__test"


@pytest.mark.parametrize("n", [2, 5])
def test_implicit_ks_keep_the_historical_scored_only_values(n):
    """Guards the fix at 20f9bcb8, including serialization/UI."""
    rates = solve_rates(
        [Sample("a", 1)] + [Sample("a", 0)] * (n - 1) + [Sample("b", None)]
    )
    entry = rates.get_scored_only(n)
    assert entry is not None
    assert entry.pass_at_k == 1
    assert str(n) in rates.to_dict()["pass_hat_k_scored_only"]
    assert f"pass@{n}, scored only" in dict(rates.rows())
    explicit = solve_rates([Sample("a", 1), Sample("b", None)], ks=[1])
    assert [p.k for p in explicit.at_k_scored_only] == [1]


def test_metrics_and_persisted_summary_have_an_explicit_v2_boundary(tmp_path):
    """Guards the fix at 20f9bcb8; existing version-2 exports stay 2."""
    job = tmp_path / "job"
    _trial(job, "pass", reward=1)
    _trial(job, "crash", reward=None, error="agent crashed")
    out = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout).get("schema_version") == 2
    assert (
        score_summary_fields([{"task_name": "crash", "rewards": None}]).get(
            "schema_version"
        )
        == 2
    )
