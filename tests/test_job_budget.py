"""A hard per-job budget cap.

Prime Intellect users ask for a hard per-run budget (prime #825), and a job's
Daytona spend had no ceiling: BenchFlow reported tokens, USD and
sandbox-seconds but never stopped a job on them. ``Budget`` caps a job on USD
(counted over trials that reported USD), sandbox-seconds (trial wall-clock,
running trials included) and tokens. When a cap is reached the job stops
launching trials and cancels running ones; those are recorded in
``summary.json``'s ``budget`` block as cancelled / not started with the
reason, and are never counted as failures (they are not in the totals, and a
resume runs them).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow import Budget
from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RolloutResult


def _tasks(tmp_path: Path, n: int) -> Path:
    tasks_dir = tmp_path / "tasks"
    for i in range(n):
        (tasks_dir / f"task-{i}").mkdir(parents=True)
        (tasks_dir / f"task-{i}" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
    return tasks_dir


def _job(
    tmp_path: Path, n: int, budget: Budget | None, concurrency: int = 1
) -> Evaluation:
    config = EvaluationConfig(
        agent="oracle", concurrency=concurrency, retry=RetryConfig(max_retries=0)
    )
    return Evaluation(
        tasks_dir=_tasks(tmp_path, n),
        jobs_dir=tmp_path / "jobs",
        config=config,
        budget=budget,
    )


def _runner(
    job: Evaluation,
    *,
    delay: float = 0.0,
    tokens: int | None = 100,
    cost: float | None = None,
    cancelled: list[str] | None = None,
):
    async def fake_run(task_path, _cfg):
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            if cancelled is not None:
                cancelled.append(task_path.name)
            raise
        rollout_dir = job._jobs_dir / job._job_name / f"{task_path.name}__abcd1234"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_name": task_path.name,
            "rollout_name": rollout_dir.name,
            "rewards": {"reward": 0.0},
            "error": None,
            "verifier_error": None,
            "agent_result": {"total_tokens": tokens, "cost_usd": cost},
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        return RolloutResult(
            task_name=task_path.name,
            rollout_name=rollout_dir.name,
            rewards={"reward": 0.0},
            total_tokens=tokens,
            cost_usd=cost,
            rollout_dir=rollout_dir,
        )

    return fake_run


def _summary(job: Evaluation) -> dict:
    return json.loads((job._jobs_dir / job._job_name / "summary.json").read_text())


def test_budget_validates() -> None:
    with pytest.raises(ValueError):
        Budget()
    with pytest.raises(ValueError):
        Budget(max_tokens=0)
    with pytest.raises(ValueError):
        Budget(max_cost_usd=-1.0)
    assert Budget(max_sandbox_seconds=60).to_dict() == {
        "max_cost_usd": None,
        "max_sandbox_seconds": 60.0,
        "max_tokens": None,
    }


@pytest.mark.asyncio
async def test_token_cap_stops_launching(tmp_path: Path) -> None:
    job = _job(tmp_path, 4, Budget(max_tokens=150))
    job._run_single_task = AsyncMock(side_effect=_runner(job, tokens=100))
    result = await job.run()
    ran = [c.args[0].name for c in job._run_single_task.await_args_list]
    assert ran == ["task-0", "task-1"]
    # Not-started trials are not failures and not in the totals.
    assert (result.total, result.failed, result.errored) == (2, 2, 0)
    budget = _summary(job)["budget"]
    assert budget["stopped"] is True
    assert "tokens" in budget["reason"] and "150" in budget["reason"]
    assert budget["caps"]["max_tokens"] == 150
    assert budget["spent"]["tokens"] == 200
    assert budget["not_started"] == ["task-2", "task-3"]
    assert budget["cancelled"] == []
    assert result.budget == budget


@pytest.mark.asyncio
async def test_sandbox_seconds_cap_cancels_running_trials(tmp_path: Path) -> None:
    job = _job(tmp_path, 4, Budget(max_sandbox_seconds=0.6), concurrency=2)
    cancelled: list[str] = []
    job._run_single_task = AsyncMock(
        side_effect=_runner(job, delay=30.0, cancelled=cancelled)
    )
    start = time.monotonic()
    result = await job.run()
    assert time.monotonic() - start < 10
    assert sorted(cancelled) == ["task-0", "task-1"]
    budget = _summary(job)["budget"]
    assert budget["cancelled"] == ["task-0", "task-1"]
    assert budget["not_started"] == ["task-2", "task-3"]
    assert "sandbox-seconds" in budget["reason"]
    assert budget["spent"]["sandbox_seconds"] >= 0.6
    assert result.total == 0 and result.failed == 0
    # A cancelled trial leaves no result.json, so a resume runs it again.
    assert not list((job._jobs_dir / job._job_name).glob("*/result.json"))


@pytest.mark.asyncio
async def test_usd_cap_counts_known_usd_only(tmp_path: Path) -> None:
    job = _job(tmp_path, 3, Budget(max_cost_usd=1.0))
    job._run_single_task = AsyncMock(side_effect=_runner(job, cost=0.6))
    await job.run()
    budget = _summary(job)["budget"]
    assert budget["spent"]["cost_usd"] == pytest.approx(1.2)
    assert budget["not_started"] == ["task-2"]

    other = _job(tmp_path / "b", 2, Budget(max_cost_usd=1.0))
    other._run_single_task = AsyncMock(side_effect=_runner(other, cost=None))
    await other.run()
    budget = _summary(other)["budget"]
    assert budget["stopped"] is False
    assert budget["spent"]["usd_unknown_trials"] == 2


@pytest.mark.asyncio
async def test_no_budget_writes_no_block(tmp_path: Path) -> None:
    job = _job(tmp_path, 1, None)
    job._run_single_task = AsyncMock(side_effect=_runner(job))
    await job.run()
    assert "budget" not in _summary(job)


@pytest.mark.asyncio
async def test_resume_counts_what_was_already_spent(tmp_path: Path) -> None:
    job = _job(tmp_path, 3, Budget(max_tokens=150))
    job._run_single_task = AsyncMock(side_effect=_runner(job, tokens=100))
    await job.run()
    resumed = Evaluation.resume(job._jobs_dir / job._job_name)
    assert resumed._config.budget == Budget(max_tokens=150)
    resumed._run_single_task = AsyncMock(side_effect=_runner(resumed))
    await resumed.run()
    assert resumed._run_single_task.await_count == 0
    budget = _summary(resumed)["budget"]
    assert budget["spent"]["tokens"] == 200
    assert budget["not_started"] == ["task-2"]


def test_config_round_trip(tmp_path: Path) -> None:
    job = _job(tmp_path, 1, Budget(max_cost_usd=5, max_tokens=1000))
    again = Evaluation.from_dict(job.to_dict())
    assert again._config.budget == Budget(max_cost_usd=5, max_tokens=1000)


@pytest.mark.asyncio
async def test_sequential_shared_stops_launching_and_cancels(tmp_path: Path) -> None:
    config = EvaluationConfig(
        agent="oracle",
        concurrency=1,
        retry=RetryConfig(max_retries=0),
        job_mode="sequential-shared",
    )
    job = Evaluation(
        tasks_dir=_tasks(tmp_path, 3),
        jobs_dir=tmp_path / "jobs",
        config=config,
        budget={"max_sandbox_seconds": 0.5},
    )
    cancelled: list[str] = []
    job._run_single_task = AsyncMock(
        side_effect=_runner(job, delay=30.0, cancelled=cancelled)
    )
    result = await job.run()
    assert cancelled == ["task-0"]
    assert result.budget is not None
    assert result.budget["cancelled"] == ["task-0"]
    assert result.budget["not_started"] == ["task-1", "task-2"]
    assert result.total == 0


HELLO = Path(__file__).parent / "examples" / "hello-world-task"


def test_cli_flags_reach_the_config() -> None:
    from benchflow.eval_plan import EvalCreateRequest, build_eval_plan

    plan = build_eval_plan(
        EvalCreateRequest(
            tasks_dir=HELLO, agent="oracle", max_cost_usd=2.5, max_tokens=10_000
        )
    )
    assert plan.make_eval_config().budget == Budget(max_cost_usd=2.5, max_tokens=10_000)
    none = build_eval_plan(EvalCreateRequest(tasks_dir=HELLO, agent="oracle"))
    assert none.make_eval_config().budget is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_tokens": 0}, "--max-tokens"),
        ({"max_sandbox_seconds": -5.0}, "--max-sandbox-seconds"),
        ({"max_cost_usd": 1.0, "worker_concurrency": 2}, "--worker-concurrency"),
    ],
)
def test_cli_flags_are_validated(kwargs, match) -> None:
    from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan

    with pytest.raises(EvalPlanError, match=match):
        build_eval_plan(EvalCreateRequest(tasks_dir=HELLO, agent="oracle", **kwargs))


def test_cli_rejects_a_budget_for_source_env() -> None:
    from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan

    with pytest.raises(EvalPlanError, match="--source-env"):
        build_eval_plan(
            EvalCreateRequest(source_env="primeintellect/x", max_tokens=5, model="m")
        )


@pytest.mark.asyncio
async def test_ci_run_summary_carries_the_budget(tmp_path: Path) -> None:
    from benchflow.job_export import run_summary_export

    job = _job(tmp_path, 3, Budget(max_tokens=150))
    job._run_single_task = AsyncMock(side_effect=_runner(job, tokens=100))
    result = await job.run()
    doc = run_summary_export(
        result,
        job_dir=None,
        timeouts=0,
        fail_under=None,
        fail_on=[],
        gate_failed=[],
        exit_code=0,
    ).model_dump(mode="json")
    assert doc["budget"]["stopped"] is True
    assert doc["budget"]["not_started"] == ["task-2"]


@pytest.mark.asyncio
async def test_a_trial_that_raises_stops_counting_sandbox_seconds(
    tmp_path: Path,
) -> None:
    job = _job(tmp_path, 2, Budget(max_sandbox_seconds=0.5))

    async def boom(task_path, _cfg):
        raise RuntimeError("unexpected")

    job._run_single_task = AsyncMock(side_effect=boom)
    await job.run()
    await asyncio.sleep(0.6)
    assert job._budget_guard is not None
    assert job._budget_guard.spent()["sandbox_seconds"] < 0.5
    budget = _summary(job)["budget"]
    assert budget["stopped"] is False and budget["cancelled"] == []
