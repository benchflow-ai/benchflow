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
    with pytest.raises(ValueError, match="max_rollouts"):
        Budget(max_rollouts=0)
    with pytest.raises(ValueError, match="max_rollouts"):
        Budget(max_rollouts=True)
    assert Budget(max_sandbox_seconds=60).to_dict() == {
        "max_cost_usd": None,
        "max_sandbox_seconds": 60.0,
        "max_tokens": None,
        "max_rollouts": None,
    }
    assert Budget.coerce({"max_rollouts": 5}) == Budget(max_rollouts=5)


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
        ({"max_rollouts": 0}, "--max-rollouts"),
        ({"max_rollouts": 3, "worker_concurrency": 2}, "--worker-concurrency"),
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


def test_budget_keyword_leaves_the_callers_config_alone(tmp_path: Path) -> None:
    """``Evaluation(config=cfg, budget=...)`` caps that job only.

    Guards the dx/sdk fix of the regression from bf6e8412 (SDK update), where
    the keyword wrote the budget into the caller's ``EvaluationConfig``: a
    config reused for a second Evaluation silently carried the first job's
    cap, and the caller's object changed under it.
    """
    config = EvaluationConfig(agent="oracle")
    tasks = _tasks(tmp_path, 1)
    capped = Evaluation(
        tasks, tmp_path / "jobs", config=config, budget=Budget(max_tokens=10)
    )
    uncapped = Evaluation(tasks, tmp_path / "jobs2", config=config)

    assert config.budget is None
    assert capped._config.budget == Budget(max_tokens=10)
    assert uncapped._config.budget is None


def _attempts_runner(job: Evaluation, plan: dict[str, list[dict]], delays=None):
    """A fake rollout per attempt, in its own folder: ``plan[task]`` lists each
    attempt's outcome (``error``, ``tokens``, ``cost``); the last repeats."""
    seen: dict[str, int] = {}

    async def fake_run(task_path, _cfg):
        name = task_path.name
        n = seen[name] = seen.get(name, 0) + 1
        outcome = plan[name][min(n, len(plan[name])) - 1]
        await asyncio.sleep((delays or {}).get(name, 0.0))
        rollout_dir = job._jobs_dir / job._job_name / f"{name}__attempt{n}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        error = outcome.get("error")
        rewards = None if error else {"reward": 1.0}
        payload = {
            "task_name": name,
            "rollout_name": rollout_dir.name,
            "rewards": rewards,
            "error": error,
            "error_category": "acp_error" if error else None,
            "verifier_error": None,
            "agent_result": {
                "total_tokens": outcome.get("tokens"),
                "cost_usd": outcome.get("cost"),
            },
        }
        (rollout_dir / "result.json").write_text(json.dumps(payload))
        return RolloutResult(
            task_name=name,
            rollout_name=rollout_dir.name,
            rewards=rewards,
            error=error,
            error_category="acp_error" if error else None,
            total_tokens=outcome.get("tokens"),
            cost_usd=outcome.get("cost"),
            rollout_dir=rollout_dir,
        )

    return fake_run


_FLAKY = {"error": "ACP error -32603: Internal error", "tokens": 100}
_OK = {"tokens": 100}


def _retrying_job(tmp_path: Path, n: int, budget: Budget, **kw) -> Evaluation:
    config = EvaluationConfig(
        agent="oracle",
        concurrency=kw.get("concurrency", 1),
        retry=RetryConfig(max_retries=kw.get("retries", 2), min_wait_sec=0.0),
    )
    return Evaluation(
        _tasks(tmp_path, n), tmp_path / "jobs", config=config, budget=budget
    )


@pytest.mark.asyncio
async def test_rollout_cap_counts_retries_and_lets_running_trials_finish(
    tmp_path: Path,
) -> None:
    """``Budget(max_rollouts=N)`` caps rollouts started, retries included.

    The hill-climb demo (docs/examples/hillclimb) needed a rollout cap and
    had to count trials between steps itself, missing retried attempts.
    """
    job = _retrying_job(tmp_path, 3, Budget(max_rollouts=3))
    plan = {"task-0": [_FLAKY, _FLAKY, _OK], "task-1": [_OK], "task-2": [_OK]}
    job._run_single_task = AsyncMock(side_effect=_attempts_runner(job, plan))
    result = await job.run()
    ran = [c.args[0].name for c in job._run_single_task.await_args_list]
    assert ran == ["task-0", "task-0", "task-0"]
    assert result.results["task-0"].reward == 1.0
    budget = _summary(job)["budget"]
    assert budget["caps"]["max_rollouts"] == 3
    assert budget["spent"]["rollouts"] == 3
    assert budget["spent"]["tokens"] == 300
    assert budget["stopped"] is True and "3 rollouts" in budget["reason"]
    assert budget["not_started"] == ["task-1", "task-2"]
    assert budget["cancelled"] == []


@pytest.mark.asyncio
async def test_a_retry_is_not_started_at_the_rollout_cap(tmp_path: Path) -> None:
    job = _retrying_job(tmp_path, 1, Budget(max_rollouts=2))
    plan = {"task-0": [_FLAKY]}
    job._run_single_task = AsyncMock(side_effect=_attempts_runner(job, plan))
    result = await job.run()
    assert job._run_single_task.await_count == 2
    # The trial keeps its last attempt's result; it is not "not started".
    assert result.results["task-0"].error == _FLAKY["error"]
    budget = _summary(job)["budget"]
    assert budget["spent"]["rollouts"] == 2 and budget["not_started"] == []


@pytest.mark.asyncio
async def test_retried_attempts_count_against_the_token_cap(tmp_path: Path) -> None:
    """Every attempt's tokens count, not only the final attempt's.

    Guards the dx/sdk fix of the budget from bf6e8412 (SDK update), which
    counted a retried trial's USD and tokens from its last attempt only.
    """
    job = _retrying_job(tmp_path, 3, Budget(max_tokens=250), retries=1)
    plan = {"task-0": [_FLAKY, _OK], "task-1": [_OK], "task-2": [_OK]}
    job._run_single_task = AsyncMock(side_effect=_attempts_runner(job, plan))
    await job.run()
    ran = [c.args[0].name for c in job._run_single_task.await_args_list]
    assert ran == ["task-0", "task-0", "task-1"]
    budget = _summary(job)["budget"]
    assert budget["spent"]["tokens"] == 300
    assert budget["not_started"] == ["task-2"]


@pytest.mark.asyncio
async def test_usd_cap_is_enforced_between_starts(tmp_path: Path) -> None:
    """A trial waits while the running ones would reach the USD cap.

    Guards the dx/sdk fix of the budget from bf6e8412 (SDK update): USD and
    tokens were counted only as trials finished, so every free slot started
    a trial until the cap was passed, and the running ones were then
    cancelled (their spend lost). With three trials running at a mean of $1,
    none may start under a $2.50 cap until the estimate allows it.
    """
    job = _retrying_job(tmp_path, 6, Budget(max_cost_usd=2.5), concurrency=3)
    plan = {f"task-{i}": [{"tokens": 10, "cost": 1.0}] for i in range(6)}
    delays = {"task-0": 0.05, **{f"task-{i}": 0.4 for i in range(1, 6)}}
    job._run_single_task = AsyncMock(
        side_effect=_attempts_runner(job, plan, delays=delays)
    )
    await job.run()
    ran = sorted(c.args[0].name for c in job._run_single_task.await_args_list)
    assert ran == ["task-0", "task-1", "task-2"]
    budget = _summary(job)["budget"]
    assert budget["cancelled"] == []
    assert budget["not_started"] == ["task-3", "task-4", "task-5"]
    assert budget["spent"]["cost_usd"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_resume_counts_every_recorded_attempt(tmp_path: Path) -> None:
    """A resume counts retried attempts' spend and every attempt as a rollout.

    Guards the same dx/sdk fix: a resume counted only the reused final
    results, so a retried task's earlier attempts were free the second time.
    """
    job = _retrying_job(tmp_path, 2, Budget(max_tokens=150))
    job_dir = job._jobs_dir / job._job_name
    for name, payload in (
        ("task-0__a1", {"error": _FLAKY["error"], "error_category": "acp_error"}),
        ("task-0__a2", {"rewards": {"reward": 1.0}}),
    ):
        (job_dir / name).mkdir(parents=True)
        (job_dir / name / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task-0",
                    "rollout_name": name,
                    "rewards": None,
                    "error": None,
                    "verifier_error": None,
                    "agent_result": {"total_tokens": 100},
                    **payload,
                }
            )
        )
    (job_dir / "task-1__cut").mkdir()  # an attempt that wrote no result.json
    (job_dir / "task-1__cut" / "config.json").write_text(json.dumps({"agent": "x"}))
    job._run_single_task = AsyncMock(side_effect=_runner(job))
    await job.run()
    assert job._run_single_task.await_count == 0
    budget = _summary(job)["budget"]
    assert budget["spent"]["tokens"] == 200
    assert budget["spent"]["rollouts"] == 3
    assert budget["not_started"] == ["task-1"]


def test_max_rollouts_reaches_the_config_from_the_cli() -> None:
    from benchflow.eval_plan import EvalCreateRequest, build_eval_plan

    plan = build_eval_plan(
        EvalCreateRequest(tasks_dir=HELLO, agent="oracle", max_rollouts=7)
    )
    assert plan.make_eval_config().budget == Budget(max_rollouts=7)


def test_a_misspelt_budget_key_is_refused() -> None:
    """``budget: {max_usd: 5}`` (YAML) or ``EvaluationConfig(budget={...})``
    with a misspelt cap silently ran the job without a budget."""
    with pytest.raises(ValueError, match=r"unknown budget key.*max_usd.*max_cost_usd"):
        EvaluationConfig(agent="oracle", budget={"max_usd": 5})
    assert Budget.coerce({"max_cost_usd": None}) is None
