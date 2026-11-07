"""Guards FF-sw-evaluation scheduling and lock races at 363d815b."""

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchflow.budget import Budget, BudgetGuard
from benchflow.evaluation import Evaluation, EvaluationConfig, UsageLimitStop
from benchflow.models import RunResult


def _job(root):
    job = object.__new__(Evaluation)
    job._jobs_dir = Path(root)
    job._job_name = "job"
    return job


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrency", [2, 32])
@pytest.mark.parametrize("stop", ["usage", "breaker"])
async def test_budget_waiter_rechecks_stop_before_start(
    tmp_path, monkeypatch, concurrency, stop
):
    monkeypatch.setenv("BENCHFLOW_API_ERROR_BREAKER_THRESHOLD", "1")
    job = _job(tmp_path)
    job._config = EvaluationConfig(concurrency=concurrency)
    job._usage_stop = UsageLimitStop()
    guard = BudgetGuard(Budget(max_tokens=250))
    guard.seed([{"agent_result": {"total_tokens": 100}}])
    job._budget_guard = guard
    job._on_task_start = None
    job._on_result = None
    job._log_and_report = Mock()
    waiting = asyncio.Event()
    original_admit = guard.admit

    async def admit(name, task=None):
        if name == "b":
            waiting.set()
        return await original_admit(name, task)

    monkeypatch.setattr(guard, "admit", admit)
    ran = []

    async def run(td):
        ran.append(td.name)
        if td.name == "a":
            await asyncio.wait_for(waiting.wait(), 3)
        return RunResult(
            task_name=td.name,
            error="usage limit reached"
            if stop == "usage"
            else "API error [openai/permanent] HTTP 401",
            error_category="usage_limit" if stop == "usage" else "api_error",
            total_tokens=0,
        )

    job._run_task = run
    await job._run_parallel_independent([tmp_path / "a", tmp_path / "b"])
    assert ran == ["a"]
    assert not guard._running
    assert guard.rollouts == 2  # seeded trial plus a; b never started
