"""Guards the fix (both models) against the burst at d39bc8a9."""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.budget import Budget, BudgetGuard
from benchflow.evaluation import RunResult
from tests import test_start_pacing
from tests.test_start_pacing import _job

vt = test_start_pacing.vt


@pytest.mark.parametrize("case", ["resumed-tokens", "cost-projection"])
@pytest.mark.parametrize("concurrency", [32, 64])
async def test_budget_wakeups_do_not_bypass_the_start_rate(
    tmp_path, vt, case, concurrency
):
    """Reproduces both review inputs through the real budget/schedule boundary."""
    n = 5 if case == "resumed-tokens" else 33
    job = _job(tmp_path, n, concurrency)
    if case == "resumed-tokens":
        guard = BudgetGuard(Budget(max_tokens=250))
        guard.seed([{"total_tokens": 100}])
    else:
        guard = BudgetGuard(Budget(max_cost_usd=18))
    job._budget_guard = guard
    starts = []

    async def trial(td, active_guard):
        i = int(td.name.rsplit("-", 1)[-1])
        starts.append(vt.now)
        if case == "resumed-tokens":
            await asyncio.sleep(10 if i == 0 else 40)
            result = SimpleNamespace(total_tokens=0, cost_usd=0)
        else:
            await asyncio.sleep(8.1 if i == 0 else 29.5 if i == 1 else 50)
            result = SimpleNamespace(
                total_tokens=0, cost_usd=1 if i == 0 else 0.01 if i == 1 else 0
            )
        active_guard.finish(td.name, result)
        return RunResult(task_name=td.name, rewards={"reward": 1.0})

    job._run_budgeted = AsyncMock(side_effect=trial)
    await vt.run(job._run_parallel_independent(job._get_task_dirs()))
    assert len(starts) == n
    assert min(b - a for a, b in itertools.pairwise(starts)) >= 0.5 - 1e-9
    assert not guard._running


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_interrupted_admission_releases_the_turn_and_slot(vt, error):
    """Guards the fix: an admission wait must not leak its slot."""
    from benchflow._utils.start_pacing import StartPacer

    pacer = StartPacer()
    slots = asyncio.Semaphore(1)

    async def interrupted():
        raise error()

    with pytest.raises(error):
        await pacer.take_slot(slots, lambda: False, admit=interrupted)
    assert not slots.locked()
    assert not pacer._turn.locked()
    # An unsuccessful admission consumed no token; a following trial can start now.
    assert await pacer.take_slot(slots, lambda: False) == 0
    slots.release()
