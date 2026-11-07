"""Guards FF-sw-tests-ci's live verifier accounting fix against 363d815b."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from benchflow import budget as mod


@pytest.mark.asyncio
@pytest.mark.parametrize("report_attempts", [False, True])
async def test_completed_attempts_and_retries_charge_once(
    tmp_path, monkeypatch, report_attempts
):
    """Live receipts replace persisted extras."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    guard = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=1000))
    guard.start("trial")
    with guard.track_trial("trial"):
        for attempt in range(2 if report_attempts else 1):
            if attempt:
                assert guard.retry("trial")
            clock[0] += 10
            charge = mod.start_verifier_charge()
            clock[0] += 25
            charge.finish()
            charge.finish()  # A repeated finally cannot bill twice.
            result_dir = tmp_path / str(attempt)
            result_dir.mkdir()
            (result_dir / "timing.json").write_text(
                json.dumps({"verifier_sandbox_total": 25})
            )
            result = SimpleNamespace(
                rollout_dir=result_dir, cost_usd=1, total_tokens=10
            )
            if report_attempts:
                guard.attempt_done("trial", result)
    guard.finish("trial", result)
    count = 2 if report_attempts else 1
    assert guard.spent()["sandbox_seconds"] == 60 * count
    assert guard.tokens == 10 * count
    assert not guard._verifier_running
    assert not guard._verifier_tracked
    # SDK use outside this job must not inherit the previous trial's budget.
    clock[0] += 10
    charge = mod.start_verifier_charge()
    clock[0] += 10
    charge.finish()
    assert guard.spent()["sandbox_seconds"] == 60 * count


@pytest.mark.asyncio
async def test_concurrent_jobs_do_not_share_verifier_charges(monkeypatch):
    """Guards the fix across task-local and child-task paths."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    guards = [mod.BudgetGuard(mod.Budget(max_sandbox_seconds=1000)) for _ in range(2)]
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def trial(index):
        guard = guards[index]
        guard.start("same-name")
        with guard.track_trial("same-name"):

            async def nested():
                charge = mod.start_verifier_charge()
                entered[index].set()
                await release.wait()
                charge.finish()

            await asyncio.create_task(nested())
        guard.finish("same-name")

    tasks = [asyncio.create_task(trial(i)) for i in range(2)]
    await asyncio.gather(*(event.wait() for event in entered))
    clock[0] = 10
    assert [g.spent()["sandbox_seconds"] for g in guards] == [20, 20]
    release.set()
    await asyncio.gather(*tasks)
    assert [g.spent()["sandbox_seconds"] for g in guards] == [20, 20]
