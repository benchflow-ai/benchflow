"""Guards FF-sw-tests-ci-confirm-2 against lost/doubled spend at 1a32bcbe."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow import budget as mod
from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    RetryConfig,
    _recorded_attempts,
)
from benchflow.rollout import Rollout, RolloutConfig, _deadline
from tests.test_job_budget import _summary, _tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["parallel-independent", "sequential-shared"])
async def test_resume_same_cap_starts_nothing_and_raised_cap_keeps_receipts(
    tmp_path, monkeypatch, mode
):
    """Exercise the public resume path twice, keeping cancelled attempts unscored."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(_deadline, "hard_deadline_sec", lambda _: None)
    job = Evaluation(
        _tasks(tmp_path, 1),
        tmp_path / "jobs",
        config=EvaluationConfig(
            agent="oracle",
            concurrency=1,
            retry=RetryConfig(max_retries=0),
            job_mode=mode,
        ),
        budget=mod.Budget(max_sandbox_seconds=60),
    )

    async def cancelled_run(evaluation, attempt, seconds):
        entered = asyncio.Event()

        async def run(task, cfg):
            folder = evaluation._jobs_dir / evaluation._job_name / f"task-0__{attempt}"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text('{"agent":"oracle"}')
            rollout = Rollout(RolloutConfig.from_legacy(task_path=task, agent="oracle"))
            rollout._rollout_dir = folder

            async def lifecycle():
                clock[0] += 10
                charge = mod.start_verifier_charge()
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    clock[0] += 3
                    charge.finish()
                    charge.finish()

            monkeypatch.setattr(rollout, "_run_lifecycle", lifecycle)
            return await rollout.run()

        evaluation._run_single_task = AsyncMock(side_effect=run)
        running = asyncio.create_task(evaluation.run())
        await asyncio.wait_for(entered.wait(), 10)
        clock[0] += seconds
        evaluation._budget_guard.check()
        result = await asyncio.wait_for(running, 10)
        assert result.total == 0
        return _summary(evaluation)["budget"]["spent"]

    first = await cancelled_run(job, 1, 25)
    assert first["sandbox_seconds"] == 66
    folder = job._jobs_dir / job._job_name
    for _ in range(2):
        resumed = Evaluation.resume(folder)
        resumed._run_single_task = AsyncMock()
        result = await resumed.run()
        resumed._run_single_task.assert_not_called()
        assert result.total == 0
        assert _summary(resumed)["budget"]["spent"] == first
    # A raised cap permits more work, while retaining every earlier attempt.
    raised = Evaluation.resume(folder)
    raised._config.budget = mod.Budget(max_sandbox_seconds=100)
    second = await cancelled_run(raised, 2, 12)
    assert second["sandbox_seconds"] == 106
    assert second["rollouts"] == 2
    assert not list(folder.glob("*/result.json"))
    assert len(list(folder.glob("*/budget-cancelled.json"))) == 2
    final = Evaluation.resume(folder)
    final._run_single_task = AsyncMock()
    await final.run()
    final._run_single_task.assert_not_called()
    assert _summary(final)["budget"]["spent"] == second


@pytest.mark.parametrize(
    "snapshot", [None, "solver.json", "solver-complete.json", "result.json"]
)
@pytest.mark.parametrize("recorded_total", [20, 50])
def test_receipt_and_result_count_one_attempt(tmp_path, snapshot, recorded_total):
    """Receipts overlap checkpoints/results; old interrupted folders still count."""
    folder = tmp_path / "task__attempt"
    folder.mkdir()
    (folder / "config.json").write_text('{"agent":"oracle"}')
    (folder / "budget-cancelled.json").write_text(
        json.dumps(
            {
                "version": 1,
                "timing": {"total": 38, "verifier_sandbox_total": 28},
            }
        )
    )
    if snapshot:
        (folder / snapshot).write_text(
            json.dumps(
                {
                    "agent_result": {"cost_usd": 2, "total_tokens": 10},
                    "timing": {"total": recorded_total, "verifier_sandbox_total": 10},
                }
            )
        )
    legacy = tmp_path / "task__old"
    legacy.mkdir()
    (legacy / "config.json").write_text('{"agent":"oracle"}')
    records, interrupted = _recorded_attempts(tmp_path)
    guard = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=1000))
    guard.seed(records, interrupted)
    assert guard.rollouts == 2
    assert (
        guard.spent()["sandbox_seconds"]
        == max(38, recorded_total if snapshot else 0) + 28
    )
    assert guard.tokens == (10 if snapshot else 0)
    assert guard.cost_usd == (2 if snapshot else 0)


@pytest.mark.parametrize(
    "body",
    [
        "{",
        "null",
        '{"version":2}',
        '{"version":1,"timing":{"total":-1,"verifier_sandbox_total":1}}',
    ],
)
@pytest.mark.parametrize("snapshot", [None, "solver.json", "result.json"])
def test_invalid_receipt_does_not_silently_refund_spend(tmp_path, body, snapshot):
    """A torn/invalid receipt stops resume even when a solver snapshot exists."""
    folder = tmp_path / "attempt"
    folder.mkdir()
    (folder / "config.json").write_text('{"agent":"oracle"}')
    (folder / "budget-cancelled.json").write_text(body)
    if snapshot:
        (folder / snapshot).write_text("{}")
    with pytest.raises(ValueError, match="cancelled budget receipt"):
        _recorded_attempts(tmp_path)


@pytest.mark.parametrize(
    "snapshot", ["solver.json", "solver-complete.json", "result.json"]
)
@pytest.mark.parametrize("body", ["{", "null", "[]", '"incomplete"', "1", "false"])
@pytest.mark.parametrize("valid_receipt", [True, False])
def test_unreadable_snapshot_cannot_hide_cancellation_receipt(
    tmp_path, snapshot, body, valid_receipt
):
    """Guards d30531fa: malformed snapshots cannot refund finalized spend."""
    folder = tmp_path / "attempt"
    folder.mkdir()
    (folder / "config.json").write_text('{"agent":"oracle"}')
    (folder / snapshot).write_text(body)
    (folder / "budget-cancelled.json").write_text(
        json.dumps(
            {
                "version": 1,
                "timing": {"total": 38, "verifier_sandbox_total": 28},
            }
        )
        if valid_receipt
        else "{"
    )
    if not valid_receipt:
        with pytest.raises(ValueError, match="cancelled budget receipt"):
            _recorded_attempts(tmp_path)
        return
    records, interrupted = _recorded_attempts(tmp_path)
    guard = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=60))
    guard.seed(records, interrupted)
    assert guard.spent()["sandbox_seconds"] == 66
    assert guard.rollouts == 1
    assert not guard.start("resumed")


@pytest.mark.asyncio
async def test_nested_rollouts_and_child_tasks_retain_each_verifier_interval(
    tmp_path, monkeypatch
):
    """Multiple verifier lifetimes survive cancellation without scalar overwrite."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(_deadline, "hard_deadline_sec", lambda _: None)
    task = tmp_path / "task"
    task.mkdir()
    parent = Rollout(RolloutConfig.from_legacy(task_path=task, agent="oracle"))
    child = Rollout(RolloutConfig.from_legacy(task_path=task, agent="oracle"))
    parent._rollout_dir = tmp_path / "job/attempt"
    child._rollout_dir = parent._rollout_dir / "child"

    async def child_lifecycle():
        for _ in range(2):
            charge = mod.start_verifier_charge()
            clock[0] += 5
            charge.finish()
        raise asyncio.CancelledError

    async def parent_lifecycle():
        await asyncio.create_task(child.run())

    monkeypatch.setattr(parent, "_run_lifecycle", parent_lifecycle)
    monkeypatch.setattr(child, "_run_lifecycle", child_lifecycle)
    with pytest.raises(asyncio.CancelledError):
        await parent.run()
    for rollout in (parent, child):
        receipt = rollout._rollout_dir / "budget-cancelled.json"
        assert receipt.exists()
        assert json.loads(receipt.read_text())["timing"] == {
            "total": 10,
            "verifier_sandbox_total": 10,
        }
        (rollout._rollout_dir / "config.json").write_text('{"agent":"oracle"}')
    # The parent's receipt owns child intervals, whether or not that child
    # published a result (or an artifact of that name) before cancellation.
    for child_finished in (False, True):
        if child_finished:
            (child._rollout_dir / "result.json").write_text(
                '{"timing":{"total":10,"verifier_sandbox_total":10}}'
            )
        records, interrupted = _recorded_attempts(parent._rollout_dir.parent)
        guard = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=100))
        guard.seed(records, interrupted)
        assert guard.spent()["sandbox_seconds"] == 20
        assert guard.rollouts == 1
