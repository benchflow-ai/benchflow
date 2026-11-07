"""Guards FF-sw-tests-ci-confirm-3/4 left incomplete at e823144d."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow import budget as mod
from benchflow import evaluation as ev
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout
from tests.test_job_budget import _tasks


def _job(tmp_path, mode, backend="docker", retry=False):
    return ev.Evaluation(
        _tasks(tmp_path, 2),
        tmp_path / "jobs",
        config=ev.EvaluationConfig(
            agent="oracle",
            environment=backend,
            concurrency=1,
            job_mode=mode,
            retry=ev.RetryConfig(max_retries=1 if retry else 0, min_wait_sec=100),
            checkpoints="every-prompt",
            retry_from_checkpoint="on-failure",
        ),
        budget=mod.Budget(max_sandbox_seconds=70),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["parallel-independent", "sequential-shared"])
@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona"])
@pytest.mark.parametrize("phase", ["checkpoint", "backoff"])
async def test_entire_task_spend_survives_resume(
    tmp_path, monkeypatch, mode, backend, phase
):
    """Original 20+10 seconds and 40 retry/backoff seconds exhaust the same cap."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    job = _job(tmp_path, mode, backend, retry=phase == "backoff")
    entered = asyncio.Event()
    calls = []

    async def create(config):
        calls.append(config.task_path.name)
        rollout = Rollout(config)
        folder = job._jobs_dir / job._job_name / f"{config.task_path.name}__first"
        folder.mkdir(parents=True)
        rollout._rollout_dir = folder
        (folder / "config.json").write_text('{"agent":"oracle"}')

        async def run():
            clock[0] += 10
            charge = mod.start_verifier_charge()
            clock[0] += 10
            charge.finish()
            result = RolloutResult(
                task_name=config.task_path.name,
                rollout_name=folder.name,
                rollout_dir=folder,
                rewards={"reward": 0},
                error="connection reset" if phase == "backoff" else None,
                error_category="infra_failure" if phase == "backoff" else None,
            )
            (folder / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": result.task_name,
                        "rollout_name": folder.name,
                        "rewards": result.rewards,
                        "timing": {"total": 20, "verifier_sandbox_total": 10},
                    }
                )
            )
            return result

        monkeypatch.setattr(rollout, "_run_with_scoring", run)
        return rollout

    async def checkpoint(*args):
        # Phased checkpoint children bypass Rollout.run. Backoff starts after
        # this callback and is also outside the old receipt boundary.
        entered.set()
        if phase == "checkpoint":
            await asyncio.Event().wait()

    monkeypatch.setattr(Rollout, "create", create)
    monkeypatch.setattr(ev, "run_checkpoint_retry", checkpoint)
    running = asyncio.create_task(job.run())
    await asyncio.wait_for(entered.wait(), 10)
    await asyncio.sleep(0)
    clock[0] += 40
    job._budget_guard.check()
    result = await asyncio.wait_for(running, 10)
    assert result.budget["spent"]["sandbox_seconds"] == 70
    assert calls == ["task-0"]
    for _ in range(2):
        resumed = ev.Evaluation.resume(job._jobs_dir / job._job_name)
        resumed._run_single_task = AsyncMock(
            return_value=RolloutResult(task_name="task-1", rewards={"reward": 0})
        )
        # A completed original attempt is deliberately left beside the retry.
        # It must not replace the wider durable task total on either scheduler.
        if mode == "sequential-shared":
            resumed._get_completed_tasks = lambda: {}
        again = await resumed.run()
        resumed._run_single_task.assert_not_called()
        assert again.budget["spent"]["sandbox_seconds"] == 70


@pytest.mark.parametrize("exit_kind", ["success", "error", "external-cancel"])
def test_task_totals_include_every_interval_once(tmp_path, monkeypatch, exit_kind):
    """Guards the e823144d gap across nested verifiers and concurrent tasks."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    guard = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=1000))
    guard.restore(tmp_path)
    guard.start("one")
    with guard.track_trial("one"):
        charge = mod.start_verifier_charge()
    clock[0] = 5
    guard.start("two")
    with guard.track_trial("two"):
        other = mod.start_verifier_charge()
    clock[0] = 10
    charge.finish()
    charge.finish()
    if exit_kind == "external-cancel":
        assert not guard.was_cancelled("one")
    else:
        guard.finish("one")
    clock[0] = 15
    other.finish()
    guard.finish("two")
    assert guard.spent()["sandbox_seconds"] == 40
    for _ in range(2):
        resumed = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=1000))
        # These old timings overlap each task's wider receipt, not one another.
        resumed.seed(
            [
                {"task_name": "one", "timing": {"total": 4}},
                {"task_name": "two", "timing": {"total": 6}},
            ]
        )
        resumed.restore(tmp_path)
        assert resumed.spent()["sandbox_seconds"] == 40
    # A further successful run adds only new spend, retaining the old gap.
    resumed.start("one")
    clock[0] += 7
    resumed.finish("one")
    again = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=1000))
    again.seed([{"task_name": "one", "timing": {"total": 11}}])
    again.restore(tmp_path)
    assert again.spent()["sandbox_seconds"] == 47


@pytest.mark.parametrize(
    "body",
    [
        "{",
        "null",
        "[]",
        '{"version": 2}',
        '{"version":1,"task_name":"one","sandbox_seconds":-1}',
        '{"version":1,"task_name":"one","sandbox_seconds":true}',
        '{"version":1,"task_name":"one","sandbox_seconds":NaN}',
        '{"version":1,"task_name":"other","sandbox_seconds":10}',
    ],
)
def test_invalid_task_total_refuses_resume(tmp_path, body):
    """Malformed wider receipts cannot silently fall back to narrower timings."""
    path = mod.task_budget_path(tmp_path, "one")
    path.parent.mkdir()
    path.write_text(body)
    guard = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=100))
    guard.seed([{"task_name": "one", "timing": {"total": 5}}])
    with pytest.raises(ValueError, match="job budget receipt"):
        guard.restore(tmp_path)


@pytest.mark.parametrize("whole", [True, False])
def test_set_aside_preserves_only_retained_task_totals(tmp_path, monkeypatch, whole):
    """A receipt follows its task when resume intentionally replaces old trials."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    guard = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=100))
    guard.restore(tmp_path)
    for name in ("one", "two"):
        (tmp_path / f"{name}__attempt").mkdir()
        guard.start(name)
        clock[0] += 10
        guard.finish(name)
    ev._set_aside(tmp_path, [tmp_path / "one__attempt"], "test", whole_job=whole)
    resumed = mod.BudgetGuard(mod.Budget(max_sandbox_seconds=100))
    resumed.restore(tmp_path)
    assert resumed.spent()["sandbox_seconds"] == (0 if whole else 10)
