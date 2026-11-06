"""Guards FF-sw-tests-ci-confirm-4: e823144d resumed scoring before budget restore."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow import budget as mod
from benchflow import evaluation as ev
from benchflow.models import RolloutResult
from benchflow.review import resume as review_resume
from tests.test_job_budget import _tasks
from tests.test_review_resume import _complete


def _saved(tmp_path, mode, seconds, cap=60):
    job = ev.Evaluation(
        _tasks(tmp_path, 1),
        tmp_path / "jobs",
        config=ev.EvaluationConfig(agent="oracle", job_mode=mode),
        budget=mod.Budget(max_sandbox_seconds=cap),
    )
    path = job._jobs_dir / job._job_name / "task-0__first"
    path.mkdir(parents=True)
    solver = {
        "task_name": "task-0",
        "rollout_name": path.name,
        "timing": {"total": 20, "verifier_sandbox_total": 10},
    }
    (path / "solver.json").write_text(json.dumps(solver))
    (path / "config.json").write_text('{"agent":"oracle","review":{}}')
    (path / "budget-cancelled.json").write_text(
        json.dumps(
            {
                "version": 1,
                "timing": {"total": seconds - 10, "verifier_sandbox_total": 10},
            }
        )
    )
    job._run_single_task = AsyncMock(return_value=RolloutResult(task_name="task-0"))
    job._save_learner_store()
    return job, path


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["parallel-independent", "sequential-shared"])
async def test_exhausted_resume_starts_no_scoring(tmp_path, monkeypatch, mode):
    """Neither verification recovery nor review may start at an exhausted cap."""
    job, path = _saved(tmp_path, mode, 60)
    score = AsyncMock(return_value={})
    monkeypatch.setattr(review_resume, "_resume_review", score)
    await job.run()
    score.assert_not_called()
    job._run_single_task.assert_not_called()
    assert not (path / "result.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["parallel-independent", "sequential-shared"])
@pytest.mark.parametrize("stage", ["recovery", "reviewer"])
@pytest.mark.parametrize(
    "end", ["budget-cancel", "external-cancel", "complete", "error"]
)
async def test_resumed_scoring_is_live_and_durable(
    tmp_path, monkeypatch, mode, stage, end
):
    """Additional scoring spend survives cancellation, failure and successful review."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    job, path = _saved(tmp_path, mode, 30)
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def score(*args, **kwargs):
        entered.set()
        charge = mod.start_verifier_charge() if stage == "recovery" else None
        try:
            await blocked.wait()
            if end == "error":
                raise RuntimeError("review unavailable")
            (path / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": "task-0",
                        "rollout_name": path.name,
                        "rewards": {"reward": 1},
                        "timing": {"total": 20, "verifier_sandbox_total": 10},
                    }
                )
            )
            return {}
        finally:
            clock[0] += 2  # Teardown remains within the charged boundary.
            if charge:
                charge.finish()

    monkeypatch.setattr(review_resume, "_resume_review", score)
    running = asyncio.create_task(job.run())
    await asyncio.wait_for(entered.wait(), 10)
    clock[0] += 40
    if end == "external-cancel":
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    elif end == "budget-cancel":
        # Exercise the real watcher (no manual cancellation by the test).
        try:
            await asyncio.wait_for(asyncio.shield(running), 1)
        except TimeoutError:
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            pytest.fail("pending scoring ignored its live sandbox budget")
    else:
        blocked.set()
        await running
    expected = 30 + 42 * (2 if stage == "recovery" else 1)
    for _ in range(2):
        # Use the same saved configuration, as a process restart would.
        resumed = ev.Evaluation(
            job._tasks_dir, job._jobs_dir, config=job._config, job_name=job._job_name
        )
        more_scoring = AsyncMock(return_value={})
        monkeypatch.setattr(review_resume, "_resume_review", more_scoring)
        result = await resumed.run()
        more_scoring.assert_not_called()
        assert result.budget["spent"]["sandbox_seconds"] == expected
        assert result.budget["spent"]["rollouts"] == 1


@pytest.mark.asyncio
async def test_scoring_can_finish_existing_rollout_at_rollout_cap(
    tmp_path, monkeypatch
):
    """Budget restoration must not charge another solver start for scoring alone."""
    clock = [0.0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    job, path = _saved(tmp_path, "parallel-independent", 30, cap=100)
    job._config.budget = mod.Budget(max_sandbox_seconds=100, max_rollouts=1)

    async def score(*args, **kwargs):
        clock[0] += 5
        payload = json.loads((path / "solver.json").read_text())
        verdict = _complete()
        payload.update(rewards=verdict.numeric_rewards(), scoring=verdict.to_dict())
        (path / "result.json").write_text(json.dumps(payload))
        return payload

    reviewer = AsyncMock(side_effect=score)
    monkeypatch.setattr(review_resume, "_resume_review", reviewer)
    result = await job.run()
    reviewer.assert_awaited_once()
    assert result.budget["spent"]["sandbox_seconds"] == 35
    assert result.budget["spent"]["rollouts"] == 1
    again = await ev.Evaluation.resume(job._jobs_dir / job._job_name).run()
    reviewer.assert_awaited_once()
    assert again.budget["spent"]["sandbox_seconds"] == 35
    assert again.budget["spent"]["rollouts"] == 1
