"""Guards terminal reviewer lifecycle boundaries after PR #1126."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig


@pytest.mark.parametrize("phase", ["verified", "cleaned", "reviewing"])
def test_result_cannot_publish_test_only_reward_while_review_is_pending(
    tmp_path, phase
):
    """Guards premature result publication after PR #1126."""
    rollout = Rollout(RolloutConfig(task_path=tmp_path))
    rollout._phase = phase
    rollout._review_plan = object()
    rollout._rewards = {"reward": 1.0}
    rollout._build_result = Mock(side_effect=AssertionError("premature final result"))
    assert rollout.result is None
    rollout._build_result.assert_not_called()


@pytest.mark.asyncio
async def test_reviewer_wait_is_outside_solver_deadline(tmp_path, monkeypatch):
    """Guards legitimate reviewer queues from solver watchdogs after PR #1126."""
    rollout = Rollout(RolloutConfig(task_path=tmp_path))
    rollout._rollout_dir = tmp_path
    (tmp_path / "solver.json").write_text("{}")
    original = RolloutResult(
        "physics", rollout_name="physics__trial", rewards={"reward": 1.0}
    )
    reviewed = RolloutResult("physics", rewards={"reward": 0.8})
    rollout._review_plan = object()

    async def solver():
        rollout._phase = "cleaned"
        return original

    async def reviewer(instance, *, result):
        assert result is original
        await asyncio.sleep(0.04)
        return reviewed

    rollout._run_lifecycle = solver
    monkeypatch.setattr(
        "benchflow.rollout._deadline.hard_deadline_sec", lambda config: 0.01
    )
    monkeypatch.setattr("benchflow.rollout.finish_terminal_review", reviewer)
    assert await rollout.run() is reviewed
    assert rollout.result is reviewed


@pytest.mark.asyncio
async def test_manual_finalize_releases_solver_and_runs_reviewer_once(
    tmp_path, monkeypatch
):
    """Guards phased SDK finalization after PR #1126."""
    rollout = Rollout(RolloutConfig(task_path=tmp_path))
    rollout._review_plan = object()
    rollout._phase = "verified"
    reviewed = RolloutResult("physics", rewards={"reward": 0.8})
    cleanup = AsyncMock()
    rollout.cleanup = cleanup

    async def reviewer(instance):
        cleanup.assert_awaited_once()
        return reviewed

    run_review = AsyncMock(side_effect=reviewer)
    monkeypatch.setattr("benchflow.rollout.finish_terminal_review", run_review)
    assert await rollout.finalize() is reviewed
    assert await rollout.finalize() is reviewed
    run_review.assert_awaited_once()
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_hard_deadline_without_checkpoint_does_not_launch_reviewer(
    tmp_path, monkeypatch
):
    """Guards watchdog cancellation before solver commit after PR #1126."""
    rollout = Rollout(RolloutConfig(task_path=tmp_path))
    rollout._rollout_dir = tmp_path
    rollout._review_plan = object()

    async def solver():
        try:
            await asyncio.sleep(1)
        finally:
            rollout._phase = "cleaned"

    rollout._run_lifecycle = solver
    reviewer = AsyncMock(side_effect=AssertionError("no completed solver to grade"))
    monkeypatch.setattr(
        "benchflow.rollout._deadline.hard_deadline_sec", lambda config: 0.01
    )
    monkeypatch.setattr("benchflow.rollout.finish_terminal_review", reviewer)
    result = await rollout.run()
    assert "hard deadline" in result.error
    reviewer.assert_not_awaited()


@pytest.mark.asyncio
async def test_scoring_disk_failure_never_becomes_a_solver_retry(tmp_path, monkeypatch):
    """Guards durable solver evidence on scoring commit errors after PR #1126."""
    from benchflow.rollout import _review
    from tests.test_automatic_review_scoring import _score

    rollout = Rollout(RolloutConfig(task_path=tmp_path))
    rollout._rollout_dir = tmp_path
    rollout._review_plan = object()
    original = RolloutResult(
        "physics", rollout_name="physics__trial", rewards={"reward": 1.0}
    )
    monkeypatch.setattr(_review, "finish_review", AsyncMock(return_value=_score()))
    monkeypatch.setattr(
        _review, "commit_scoring_result", Mock(side_effect=OSError("disk full"))
    )
    result = await _review.finish_terminal_review(rollout, result=original)
    assert result.scoring.status == "error"
    assert result.rewards is None
    assert "disk full" in result.verifier_error
    assert not (tmp_path / "result.json").exists()
