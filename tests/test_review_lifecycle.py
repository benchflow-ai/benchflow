"""Guards terminal reviewer lifecycle boundaries after PR #1126."""

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow._utils.result_paths import iter_task_result_paths
from benchflow.models import RolloutResult
from benchflow.review import automatic
from benchflow.review.options import ReviewerConfig
from benchflow.review.resume import resume_pending_reviews
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout import _verifier_recovery as recovery
from benchflow.trajectories.rollout_stream import stream_rollouts
from tests.test_automatic_review_scoring import _score
from tests.test_review_runtime import WEIGHTED_RUBRIC, make_task

REVIEWER = ReviewerConfig(
    agent="codex",
    model="azure/gpt5.6terra",
    environment="daytona",
    agent_env={"AZURE_API_KEY": "explicit-test-secret"},
)


@pytest.fixture
def reviewed_rollout(tmp_path, monkeypatch):
    """A real Rollout lifecycle for a rubric task whose verifier scores 1.0.

    Only the sandbox phases are stubbed, so ``_run_lifecycle``,
    ``prepare_terminal_result`` and ``_finish_scoring`` decide what is
    published exactly as in a real run; the review is the only other stub.
    """
    monkeypatch.setattr(
        "benchflow.agents.env.resolve_agent_env",
        Mock(return_value={"AZURE_API_KEY": "private-test-secret"}),
    )
    monkeypatch.setattr("benchflow.review.preflight.validate_reviewer_backend", Mock())
    monkeypatch.setenv("BENCHFLOW_ROLLOUT_HARD_DEADLINE", "off")
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    plan = automatic.prepare_review(task, REVIEWER)
    assert plan is not None
    rollout = Rollout(
        RolloutConfig(task_path=task, task_digest=plan.task_digest, reviewer=REVIEWER)
    )
    root = tmp_path / "jobs" / "job" / f"{task.name}__a"
    root.mkdir(parents=True)
    # What setup() records for a trial with a review plan.
    (root / "config.json").write_text(
        json.dumps({"task_digest": plan.task_digest, "review": plan.metadata()})
    )
    rollout._rollout_dir = root
    rollout._rollout_name = root.name
    rollout._started_at = datetime.now()
    rollout._review_plan = plan
    rollout.setup = AsyncMock()
    rollout.start = AsyncMock()
    rollout.install_agent = AsyncMock()
    rollout._run_steps = AsyncMock()

    async def verify():
        rollout._rewards = {"reward": 1.0}
        rollout._phase = "verified"
        return rollout._rewards

    async def cleanup():
        rollout._phase = "cleaned"

    rollout.verify = AsyncMock(side_effect=verify)
    rollout.cleanup = AsyncMock(side_effect=cleanup)
    return rollout


def _assert_nothing_published(root) -> None:
    """Neither the result readers nor `bench train stream` see the trial."""
    for name in ("result.json", "results.jsonl", "rewards.jsonl", "trainer"):
        assert not (root / name).exists(), name
    assert iter_task_result_paths(root.parent) == []
    assert list(stream_rollouts(root.parent, follow=False)) == []


@pytest.mark.asyncio
async def test_reviewed_trial_publishes_no_result_until_its_scoring_commits(
    reviewed_rollout, monkeypatch
):
    """Guards main's rule that a reviewed trial has no result.json until its
    scoring commits. result.json with the unreviewed verifier reward (1.0)
    used to appear before the review, and the stream emitted that reward
    once, never rereading the trial; the review below fails a blocker."""
    rollout = reviewed_rollout
    root = rollout._rollout_dir
    seen_during_review = []

    async def review(plan, rollout_dir):
        assert rollout_dir == root
        assert json.loads((root / "solver.json").read_text())["rewards"] == {
            "reward": 1.0
        }
        _assert_nothing_published(root)
        seen_during_review.append(True)
        return _score(blocker=False)

    monkeypatch.setattr("benchflow.rollout._review.finish_review", review)
    result = await rollout.run()

    assert seen_during_review == [True]
    assert result.rewards["reward"] == 0.0
    saved = json.loads((root / "result.json").read_text())
    assert saved["rewards"]["reward"] == 0.0
    assert saved["rewards"]["verifier_reward"] == 1.0
    assert saved["scoring"]["status"] == "complete"
    [streamed] = stream_rollouts(root.parent, follow=False)
    assert streamed.reward == 0.0


@pytest.mark.asyncio
async def test_trial_cut_off_mid_review_is_reviewed_on_resume(
    reviewed_rollout, monkeypatch
):
    """Guards resume after a review is cut off by a crash or a budget stop.

    The unreviewed result.json used to survive the cut, and resume counted a
    result with rewards and no scoring block as complete, so the verifier's
    1.0 stayed final although the review fails a blocker."""
    rollout = reviewed_rollout
    root = rollout._rollout_dir
    task = rollout._config.task_path

    async def cut_off(plan, rollout_dir):
        raise asyncio.CancelledError  # a budget stop cancels the trial here

    monkeypatch.setattr("benchflow.rollout._review.finish_review", cut_off)
    with pytest.raises(asyncio.CancelledError):
        await rollout.run()
    assert (root / "solver.json").is_file()
    _assert_nothing_published(root)

    review = AsyncMock(return_value=_score(blocker=False))
    monkeypatch.setattr(automatic, "finish_review", review)
    await resume_pending_reviews(
        root.parent,
        tasks_root=task.parent,
        reviewer=REVIEWER,
        task_names={task.name},
    )

    review.assert_awaited_once()
    saved = json.loads((root / "result.json").read_text())
    assert saved["scoring"]["status"] == "complete"
    assert saved["rewards"]["reward"] == 0.0
    assert saved["rewards"]["verifier_reward"] == 1.0
    [streamed] = stream_rollouts(root.parent, follow=False)
    assert streamed.reward == 0.0


@pytest.mark.asyncio
async def test_recovered_verifier_reward_waits_for_the_review(
    reviewed_rollout, monkeypatch
):
    """Guards the verifier-recovery path of a reviewed trial: the recovered
    reward used to be written to result.json before the review ran."""
    rollout = reviewed_rollout
    root = rollout._rollout_dir

    async def verify():
        rollout._rewards = None
        rollout._verifier_error = "verifier_wedge: no receipt"
        rollout._phase = "verified"

    rollout.verify = AsyncMock(side_effect=verify)

    async def recover(parent):
        attempt = root / "verifier-recovery" / "a1"
        attempt.mkdir(parents=True)
        (attempt / "recovery.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "rewards": {"reward": 1.0},
                    "task_digest": rollout._config.task_digest,
                }
            )
        )
        (root / "verification.json").write_text(
            json.dumps({"attempt": "verifier-recovery/a1"})
        )
        return {"reward": 1.0}, None

    monkeypatch.setattr(recovery, "recovery_ineligible_reason", lambda *a, **k: None)
    monkeypatch.setattr(recovery, "recover_verifier", recover)

    async def review(plan, rollout_dir):
        assert (root / "verification.json").is_file()
        _assert_nothing_published(root)
        solver = json.loads((root / "solver.json").read_text())
        assert solver["verifier_error"] == "verifier_wedge: no receipt"
        return _score(blocker=False)

    monkeypatch.setattr("benchflow.rollout._review.finish_review", review)
    result = await rollout.run()

    assert result.rewards["reward"] == 0.0
    saved = json.loads((root / "result.json").read_text())
    assert saved["rewards"]["reward"] == 0.0
    assert saved["rewards"]["verifier_reward"] == 1.0
    assert saved["verifier_error"] is None


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

    async def reviewer(instance, *, result, lock_held=False):
        assert lock_held
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
