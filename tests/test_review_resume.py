"""Scoring-stage resume for automatic review introduced after 6bc55f66.

Guards the rubric integration against the solver replay behavior at commit
6bc55f66: a reviewer failure must never spend on a second solver trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from typer.testing import CliRunner

from benchflow._utils.task_authoring import task_digest
from benchflow.review import automatic, persistence
from benchflow.review.options import ReviewerConfig
from benchflow.review.outcome import ScoringResult, scoring_error
from benchflow.review.resume import (
    ReviewResumeError,
    resume_pending_reviews,
    resume_review,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def saved_trial(tmp_path: Path) -> tuple[Path, Path]:
    task = tmp_path / "tasks" / "physics"
    _write(task / "rubric.json", [{"name": "quality"}])
    rollout = tmp_path / "job" / "physics__a"
    _write(
        rollout / "solver.json",
        {
            "task_name": "physics",
            "task_digest": task_digest(task),
            "purpose": "task",
            "rollout_name": rollout.name,
            "rewards": {"reward": 1.0},
            "started_at": "2026-09-14 00:00:00",
            "finished_at": "2026-09-14 00:01:00",
            "agent": "opencode",
            "model": "solver-model",
            "agent_result": {"total_tokens": 123},
        },
    )
    _write(
        rollout / "config.json",
        {
            "task_digest": task_digest(task),
            "review": {
                "reviewer": {
                    "agent": "opencode",
                    "model": "original-review-model",
                    "environment": "daytona",
                    "timeout_sec": 1234,
                    "agent_env_keys": ["AZURE_API_KEY"],
                },
            },
        },
    )
    _write(rollout / "prompts.json", ["Solve physics"])
    trajectory = rollout / "trajectory" / "acp_trajectory.jsonl"
    trajectory.parent.mkdir()
    trajectory.write_text("")
    return rollout, task


def _complete(*, passed: bool = True) -> ScoringResult:
    return ScoringResult(
        status="complete",
        passed=passed,
        tests_pass=True,
        all_blockers_pass=passed,
        failed_blockers=[] if passed else ["correctness"],
        verifier_reward=1.0,
        rubric_reward=0.8,
        reviewer_run="reviews/attempt-001/reviewer",
    )


def _parent(rollout: Path, scoring: ScoringResult) -> dict:
    source = json.loads((rollout / "solver.json").read_text())
    return {
        **source,
        "scoring": scoring.to_dict(),
        "rewards": scoring.numeric_rewards(),
    }


@pytest.mark.asyncio
async def test_resume_only_reviews_and_preserves_solver_identity(
    saved_trial, monkeypatch
):
    rollout, task = saved_trial
    original = (rollout / "solver.json").read_bytes()
    verdict = _complete()
    prepare = Mock(return_value=SimpleNamespace(config=ReviewerConfig()))
    finish = AsyncMock(return_value=verdict)
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    monkeypatch.setattr(automatic, "finish_review", finish)
    monkeypatch.setattr(
        persistence, "commit_scoring_result", lambda path, score: _parent(path, score)
    )

    result = await resume_review(rollout, tasks_root=task.parent)

    finish.assert_awaited_once_with(prepare.return_value, rollout)
    assert (rollout / "solver.json").read_bytes() == original
    assert result["agent_result"]["total_tokens"] == 123
    assert result["scoring"]["passed"] is True
    assert result["rewards"]["reward"] == 0.8
    saved_config = prepare.call_args.args[1]
    assert saved_config.model == "original-review-model"
    assert saved_config.environment == "daytona"
    assert saved_config.agent_env == {}


@pytest.mark.asyncio
async def test_valid_negative_is_not_rejudged(saved_trial, monkeypatch):
    rollout, task = saved_trial
    negative = _parent(rollout, _complete(passed=False))
    _write(rollout / "result.json", negative)
    prepare = Mock(side_effect=AssertionError("must not prepare another reviewer"))
    monkeypatch.setattr(automatic, "prepare_review", prepare)

    assert await resume_review(rollout, tasks_root=task.parent) == negative
    prepare.assert_not_called()


@pytest.mark.asyncio
async def test_force_rejudges_valid_negative_with_explicit_overrides(
    saved_trial, monkeypatch
):
    rollout, task = saved_trial
    _write(rollout / "result.json", _parent(rollout, _complete(passed=False)))
    prepare = Mock(return_value=object())
    finish = AsyncMock(return_value=_complete())
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    monkeypatch.setattr(automatic, "finish_review", finish)
    monkeypatch.setattr(
        persistence, "commit_scoring_result", lambda path, score: _parent(path, score)
    )

    await resume_review(
        rollout,
        tasks_root=task,
        reviewer=ReviewerConfig(model="new-review-model"),
        force=True,
    )

    config = prepare.call_args.args[1]
    assert config.model == "new-review-model"
    assert config.environment == "daytona"
    assert config.timeout_sec == 1234
    finish.assert_awaited_once()


@pytest.mark.asyncio
async def test_changed_task_rejected_before_reviewer(saved_trial, monkeypatch):
    rollout, task = saved_trial
    (task / "rubric.json").write_text("changed")
    finish = AsyncMock()
    monkeypatch.setattr(automatic, "finish_review", finish)
    with pytest.raises(ReviewResumeError, match="Task digest mismatch"):
        await resume_review(rollout, tasks_root=task.parent)
    finish.assert_not_awaited()


@pytest.mark.asyncio
async def test_untrusted_task_symlink_cannot_escape_root(saved_trial, tmp_path):
    rollout, task = saved_trial
    unsafe_root = tmp_path / "unsafe"
    unsafe_root.mkdir()
    (unsafe_root / "physics").symlink_to(task, target_is_directory=True)
    with pytest.raises(ReviewResumeError, match="not inside"):
        await resume_review(rollout, tasks_root=unsafe_root)


@pytest.mark.asyncio
async def test_batch_retries_saved_solver_even_without_final_result(
    saved_trial, monkeypatch
):
    from benchflow.review import resume

    rollout, task = saved_trial
    retry = AsyncMock()
    monkeypatch.setattr(resume, "resume_review", retry)
    # A child review with an identical task name must not become a second trial.
    _write(rollout / "reviews" / "child" / "solver.json", {"task_name": "physics"})

    config = ReviewerConfig()
    await resume_pending_reviews(
        rollout.parent,
        tasks_root=task.parent,
        reviewer=config,
        task_names={"physics"},
    )
    retry.assert_awaited_once_with(rollout, tasks_root=task.parent, reviewer=config)


@pytest.mark.asyncio
async def test_batch_completed_result_wins_over_incomplete_retry(
    saved_trial, monkeypatch
):
    from benchflow.review import resume

    rollout, task = saved_trial
    _write(rollout / "result.json", _parent(rollout, _complete(passed=False)))
    orphan = rollout.parent / "physics__newer"
    _write(orphan / "solver.json", json.loads((rollout / "solver.json").read_text()))
    _write(orphan / "result.json", _parent(rollout, scoring_error("timeout")))
    retry = AsyncMock()
    monkeypatch.setattr(resume, "resume_review", retry)

    await resume_pending_reviews(
        rollout.parent,
        tasks_root=task.parent,
        reviewer=ReviewerConfig(),
        task_names={"physics"},
    )
    retry.assert_not_awaited()


def test_eval_score_cli_uses_shared_reviewer_options(saved_trial, monkeypatch):
    from benchflow.cli import rescore
    from benchflow.cli.main import app

    rollout, task = saved_trial
    retry = AsyncMock(return_value=_parent(rollout, _complete()))
    monkeypatch.setattr(rescore, "resume_review", retry)
    monkeypatch.setattr(rescore, "_apply_dotenv_to_process_env", lambda: None)
    monkeypatch.setattr(rescore, "write_job_results_jsonl", lambda _: None)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "score",
            str(rollout),
            "--tasks-root",
            str(task.parent),
            "--reviewer-model",
            "gpt-5.6-terra",
            "--reviewer-sandbox",
            "daytona",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "passed=True, reward=0.800" in result.output
    config = retry.call_args.kwargs["reviewer"]
    assert config.model == "gpt-5.6-terra"
    assert config.environment == "daytona"


def test_job_summary_refresh_preserves_solver_usage_and_deduplicates(saved_trial):
    from benchflow.cli.rescore import _refresh_job_artifacts

    rollout, _ = saved_trial
    _write(rollout / "result.json", _parent(rollout, _complete()))
    orphan = rollout.parent / "physics__orphan"
    _write(
        orphan / "result.json", _parent(rollout, scoring_error("reviewer timed out"))
    )
    original = {
        "job_name": "job",
        "total": 1,
        "passed": 0,
        "verifier_errored": 1,
        "n_input_tokens": 500,
        "model": "solver-model",
    }
    _write(rollout.parent / "summary.json", original)
    _write(rollout.parent.parent / "summary.json", original)

    _refresh_job_artifacts(rollout.parent)

    summary = json.loads((rollout.parent / "summary.json").read_text())
    assert summary["total"] == 1
    assert summary["passed"] == 1
    assert summary["verifier_errored"] == 0
    assert summary["mean_reward"] == 0.8
    assert summary["n_input_tokens"] == 500
    assert summary["model"] == "solver-model"
    assert json.loads((rollout.parent.parent / "summary.json").read_text()) == summary


def test_failed_review_is_retained_for_review_only_resume(saved_trial):
    from benchflow.evaluation import Evaluation

    rollout, task = saved_trial
    result = _parent(rollout, scoring_error("verifier timed out after 900s"))
    result["verifier_error"] = "verifier timed out after 900s"
    _write(rollout / "result.json", result)
    job = Evaluation(
        tasks_dir=task.parent,
        jobs_dir=rollout.parent.parent,
        job_name=rollout.parent.name,
    )

    assert job._get_completed_tasks()["physics"]["scoring"]["status"] == "error"


def test_deferred_score_uses_execution_time_and_preserves_solver_timestamps(
    saved_trial,
):
    """Guards delayed rubric retries against elapsed-time inflation after 6bc55f66."""
    rollout, _ = saved_trial
    source = json.loads((rollout / "solver.json").read_text())
    source.update(
        started_at="2020-01-01 00:00:00",
        finished_at="2020-01-01 00:01:00",
        timing={"agent": 50.0, "verify": 10.0, "total": 60.0},
    )
    _write(rollout / "solver.json", source)
    original = (rollout / "solver.json").read_bytes()
    _write(
        rollout / "reviews/attempt-001/reviewer/result.json",
        {"timing": {"total": 12.4}},
    )

    committed = persistence.commit_scoring_result(rollout, _complete())

    assert committed["timing"]["review"] == 12.4
    assert committed["timing"]["total"] == 72.4
    assert committed["solver_started_at"] == source["started_at"]
    assert committed["solver_finished_at"] == source["finished_at"]
    assert committed["scoring_finished_at"] == committed["finished_at"]
    assert committed["finished_at"] > source["finished_at"]
    assert (rollout / "solver.json").read_bytes() == original


def test_force_commit_retains_prior_revision_exports(saved_trial):
    """Guards immutable scoring exports when revising a verdict after 6bc55f66."""
    rollout, _ = saved_trial
    first = _complete().model_copy(update={"revision": "scoring/first.json"})
    second = _complete(passed=False).model_copy(
        update={"revision": "scoring/second.json"}
    )
    _write(rollout / "scoring/first.json", {"scoring": first.to_dict()})
    _write(rollout / "scoring/second.json", {"scoring": second.to_dict()})

    persistence.commit_scoring_result(rollout, first)
    old_exports = (rollout / "scoring/first/results.jsonl").read_bytes()
    persistence.commit_scoring_result(rollout, second)

    assert (rollout / "scoring/first/results.jsonl").read_bytes() == old_exports
    first_parent = json.loads((rollout / "scoring/first/parent.json").read_text())
    assert first_parent["rewards"]["reward"] == 0.8
    second_parent = json.loads((rollout / "result.json").read_text())
    assert second_parent["rewards"]["reward"] == 0.0
    assert second_parent["scoring"]["revision"] == "scoring/second.json"


def test_trainer_export_error_does_not_publish_a_new_final_result(
    saved_trial, monkeypatch
):
    """Guards required export failures being swallowed after commit 6bc55f66."""
    rollout, _ = saved_trial
    original = _parent(rollout, _complete())
    _write(rollout / "result.json", original)
    monkeypatch.setattr(
        "benchflow.trajectories.export.write_rollout_verifiers_jsonl",
        Mock(side_effect=OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        persistence.commit_scoring_result(rollout, _complete(passed=False))

    assert json.loads((rollout / "result.json").read_text()) == original


@pytest.mark.asyncio
async def test_concurrent_scoring_is_rejected_before_second_reviewer(
    saved_trial, monkeypatch
):
    rollout, task = saved_trial
    prepare = Mock(side_effect=AssertionError("must not prepare a second reviewer"))
    monkeypatch.setattr(automatic, "prepare_review", prepare)
    with (
        persistence.scoring_lock(rollout),
        pytest.raises(ValueError, match="already running"),
    ):
        await resume_review(rollout, tasks_root=task.parent)
    prepare.assert_not_called()
    # Releasing an earlier attempt leaves no stale lock to clean up.
    with persistence.scoring_lock(rollout):
        pass
