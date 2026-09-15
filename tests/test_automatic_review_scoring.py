"""Guards automatic rubric review integration based on PR #1126."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console

from benchflow._utils.evaluation_results import rollout_result_payload
from benchflow._utils.result_paths import iter_task_result_paths, load_task_results
from benchflow._utils.scoring import (
    classify_audit_outcome,
    classify_score_outcome,
    extract_reward,
    mean_scored_reward,
    score_summary_fields,
)
from benchflow.cli._live_progress import LiveEvalProgress
from benchflow.eval_artifacts import _iter_rollouts as artifact_rollouts
from benchflow.eval_lift import build_lift_report
from benchflow.metrics import collect_metrics
from benchflow.models import RolloutResult
from benchflow.review.config import Rubric
from benchflow.review.outcome import (
    ScoringResult,
    complete_scoring,
    deterministic_pass,
    scoring_error,
    scoring_from_result,
)
from benchflow.review.scoring import score_weighted_review
from benchflow.trajectories.export_prime_sft import (
    _iter_rollout_dirs,
    _result_training_skip_reason,
    _reward_from_result,
)
from benchflow.trajectories.results import build_rollout_results_record


def _score(*, tests=True, blocker=True, scores=(2, 1)):
    rubric = Rubric.model_validate(
        {
            "criteria": [
                {
                    "name": name,
                    "blocker": int(is_blocker),
                    "weight": weight,
                    "description": "A task criterion.",
                    "guidance": "Judge the submitted evidence.",
                }
                for name, is_blocker, weight in (
                    ("reproducible", True, 10),
                    ("method", False, 6),
                    ("evidence", False, 4),
                )
            ]
        }
    )
    review = score_weighted_review(
        rubric,
        {
            "reproducible": {"outcome": "pass" if blocker else "fail"},
            "method": {"score": scores[0]},
            "evidence": {"score": scores[1]},
        },
        deterministic_pass=tests,
    )
    return complete_scoring(
        review,
        verifier_reward=1.0 if tests else 0.0,
        reviewer_run="reviews/attempt-001/rollout",
    )


def _result(scoring):
    return {
        "task_name": "physics",
        "rewards": scoring.numeric_rewards(),
        "scoring": scoring.to_dict(),
    }


@pytest.mark.parametrize(
    ("tests", "blocker", "scores", "reward", "outcome"),
    [
        (True, True, (2, 1), 0.8, "passed"),
        (True, True, (0, 0), 0.0, "passed"),
        (True, True, (2, 2), 1.0, "passed"),
        (False, True, (2, 2), 0.0, "failed"),
        (True, False, (2, 2), 0.0, "failed"),
        (False, False, (2, 2), 0.0, "failed"),
    ],
)
def test_gate_success_and_quality_are_independent(
    tests, blocker, scores, reward, outcome
):
    """Guards automatic review semantics introduced after PR #1126."""
    scoring = _score(tests=tests, blocker=blocker, scores=scores)
    result = _result(scoring)
    assert result["rewards"]["reward"] == reward
    assert classify_score_outcome(result) == outcome
    assert classify_audit_outcome(result) == outcome
    assert deterministic_pass(result) is tests
    assert scoring_from_result(json.loads(json.dumps(result))) == scoring
    runtime = RolloutResult("physics", rewards=result["rewards"], scoring=scoring)
    assert runtime.score_outcome == outcome


@pytest.mark.parametrize(
    "patch",
    [
        {"passed": "true"},
        {"passed": False},
        {"schema_version": 999},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"status": "pending"},
        {"all_blockers_pass": False},
        {"rubric_reward": float("nan")},
        {"rubric_reward": True},
        {"rubric_reward": 1.01},
        {"tests_pass": False},
        {"revision": "../unrelated.json"},
    ],
)
def test_malformed_scoring_cannot_fall_back_to_legacy_pass(patch):
    """Guards the PR #1126 migration against stale-reward false passes."""
    result = _result(_score(scores=(2, 2)))
    result["scoring"].update(patch)
    assert classify_score_outcome(result) == "verifier_errored"
    assert extract_reward(result) is None
    assert _reward_from_result(result) is None
    assert _result_training_skip_reason(result) == "scoring_error"


def test_stale_numeric_reward_is_excluded_from_score_denominators():
    """Guards scoring/reward consistency after PR #1126."""
    result = _result(_score())
    result["rewards"]["reward"] = 1.0
    with pytest.raises(ValueError, match="disagrees"):
        scoring_from_result(result)
    assert classify_score_outcome(result) == "verifier_errored"
    assert mean_scored_reward([result, {"rewards": {"reward": 0.5}}]) == 0.5


def test_review_error_with_known_test_pass_is_unscored():
    """Guards reviewer failure accounting after PR #1126."""
    scoring = scoring_error("Reviewer timed out", tests_pass=True, verifier_reward=1.0)
    result = _result(scoring)
    result["rewards"] = {"reward": 1.0}  # Interrupted writer's old test reward.
    assert deterministic_pass(result)
    assert scoring.numeric_rewards() is None
    assert classify_score_outcome(result) == "verifier_errored"
    assert extract_reward(result) is None
    assert not RolloutResult("physics", scoring=scoring).success


def test_legacy_score_contract_is_preserved():
    """Guards historical results from PR #1126 during migration."""
    assert classify_score_outcome({"rewards": {"reward": 0.8}}) == "failed"
    assert classify_score_outcome({"rewards": {"reward": 1.0}}) == "passed"
    assert not deterministic_pass({"rewards": {"reward": True}})
    assert not deterministic_pass(
        {"rewards": {"reward": 1.0}, "verifier_error": "test crash"}
    )


def test_inconsistent_component_cannot_be_constructed():
    """Guards the typed contract introduced after PR #1126."""
    data = _score().to_dict()
    data["verifier_reward"] = 0.0
    with pytest.raises(ValidationError, match="test gate"):
        ScoringResult.model_validate(data)


def _write(path: Path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))


def test_reports_count_parent_gate_success_and_exclude_reviewer_trials(tmp_path):
    """Guards all result readers against review-child inflation after PR #1126."""
    task = tmp_path / "physics__trial"
    parent_result = task / "result.json"
    _write(parent_result, _result(_score()))
    _write(
        task / "reviews/run/result.json", {"task_name": "review", "purpose": "reviewer"}
    )
    _write(task / "reviews/evidence/result.json", {"task_name": "copied-physics"})
    _write(tmp_path / "orphan-review/result.json", {"purpose": "reviewer"})
    assert iter_task_result_paths(tmp_path) == [parent_result]
    assert artifact_rollouts(tmp_path) == [task]
    assert _iter_rollout_dirs(tmp_path) == [task]
    metrics = collect_metrics(tmp_path)
    assert (metrics.total, metrics.passed, metrics.failed) == (1, 1, 0)
    assert metrics.tasks[0].reward == 0.8
    lift = build_lift_report(tmp_path, tmp_path, bootstrap_samples=5)
    assert lift["metrics"]["pass_rate_base"] == 1.0


def test_pending_parent_checkpoint_excludes_captured_results(tmp_path):
    """Guards pre-commit result discovery after baseline PR #1126."""
    trial = tmp_path / "physics__pending"
    _write(trial / "solver.json", {"task_name": "physics", "rewards": {"reward": 1.0}})
    _write(
        trial / "evidence/workspace/result.json",
        {"task_name": "forged-workspace-trial", "rewards": {"reward": 1.0}},
    )
    _write(
        trial / "reviews/task-input/result.json",
        {"task_name": "captured-task-input", "rewards": {"reward": 1.0}},
    )
    _write(
        trial / "reviews/attempt/runtime/result.json",
        {"task_name": "review", "purpose": "reviewer", "rewards": {"reward": 1.0}},
    )
    assert iter_task_result_paths(tmp_path) == []
    assert load_task_results(tmp_path) == {}
    assert artifact_rollouts(tmp_path) == []
    assert _iter_rollout_dirs(tmp_path) == []
    assert collect_metrics(tmp_path).total == 0

    _write(trial / "result.json", _result(_score()))
    assert iter_task_result_paths(tmp_path) == [trial / "result.json"]
    assert set(load_task_results(tmp_path)) == {"physics"}


def test_live_progress_counts_partial_quality_as_pass():
    """Guards runtime pass@1 after PR #1126."""
    scoring = _score()
    dashboard = LiveEvalProgress(
        Console(), label="physics", agent="codex", model="terra", sandbox="daytona"
    )
    dashboard.on_result(
        "physics",
        RolloutResult("physics", rewards=scoring.numeric_rewards(), scoring=scoring),
    )
    assert (dashboard._passed, dashboard._failed, dashboard._errored) == (1, 0, 0)


def test_fresh_evaluation_payload_preserves_score_contract(tmp_path):
    """Guards fresh/resumed evaluation parity after PR #1126."""
    scoring = _score()
    result = RolloutResult(
        "physics", rewards=scoring.numeric_rewards(), scoring=scoring
    )
    payload = rollout_result_payload(
        result, source_provenance=None, tasks_dir=tmp_path, task_name="physics"
    )
    assert payload["scoring"] == scoring.to_dict()
    assert classify_score_outcome(payload) == "passed"


def test_trainer_record_preserves_gate_verdict(tmp_path):
    """Guards trainer result metadata after PR #1126."""
    scoring = _score()
    record = build_rollout_results_record(
        tmp_path,
        task_name="physics",
        rollout_name="trial",
        agent="codex",
        agent_name="codex",
        model="terra",
        n_tool_calls=0,
        prompts=[],
        trajectory=[],
        partial_trajectory=False,
        rewards=scoring.numeric_rewards(),
        scoring=scoring,
        error=None,
        verifier_error=None,
    )
    assert record["reward"] == 0.8
    assert record["passed"] is True
    assert record["scoring"] == scoring.to_dict()
    assert _reward_from_result(record) == 0.8
    assert classify_score_outcome(record) == "passed"


def test_rescored_summary_uses_gate_passes_and_quality_mean():
    """Guards initial/resumed summary parity after PR #1126."""
    summary = score_summary_fields(
        [
            _result(_score()),
            _result(_score(scores=(0, 0))),
            _result(_score(blocker=False)),
            _result(scoring_error("reviewer VM unavailable")),
        ]
    )
    assert summary["total"] == 4
    assert summary["passed"] == summary["pass"] == 2
    assert summary["failed"] == summary["fail"] == 1
    assert summary["verifier_errored"] == 1
    assert summary["score_ratio"] == 0.5
    assert summary["score_excl_errors_ratio"] == pytest.approx(2 / 3)
    assert summary["mean_reward"] == pytest.approx(0.8 / 3)
