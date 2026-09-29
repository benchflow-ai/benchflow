"""Shared scoring treats a pending or unassessable assessment as unscored.

Embodied trials separate execution from assessment; a physical run that ended
cleanly (legacy status ``awaiting_assessment``) is not a success. These tests
pin that the shared helpers never pass it, never average it and never re-run
it on resume, whatever stale reward it carries.
"""

import json
import os

import pytest

from benchflow._utils.result_paths import load_task_results
from benchflow._utils.scoring import (
    assessment_status,
    assessment_withholds_score,
    classify_audit_outcome,
    classify_score_outcome,
    extract_reward,
    mean_scored_reward,
    score_summary_fields,
)
from tests.test_automatic_review_scoring import _score


def _result(assessment, reward=1.0, **extra):
    return {
        "task_name": "t2-plane",
        "rewards": {"reward": reward} if reward is not None else None,
        "assessment": assessment,
        "error": None,
        "verifier_error": None,
        **extra,
    }


@pytest.mark.parametrize(
    "result",
    [
        _result({"status": "pending"}),
        _result("pending"),
        _result({"status": "unassessable", "reason": "evidence_not_admissible"}),
        _result({"status": "done"}),
        # Malformed values are unassessable, never a crash in a summary.
        _result(["verified"]),
        _result({"status": ["verified"]}),
        _result({"status": {"verified": True}}),
        # A report row built from a legacy robotics manifest.
        {"status": "awaiting_assessment", "rewards": {"reward": 1.0}},
    ],
)
def test_unassessed_result_is_never_a_pass(result):
    assert assessment_withholds_score(result)
    assert extract_reward(result) is None
    assert classify_score_outcome(result) != "passed"
    assert classify_audit_outcome(result) == "unscored"
    assert mean_scored_reward([result]) is None


def test_unknown_assessment_value_is_unassessable():
    assert assessment_status(_result({"status": "done"})) == "unassessable"
    assert assessment_status({"rewards": None}) is None


@pytest.mark.parametrize(
    "status,reward,outcome", [("verified", 1.0, "passed"), ("failed", 0.0, "failed")]
)
def test_assessed_result_scores_normally(status, reward, outcome):
    result = _result({"status": status}, reward=reward)
    assert not assessment_withholds_score(result)
    assert extract_reward(result) == reward
    assert classify_score_outcome(result) == outcome
    assert classify_audit_outcome(result) == outcome


def test_complete_gate_scoring_cannot_override_pending_assessment():
    scoring = _score()
    result = {
        "task_name": "t2-plane",
        "rewards": scoring.numeric_rewards(),
        "scoring": scoring.to_dict(),
    }
    assert classify_score_outcome(result) == "passed"
    result["assessment"] = {"status": "pending"}
    assert extract_reward(result) is None
    assert classify_score_outcome(result) != "passed"
    assert classify_audit_outcome(result) == "unscored"


def test_summary_counts_unscored_and_excludes_it_from_means():
    summary = score_summary_fields(
        [
            _result({"status": "verified"}, reward=1.0),
            _result({"status": "failed"}, reward=0.0),
            _result({"status": "pending"}, reward=1.0),
            _result({"status": "unassessable"}, reward=None),
        ]
    )
    assert (summary["passed"], summary["failed"], summary["unscored"]) == (1, 1, 2)
    assert summary["mean_reward"] == 0.5
    assert summary["score_ratio"] == 0.25
    assert summary["score_excl_errors_ratio"] == 0.5


def test_resume_keeps_pending_trials_and_prefers_assessed_ones(tmp_path):
    pending = tmp_path / "pending-trial" / "result.json"
    pending.parent.mkdir()
    pending.write_text(json.dumps(_result({"status": "pending"}, reward=None)))
    loaded = load_task_results(tmp_path)
    # Durable: a finished physical episode is never scheduled to run again.
    assert loaded["t2-plane"]["assessment"] == {"status": "pending"}

    assessed = tmp_path / "assessed-trial" / "result.json"
    assessed.parent.mkdir()
    assessed.write_text(json.dumps(_result({"status": "verified"}, reward=1.0)))
    # A newer pending result with a stale reward does not outrank it.
    stale = tmp_path / "stale-trial" / "result.json"
    stale.parent.mkdir()
    stale.write_text(json.dumps(_result({"status": "pending"}, reward=1.0)))
    newest = assessed.stat().st_mtime + 10
    os.utime(stale, (newest, newest))
    assert load_task_results(tmp_path)["t2-plane"]["assessment"] == {
        "status": "verified"
    }
