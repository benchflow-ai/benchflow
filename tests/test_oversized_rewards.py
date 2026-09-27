"""Oversized integer rewards must not crash scoring, outcome, or results paths.

Python's ``json`` parses ``1e1000``-sized integer literals into exact ``int``
values that ``float()`` and ``math.isfinite()`` reject with ``OverflowError``.
The oversized-reward change made ``review/automatic.py`` treat such a reward like a
non-finite one; these tests guard the remaining aggregation and export paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow._utils.scoring import (
    classify_audit_outcome,
    classify_score_outcome,
    extract_reward,
    mean_scored_reward,
    score_summary_fields,
)
from benchflow.review.outcome import deterministic_pass, scoring_from_result
from benchflow.trajectories.results import write_rollout_results_jsonl
from tests.test_automatic_review_scoring import _result, _score

OVERSIZED = [10**1000, -(10**1000)]


@pytest.mark.parametrize("oversized", OVERSIZED, ids=["positive", "negative"])
def test_mean_scored_reward_excludes_oversized_like_non_finite(oversized):
    """Guards the scoring aggregate left unhandled by the oversized-reward change."""
    results = [{"rewards": {"reward": oversized}}, {"rewards": {"reward": 0.5}}]
    assert mean_scored_reward(results) == pytest.approx(0.5)
    assert score_summary_fields(results)["mean_reward"] == pytest.approx(0.5)


@pytest.mark.parametrize("oversized", OVERSIZED, ids=["positive", "negative"])
def test_oversized_reward_disagrees_with_scoring_verdict(oversized):
    """Guards review/outcome.py consistency checks left by the oversized-reward change.

    A stale or corrupt reward envelope must be rejected as a ``ValueError``
    (which every caller already handles), not escape as ``OverflowError``.
    """
    result = _result(_score())
    result["rewards"]["reward"] = oversized
    with pytest.raises(ValueError, match="disagrees"):
        scoring_from_result(result)
    assert extract_reward(result) is None
    assert classify_score_outcome(result) == "verifier_errored"
    assert classify_audit_outcome(result) == "verifier_errored"
    assert deterministic_pass(result) is False
    assert mean_scored_reward([result, {"rewards": {"reward": 0.25}}]) == 0.25


def _written_row(tmp_path: Path, name: str, rewards: dict) -> dict:
    rollout_dir = tmp_path / name
    write_rollout_results_jsonl(
        rollout_dir,
        task_name="physics",
        rollout_name="trial",
        agent="codex",
        agent_name="codex",
        model="terra",
        n_tool_calls=0,
        prompts=[],
        trajectory=[],
        partial_trajectory=False,
        rewards=rewards,
        error=None,
        verifier_error=None,
    )
    return json.loads((rollout_dir / "results.jsonl").read_text())


@pytest.mark.parametrize("oversized", OVERSIZED, ids=["positive", "negative"])
def test_results_jsonl_emits_oversized_reward_like_infinity(tmp_path, oversized):
    """Guards trajectories/results.py export left unhandled by the oversized-reward change.

    Non-finite rewards and metrics are already written as ``null``; an
    oversized integer is the same value class and must follow that path.
    """
    sign = 1 if oversized > 0 else -1
    oversized_row = _written_row(
        tmp_path,
        "oversized",
        {
            "reward": oversized,
            "partial": oversized,
            "metrics": {"nested": oversized},
            "rubric": [{"name": "criterion", "score": oversized}],
        },
    )
    infinite_row = _written_row(
        tmp_path,
        "infinite",
        {
            "reward": sign * float("inf"),
            "partial": sign * float("inf"),
            "metrics": {"nested": sign * float("inf")},
            "rubric": [{"name": "criterion", "score": sign * float("inf")}],
        },
    )
    assert oversized_row["reward"] is None
    assert oversized_row["score"] is None
    for key in ("reward", "partial", "nested", "criterion"):
        assert key in oversized_row["metrics"]
        assert oversized_row["metrics"][key] is None
    assert oversized_row["metrics"] == infinite_row["metrics"]
    assert oversized_row["reward"] == infinite_row["reward"]
