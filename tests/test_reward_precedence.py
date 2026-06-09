"""Reward-precedence characterization tests (MEDIUM: reward.json vs reward.txt).

The RFC changed the verifier so that ``reward.json`` is the primary reward
source (``Verifier.verify`` reads it first when present) and added a strict
cross-check against ``reward.txt`` when both files exist: a scalar ``reward``
in the JSON that disagrees with ``reward.txt`` beyond a tiny float tolerance
(``math.isclose(..., abs_tol=1e-9)``) raises ``VerifierOutputParseError``.

These tests LOCK that behavior by driving the real reward-parsing helpers on a
``Verifier`` built over real ``RolloutPaths`` (the only state the parsers
touch). They are characterization tests over existing code — no source change.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from benchflow.task.paths import RolloutPaths
from benchflow.task.verifier import Verifier, VerifierOutputParseError


def _make_verifier(tmp_path):
    """Build a real Verifier whose reward files live under a real
    ``RolloutPaths``. The reward parsers only read ``reward_text_path`` /
    ``reward_json_path``, so task and sandbox can be bare mocks."""
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()  # creates verifier_dir (parent of reward files)
    return Verifier(
        task=MagicMock(),
        rollout_paths=rollout_paths,
        sandbox=MagicMock(),
    ), rollout_paths


def _write_json(rollout_paths, payload) -> None:
    rollout_paths.reward_json_path.write_text(json.dumps(payload))


def _write_text(rollout_paths, value) -> None:
    rollout_paths.reward_text_path.write_text(str(value))


# reward.json alone parses (it is the primary source)


def test_reward_json_only_parses(tmp_path):
    verifier, rollout_paths = _make_verifier(tmp_path)
    _write_json(rollout_paths, {"reward": 0.75})
    assert rollout_paths.reward_json_path.exists()
    assert not rollout_paths.reward_text_path.exists()

    rewards = verifier._parse_reward_json_with_text_compat()
    assert rewards["reward"] == 0.75


def test_reward_json_only_with_metrics_parses(tmp_path):
    """A richer reward.json (scalar + metrics) is preserved, not flattened."""
    verifier, rollout_paths = _make_verifier(tmp_path)
    _write_json(
        rollout_paths,
        {"reward": 1.0, "metrics": {"tests_passed": 1.0, "lint_ok": 1.0}},
    )
    rewards = verifier._parse_reward_json_with_text_compat()
    assert rewards["reward"] == 1.0
    assert rewards["metrics"] == {"tests_passed": 1.0, "lint_ok": 1.0}


# reward.txt alone parses


def test_reward_text_only_parses(tmp_path):
    verifier, rollout_paths = _make_verifier(tmp_path)
    _write_text(rollout_paths, "0.5")
    assert rollout_paths.reward_text_path.exists()
    assert not rollout_paths.reward_json_path.exists()

    rewards = verifier._parse_reward_text()
    assert rewards == {"reward": 0.5}


# both present and AGREE -> ok


def test_reward_json_and_text_agree_ok(tmp_path):
    verifier, rollout_paths = _make_verifier(tmp_path)
    _write_json(rollout_paths, {"reward": 0.25})
    _write_text(rollout_paths, "0.25")

    rewards = verifier._parse_reward_json_with_text_compat()
    assert rewards["reward"] == 0.25


def test_reward_json_and_text_agree_within_float_tolerance_ok(tmp_path):
    """Values within ``abs_tol=1e-9`` are treated as agreeing."""
    verifier, rollout_paths = _make_verifier(tmp_path)
    _write_json(rollout_paths, {"reward": 0.25})
    _write_text(rollout_paths, "0.2500000000")  # < 1e-9 apart

    rewards = verifier._parse_reward_json_with_text_compat()
    assert rewards["reward"] == 0.25


# both present and DISAGREE beyond tolerance -> raises


def test_reward_json_and_text_disagree_raises(tmp_path):
    verifier, rollout_paths = _make_verifier(tmp_path)
    _write_json(rollout_paths, {"reward": 0.9})
    _write_text(rollout_paths, "0.1")

    with pytest.raises(VerifierOutputParseError) as exc:
        verifier._parse_reward_json_with_text_compat()
    msg = str(exc.value)
    assert "does not match" in msg
    assert "reward.txt" in msg


def test_reward_json_and_text_disagree_beyond_tolerance_raises(tmp_path):
    """Just past ``abs_tol=1e-9`` must raise, guarding the cross-check bound."""
    verifier, rollout_paths = _make_verifier(tmp_path)
    _write_json(rollout_paths, {"reward": 0.5})
    _write_text(rollout_paths, "0.500001")  # ~1e-6 apart, well beyond 1e-9

    with pytest.raises(VerifierOutputParseError):
        verifier._parse_reward_json_with_text_compat()
