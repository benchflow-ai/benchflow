"""Only the reward is held to the reward range (EN-278).

Guards the EN-278 fix against the rc3 behaviour introduced by #1178/#1192:
every top-level number in ``verifier/reward.json`` was a metric held to the
reward range, so a count beside the reward failed the trial with
``reward out of range``. DeepSWE 1.1's verifier writes
``{"reward": 0, "f2p_total": 20, ...}`` and every trial of it scored 0 as a
verifier problem under 0.10.0rc3. Beside a numeric ``reward`` with no
aggregate declared, such a number is now a detail: kept as written, noted in
the log, never scored. An out-of-range ``reward`` still fails, and so does an
out-of-range number an aggregate makes the reward from.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.rewards.validation import RewardOutOfRange, validate_reward_map
from benchflow.rollout import _ensure_canonical_rewards
from benchflow.task.verifier import VerifierOutputParseError
from tests.test_reward_record import _raw, _verifier

# DeepSWE 1.1's reward.json for a failed trial, as its verifier writes it.
DEEPSWE = {
    "reward": 0,
    "f2p_total": 20,
    "f2p_passed": 0,
    "p2p_total": 3,
    "p2p_passed": 3,
    "f2p": 0.0,
    "p2p": 1.0,
    "partial": 0.13043478260869565,
}


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_REWARD_LENIENT", raising=False)


def _parse(tmp_path: Path, payload: dict, **kwargs) -> dict:
    (tmp_path / "reward.json").write_text(json.dumps(payload))
    return _verifier(tmp_path)._parse_reward_json_with_text_compat(**kwargs)


def test_deepswe_failed_trial_scores_zero_without_an_error(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="benchflow.rewards.validation"):
        rewards = _parse(tmp_path, DEEPSWE)
    assert rewards == DEEPSWE
    assert rewards["reward"] == 0
    # The final gate before scoring agrees.
    assert _ensure_canonical_rewards(rewards) == DEEPSWE
    note = caplog.text
    assert "f2p_total = 20" in note and "p2p_total = 3" in note
    assert "p2p_passed = 3" in note and "not scored" in note
    # In-range numbers are not listed as details.
    assert "partial" not in note and "f2p_passed" not in note


def test_deepswe_solved_trial_scores_one(tmp_path):
    solved = {**DEEPSWE, "reward": 1, "f2p_passed": 20, "f2p": 1.0, "partial": 1.0}
    rewards = _parse(tmp_path, solved)
    assert rewards == solved
    assert float(rewards["reward"]) == 1.0
    assert _ensure_canonical_rewards(rewards) == solved


# researchcodebench's reward.json (dyt_forward, fractalgen_sample_orders): a line
# count beside the reward. rel-0.9.x (as 0.9.2) failed it with "code_lines = 5
# is outside [0.0, 1.0]" and the trial scored 0 (EN-737).
RESEARCHCODEBENCH = {"reward": 1.0, "code_lines": 5}


def test_researchcodebench_line_count_beside_the_reward_scores_one(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="benchflow.rewards.validation"):
        rewards = _parse(tmp_path, RESEARCHCODEBENCH)
    assert rewards == RESEARCHCODEBENCH
    assert float(rewards["reward"]) == 1.0
    assert _ensure_canonical_rewards(rewards) == RESEARCHCODEBENCH
    assert "code_lines = 5" in caplog.text and "not scored" in caplog.text


@pytest.mark.parametrize("reward", [20, 1.5, -0.1])
@pytest.mark.usefixtures("strict_reward_rule")
def test_an_out_of_range_reward_still_fails(tmp_path, reward):
    (tmp_path / "reward.json").write_text(json.dumps({**DEEPSWE, "reward": reward}))
    with pytest.raises(VerifierOutputParseError) as caught:
        _verifier(tmp_path)._parse_reward_json_with_text_compat()
    assert _raw(caught.value) == ("reward", reward, "verifier/reward.json")
    assert f"reward = {reward!r} is outside [0.0, 1.0]" in str(caught.value)


def test_a_reward_txt_score_keeps_the_counts_as_details(tmp_path):
    (tmp_path / "reward.txt").write_text("1\n")
    details = {k: v for k, v in DEEPSWE.items() if k != "reward"}
    (tmp_path / "reward.json").write_text(json.dumps(details))
    rewards = _verifier(tmp_path)._parse_reward_json_with_text_compat()
    assert rewards == {"reward": 1.0, **details}


def test_lenient_mode_keeps_the_details_too(monkeypatch):
    monkeypatch.setenv("BENCHFLOW_REWARD_LENIENT", "1")
    assert _ensure_canonical_rewards(dict(DEEPSWE)) == DEEPSWE
    assert validate_reward_map(DEEPSWE, lenient=True) == DEEPSWE


def test_a_declared_range_still_judges_the_reward_alone():
    floored = {**DEEPSWE, "reward": -1.0}
    assert validate_reward_map(floored, reward_range=(-1.0, 1.0)) == floored
    with pytest.raises(RewardOutOfRange, match=r"reward = -1\.5"):
        validate_reward_map({**DEEPSWE, "reward": -1.5}, reward_range=(-1.0, 1.0))


def test_numbers_an_aggregate_makes_the_reward_from_are_held_to_the_range():
    """No reward: an aggregate makes it from the top-level numbers."""
    with pytest.raises(RewardOutOfRange) as caught:
        validate_reward_map(
            {"f2p": 0.0, "f2p_total": 20}, aggregate_policy={"method": "mean"}
        )
    assert (caught.value.key, caught.value.value) == ("f2p_total", 20)


@pytest.mark.parametrize(
    ("rewards", "policy"),
    [
        # Fable's audit of 1633fea33: exactly this map raised RewardOutOfRange.
        ({"reward": 1, "f2p_total": 20, "aggregate": {"method": "mean"}}, None),
        ({**DEEPSWE, "aggregate": {"method": "mean"}}, None),
        (DEEPSWE, {"method": "mean"}),
    ],
)
def test_a_count_beside_the_reward_is_a_detail_under_an_aggregate(
    tmp_path, rewards, policy
):
    """An aggregate makes or checks the reward only from the numbers in the
    range, so a count beside the reward is never one of its components."""
    assert validate_reward_map(rewards, aggregate_policy=policy) == rewards
    parsed = _parse(tmp_path, rewards, aggregate_policy=policy)
    assert (parsed["reward"], parsed["f2p_total"]) == (rewards["reward"], 20)


def test_a_strict_aggregate_checks_the_reward_against_the_numbers_in_range():
    from benchflow.rewards.validation import apply_aggregate_policy

    rewards = {
        "reward": 1,
        "f2p": 1.0,
        "f2p_total": 20,
        "aggregate": {"method": "mean"},
    }
    assert apply_aggregate_policy(rewards, strict=True)["reward"] == 1.0
    with pytest.raises(ValueError, match="does not match criteria aggregate"):
        apply_aggregate_policy({**rewards, "f2p": 0.0}, strict=True)


def test_metrics_entries_are_still_held_to_the_range():
    with pytest.raises(RewardOutOfRange, match=r"metrics\.speedup = 2\.5"):
        validate_reward_map({"reward": 1.0, "metrics": {"speedup": 2.5}})


def test_a_non_finite_detail_is_still_refused():
    with pytest.raises(ValueError, match="invalid reward value for 'f2p_total'"):
        validate_reward_map({"reward": 0.0, "f2p_total": float("inf")})


@pytest.mark.asyncio
async def test_the_trial_records_the_details(tmp_path):
    """Through the rollout's verify and result writer: scored 0, no verifier
    error, every detail in ``result.json``'s ``rewards``."""
    from benchflow.diagnostics import RolloutDiagnostics
    from benchflow.rollout._results import _build_rollout_result
    from benchflow.sdk import SDK

    task = MagicMock()
    task.config.verifier.timeout_sec = 5
    task.config.verifier.env = None
    task.config.verifier.reward_range = None
    paths = MagicMock()
    paths.verifier_dir = tmp_path / "verifier"
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
    verifier = MagicMock()
    verifier.verify = AsyncMock(return_value=MagicMock(rewards=dict(DEEPSWE)))
    with (
        patch("benchflow.task.Verifier", return_value=verifier),
        patch(
            "benchflow.sandbox.lockdown.harden_before_verify", new_callable=AsyncMock
        ),
    ):
        rewards, verifier_error, diagnostic = await SDK()._verify(env, task, paths, {})
    assert (rewards, verifier_error, diagnostic) == (DEEPSWE, None, None)

    trial = tmp_path / "trial"
    trial.mkdir()
    _build_rollout_result(
        trial,
        task_name="deepswe",
        rollout_name="deepswe__1",
        agent="oracle",
        agent_name="oracle",
        model=None,
        n_tool_calls=0,
        prompts=[],
        error=None,
        verifier_error=verifier_error,
        trajectory=[],
        partial_trajectory=False,
        rewards=rewards,
        started_at=datetime.now(),
        timing={},
        diagnostics=RolloutDiagnostics(),
    )
    saved = json.loads((trial / "result.json").read_text())
    assert saved["rewards"] == DEEPSWE
    assert saved.get("verifier_error") is None
    assert saved.get("reward_out_of_range_info") is None


def test_exports_take_every_key_whatever_its_value():
    """EN-278 re-review: ffa9afe05 left a number out of the exported metrics
    by its value, so a key such as f2p_passed was exported from a trial where
    it was 0 and dropped from one where it was 20, and a job's mean of it ran
    over the trials that happened to fit. Export is by key: every top-level
    number of the reward map, as before EN-278; the map in result.json still
    holds the details as the verifier wrote them."""
    from benchflow.trajectories.export import reward_map_to_verify_result
    from benchflow.trajectories.results import _metrics_from_rewards

    failed = DEEPSWE
    solved = {**DEEPSWE, "reward": 1, "f2p_passed": 20, "f2p": 1.0, "partial": 1.0}
    keys = set(DEEPSWE) - {"reward"}
    for rewards in (failed, solved):
        assert set(reward_map_to_verify_result(rewards).items) == keys
        metrics = _metrics_from_rewards(rewards, n_tool_calls=0, n_prompts=1)
        assert set(metrics) == keys | {"reward", "n_tool_calls", "n_prompts"}
        assert metrics["f2p_passed"] == rewards["f2p_passed"]
