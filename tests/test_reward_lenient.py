"""Tests for BF-3: lenient reward.json validation (legacy rich-reward flow).

Guards the fix from benchflow-ai/benchflow PR #670 (BF-3).

The strict contract (the default) rejects any non-numeric metric, and an
unrecognized non-numeric top-level key in a map with no numeric ``reward``.
Beside a numeric ``reward`` such a key (the task.toml-era
``{"reward": 1.0, "done": true}``) is kept in both modes since #1178 (round
PY8, V4). Lenient mode (opt-in via ``BENCHFLOW_REWARD_LENIENT=1``) drops the
rest with a single warning while still requiring a usable scalar ``reward``.
"""

from __future__ import annotations

import warnings

import pytest

from benchflow.rewards.validation import (
    RewardOutOfRange,
    reward_lenient_from_env,
    validate_reward_map,
)


class TestStrictDefaultUnchanged:
    def test_strict_keeps_legacy_done_key(self) -> None:
        # The classic test.sh→reward.json payload scores, its extra key kept
        # (#1178; before, strict mode threw the whole reward away).
        assert validate_reward_map(
            {"reward": 1.0, "done": True}, source="reward JSON"
        ) == {"reward": 1.0, "done": True}

    def test_strict_rejects_non_numeric_metric(self) -> None:
        with pytest.raises(ValueError, match="invalid metric value"):
            validate_reward_map(
                {"reward": 1.0, "metrics": {"acc": 0.5, "label": "good"}},
                source="reward JSON",
            )

    def test_strict_is_the_default(self) -> None:
        # No lenient kwarg → the strict raise.
        with pytest.raises(ValueError):
            validate_reward_map({"reward": 1.0, "metrics": {"label": "good"}})

    def test_strict_still_accepts_clean_map(self) -> None:
        parsed = validate_reward_map(
            {"reward": 1.0, "metrics": {"acc": 0.5}}, source="reward JSON"
        )
        assert parsed == {"reward": 1.0, "metrics": {"acc": 0.5}}


class TestLenientDropsAndWarns:
    def test_lenient_drops_non_numeric_metric_keeps_reward_and_done(self) -> None:
        with pytest.warns(UserWarning, match="lenient mode") as record:
            parsed = validate_reward_map(
                {
                    "reward": 1.0,
                    "done": True,
                    "metrics": {"accuracy": 0.5, "label": "good", "note": None},
                },
                source="reward JSON",
                lenient=True,
            )
        assert parsed == {"reward": 1.0, "done": True, "metrics": {"accuracy": 0.5}}
        # A single aggregated warning listing every dropped key.
        assert len(record) == 1
        message = str(record[0].message)
        assert "done" not in message
        assert "metrics.label" in message and "metrics.note" in message

    def test_lenient_keeps_recognized_structured_keys(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning when nothing is dropped
            parsed = validate_reward_map(
                {"reward": 0.4, "metrics": {"acc": 0.4}, "reason": "ok"},
                source="reward JSON",
                lenient=True,
            )
        assert parsed == {"reward": 0.4, "metrics": {"acc": 0.4}, "reason": "ok"}

    def test_lenient_derives_reward_from_score_alias(self) -> None:
        parsed = validate_reward_map(
            {"score": 0.75, "done": True}, source="reward JSON", lenient=True
        )
        assert parsed == {"reward": 0.75, "done": True}

    def test_lenient_derives_reward_from_rewards_alias(self) -> None:
        parsed = validate_reward_map(
            {"rewards": 0.5}, source="reward JSON", lenient=True
        )
        assert parsed == {"reward": 0.5}

    def test_lenient_drops_unusable_reward_then_derives(self) -> None:
        with pytest.warns(UserWarning, match="reward"):
            parsed = validate_reward_map(
                {"reward": "n/a", "score": 0.9}, source="reward JSON", lenient=True
            )
        assert parsed == {"reward": 0.9}

    def test_lenient_never_swaps_an_out_of_range_reward_for_an_alias(self) -> None:
        # An out-of-range reward is kept raw and the trial unscored;
        # scoring ``score`` in its place would be a score nobody gave.
        with pytest.raises(RewardOutOfRange, match=r"reward = 5\.0 is outside"):
            validate_reward_map(
                {"reward": 5.0, "score": 0.9}, source="reward JSON", lenient=True
            )

    def test_lenient_drops_whole_non_mapping_metrics(self) -> None:
        with pytest.warns(UserWarning, match="metrics"):
            parsed = validate_reward_map(
                {"reward": 1.0, "metrics": "n/a"}, source="reward JSON", lenient=True
            )
        assert parsed == {"reward": 1.0}

    def test_lenient_still_requires_usable_reward(self) -> None:
        # No reward, no alias, no aggregate-able metric → still raises.
        with pytest.raises(ValueError, match="without numeric 'reward'"):
            validate_reward_map(
                {"done": True, "note": "x"}, source="reward JSON", lenient=True
            )

    def test_lenient_allows_metrics_plus_aggregate_policy_without_reward(self) -> None:
        # Lenient must not break the structured metrics+aggregate path: the
        # bookkeeping ``done`` is dropped (and warned) while the metrics map is
        # accepted for downstream aggregation.
        with pytest.warns(UserWarning, match="done"):
            parsed = validate_reward_map(
                {"metrics": {"a": 1.0, "b": 0.0}, "done": True},
                source="reward JSON",
                aggregate_policy={"field": "reward", "method": "mean"},
                lenient=True,
            )
        assert parsed == {"metrics": {"a": 1.0, "b": 0.0}}

    def test_lenient_no_warning_when_nothing_dropped(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parsed = validate_reward_map(
                {"reward": 1.0}, source="reward JSON", lenient=True
            )
        assert parsed == {"reward": 1.0}


class TestRewardLenientFromEnv:
    def test_unset_is_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BENCHFLOW_REWARD_LENIENT", raising=False)
        assert reward_lenient_from_env() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_truthy_values_enable(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("BENCHFLOW_REWARD_LENIENT", value)
        assert reward_lenient_from_env() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "nope"])
    def test_falsy_values_stay_strict(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("BENCHFLOW_REWARD_LENIENT", value)
        assert reward_lenient_from_env() is False
