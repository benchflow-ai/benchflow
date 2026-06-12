"""Tests for BF-8: task-declared reward range (``[verifier] reward_range``).

Guards the fix tracked as benchflow-ai/benchflow#675 (BF-8).

skillsgym-style safety tasks floor reward at -1.0 — deliberately BELOW
doing-nothing 0.0 — but BenchFlow's reward contract is hard-coded [0, 1] at
every chokepoint, so unsafe runs become verifier ERRORS instead of scored
-1.0. The fix is a task-config opt-in: ``[verifier] reward_range = [lo, hi]``
may only WIDEN the canonical [0, 1] contract (lo <= 0.0, hi >= 1.0, lo < hi,
finite). Undeclared tasks keep the strict [0, 1] contract byte-for-byte, and
lenient mode (BF-3) never accepts negatives on its own.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from benchflow.rewards.validation import (
    apply_aggregate_policy,
    declared_reward_range,
    validate_declared_reward_range,
    validate_reward_map,
)
from benchflow.rollout import _ensure_canonical_rewards
from benchflow.task import TaskConfig, TaskDocument
from benchflow.task.verifier import Verifier, VerifierOutputParseError


def _task_with_config(verifier: dict | None = None) -> MagicMock:
    """A Verifier-shaped task fake carrying a REAL parsed TaskConfig."""
    task = MagicMock()
    task.config = TaskConfig.model_validate({"verifier": verifier or {}})
    return task


def _verifier(task: MagicMock, reward_text_path: Path) -> Verifier:
    rollout_paths = MagicMock()
    rollout_paths.reward_text_path = reward_text_path
    return Verifier(task=task, rollout_paths=rollout_paths, sandbox=MagicMock())


class TestValidateDeclaredRewardRange:
    """The widen-only rule, enforced once at config parse."""

    def test_accepts_safety_floor_range(self) -> None:
        assert validate_declared_reward_range([-1.0, 1.0]) == (-1.0, 1.0)

    def test_accepts_canonical_range_and_coerces_ints(self) -> None:
        lo, hi = validate_declared_reward_range([0, 1])
        assert (lo, hi) == (0.0, 1.0)
        assert isinstance(lo, float) and isinstance(hi, float)

    @pytest.mark.parametrize(
        "value",
        [[-1.0], [-1.0, 0.0, 1.0], "[-1.0, 1.0]", {"lo": -1.0, "hi": 1.0}, None],
    )
    def test_rejects_non_pair(self, value: object) -> None:
        with pytest.raises(ValueError, match=r"\[lo, hi\] pair of numbers"):
            validate_declared_reward_range(value)

    @pytest.mark.parametrize("value", [[True, 1.0], [-1.0, "1.0"]])
    def test_rejects_non_numeric_members(self, value: list) -> None:
        with pytest.raises(ValueError, match=r"pair of numbers"):
            validate_declared_reward_range(value)

    @pytest.mark.parametrize(
        "value",
        [[float("-inf"), 1.0], [-1.0, float("nan")], [1.0, -1.0], [0.0, 0.0]],
    )
    def test_rejects_non_finite_or_inverted(self, value: list) -> None:
        with pytest.raises(ValueError, match="finite with lo < hi"):
            validate_declared_reward_range(value)

    @pytest.mark.parametrize(
        "value",
        [
            [0.2, 0.8],  # narrowed: drops both anchors
            [0.5, 2.0],  # shifted up: 0 ("did nothing") no longer scorable
            [-2.0, 0.5],  # shifted down: 1 ("solved") no longer scorable
        ],
    )
    def test_rejects_narrowing_or_shifting_the_contract(self, value: list) -> None:
        with pytest.raises(ValueError, match="only widen"):
            validate_declared_reward_range(value)


class TestVerifierConfigParse:
    """``reward_range`` enters via task.toml AND task.md frontmatter."""

    def test_task_toml_round_trip(self) -> None:
        config = TaskConfig.model_validate_toml(
            "[verifier]\nreward_range = [-1.0, 1.0]\ntimeout_sec = 60\n"
        )
        assert config.verifier.reward_range == (-1.0, 1.0)
        reparsed = TaskConfig.model_validate_toml(config.model_dump_toml())
        assert reparsed.verifier.reward_range == (-1.0, 1.0)

    def test_task_toml_undeclared_stays_none_and_is_not_serialized(self) -> None:
        config = TaskConfig.model_validate_toml("[verifier]\ntimeout_sec = 60\n")
        assert config.verifier.reward_range is None
        assert "reward_range" not in config.model_dump_toml()

    def test_task_md_frontmatter_round_trip(self) -> None:
        document = TaskDocument.from_text(
            "---\n"
            "verifier:\n"
            "  reward_range: [-1.0, 1.0]\n"
            "---\n\n"
            "## prompt\n\n"
            "Do the thing safely.\n"
        )
        assert document.config.verifier.reward_range == (-1.0, 1.0)

    @pytest.mark.parametrize(
        "toml_range",
        ["[0.2, 0.8]", "[1.0, -1.0]", "[-1.0]", "[-inf, 1.0]", '["a", 1.0]'],
    )
    def test_malformed_range_rejected_at_parse(self, toml_range: str) -> None:
        with pytest.raises(ValidationError):
            TaskConfig.model_validate_toml(f"[verifier]\nreward_range = {toml_range}\n")


class TestValidateRewardMapRange:
    def test_undeclared_still_rejects_negative_reward(self) -> None:
        # The pre-BF-8 strict contract, byte-for-byte (message included).
        with pytest.raises(ValueError, match=re.escape("between 0.0 and 1.0")):
            validate_reward_map({"reward": -1.0}, source="reward JSON")

    def test_declared_range_accepts_safety_floor(self) -> None:
        parsed = validate_reward_map(
            {"reward": -1.0}, source="reward JSON", reward_range=(-1.0, 1.0)
        )
        assert parsed == {"reward": -1.0}

    def test_declared_range_still_rejects_below_floor(self) -> None:
        with pytest.raises(ValueError, match=re.escape("between -1.0 and 1.0")):
            validate_reward_map(
                {"reward": -1.5}, source="reward JSON", reward_range=(-1.0, 1.0)
            )

    def test_declared_range_widens_metrics(self) -> None:
        with pytest.raises(ValueError, match="invalid metric value"):
            validate_reward_map(
                {"reward": 1.0, "metrics": {"safety_floor": -1.0}},
                source="reward JSON",
            )
        parsed = validate_reward_map(
            {"reward": -1.0, "metrics": {"safety_floor": -1.0}},
            source="reward JSON",
            reward_range=(-1.0, 1.0),
        )
        assert parsed["metrics"] == {"safety_floor": -1.0}

    def test_lenient_alone_still_rejects_negative_reward(self) -> None:
        # BF-3 lenient drops the out-of-range reward, then fails on the
        # missing scalar — no silent contract erosion via lenient mode.
        with pytest.raises(ValueError, match="missing numeric 'reward'"):
            validate_reward_map({"reward": -1.0}, source="reward JSON", lenient=True)

    def test_range_composes_with_lenient(self) -> None:
        # skillsgym-shaped payload: floored reward plus a string safety
        # verdict. Lenient drops the string key (warned, BF-3 semantics);
        # the declared range keeps the -1.0 reward usable.
        with pytest.warns(UserWarning, match="safety_gate"):
            parsed = validate_reward_map(
                {"reward": -1.0, "safety_gate": "blocked"},
                source="reward JSON",
                lenient=True,
                reward_range=(-1.0, 1.0),
            )
        assert parsed == {"reward": -1.0}

    def test_range_composes_with_lenient_alias_rehoming(self) -> None:
        with pytest.warns(UserWarning, match="lenient mode"):
            parsed = validate_reward_map(
                {"score": -0.5, "done": True},
                source="reward JSON",
                lenient=True,
                reward_range=(-1.0, 1.0),
            )
        assert parsed == {"reward": -0.5}

    def test_string_safety_verdicts_survive_under_structured_keys(self) -> None:
        # The strict-mode escape hatch for non-numeric verdicts: structured
        # keys pass through untouched regardless of range.
        parsed = validate_reward_map(
            {"reward": -1.0, "details": {"safety_gate": "blocked"}},
            source="reward JSON",
            reward_range=(-1.0, 1.0),
        )
        assert parsed["details"] == {"safety_gate": "blocked"}


class TestAggregatePolicyRange:
    def test_aggregate_over_widened_metrics(self) -> None:
        parsed = apply_aggregate_policy(
            {"metrics": {"unsafe": -1.0, "safe": 1.0}},
            aggregate_policy={"field": "reward", "method": "mean"},
            source="reward JSON",
            reward_range=(-1.0, 1.0),
        )
        assert parsed["reward"] == 0.0

    def test_aggregate_without_range_filters_negative_metrics(self) -> None:
        # Undeclared range: negative metrics are not aggregable, exactly as
        # before BF-8.
        with pytest.raises(ValueError):
            apply_aggregate_policy(
                {"metrics": {"unsafe": -1.0}},
                aggregate_policy={"field": "reward", "method": "mean"},
                source="reward JSON",
            )

    def test_strict_consistency_check_accepts_floored_reward(self) -> None:
        parsed = apply_aggregate_policy(
            {"reward": -1.0, "metrics": {"unsafe": -1.0}},
            aggregate_policy={"field": "reward", "method": "mean"},
            source="reward JSON",
            strict=True,
            reward_range=(-1.0, 1.0),
        )
        assert parsed["reward"] == -1.0


class TestVerifierRewardText:
    """The reward.txt chokepoint honors the task's declared range."""

    def test_declared_range_accepts_floored_reward_txt(self, tmp_path: Path) -> None:
        reward_txt = tmp_path / "reward.txt"
        reward_txt.write_text("-1.0")
        task = _task_with_config({"reward_range": [-1.0, 1.0]})
        assert _verifier(task, reward_txt)._parse_reward_text() == {"reward": -1.0}

    def test_undeclared_task_still_rejects_negative_reward_txt(
        self, tmp_path: Path
    ) -> None:
        reward_txt = tmp_path / "reward.txt"
        reward_txt.write_text("-1.0")
        # MagicMock task: no usable declaration -> canonical [0, 1].
        with pytest.raises(
            VerifierOutputParseError, match=re.escape("between 0.0 and 1.0")
        ):
            _verifier(MagicMock(), reward_txt)._parse_reward_text()

    def test_declared_range_still_bounds_reward_txt(self, tmp_path: Path) -> None:
        reward_txt = tmp_path / "reward.txt"
        reward_txt.write_text("-1.5")
        task = _task_with_config({"reward_range": [-1.0, 1.0]})
        with pytest.raises(
            VerifierOutputParseError, match=re.escape("between -1.0 and 1.0")
        ):
            _verifier(task, reward_txt)._parse_reward_text()


class TestDeclaredRewardRangeAccessor:
    def test_real_config_task_reads_declared_range(self) -> None:
        task = _task_with_config({"reward_range": [-1.0, 1.0]})
        assert declared_reward_range(task) == (-1.0, 1.0)

    def test_real_config_task_without_declaration_reads_none(self) -> None:
        assert declared_reward_range(_task_with_config()) is None

    @pytest.mark.parametrize("task", [None, MagicMock(), object()])
    def test_unshaped_tasks_read_as_undeclared(self, task: object) -> None:
        assert declared_reward_range(task) is None


class TestRolloutFinalGate:
    """``_ensure_canonical_rewards`` — the last chokepoint before scoring."""

    def test_final_gate_honors_declared_range(self) -> None:
        task = _task_with_config({"reward_range": [-1.0, 1.0]})
        assert _ensure_canonical_rewards({"reward": -1.0}, task=task) == {
            "reward": -1.0
        }

    def test_final_gate_default_stays_strict(self) -> None:
        with pytest.raises(ValueError, match=re.escape("between 0.0 and 1.0")):
            _ensure_canonical_rewards({"reward": -1.0})
