"""Tests for the composable Rubric + RewardFunc protocol (ENG-49)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from benchflow.rewards.builtins import (
    CodeExecRewardFunc,
    LLMJudgeRewardFunc,
    StringMatchRewardFunc,
    TestRewardFunc,
)
from benchflow.rewards.events import RewardEvent
from benchflow.rewards.protocol import RewardFunc, VerifyResult
from benchflow.rewards.rubric import Rubric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ConstRewardFunc:
    """Returns a constant score — useful for deterministic tests."""

    def __init__(self, value: float) -> None:
        self._value = value

    async def score(self, rollout_dir: Path) -> float:
        return self._value


class FailingRewardFunc:
    """Always raises — used to verify error handling in Rubric."""

    async def score(self, rollout_dir: Path) -> float:
        raise RuntimeError("deliberate failure")


# ---------------------------------------------------------------------------
# RewardEvent construction
# ---------------------------------------------------------------------------


class TestRewardEventConstruction:
    def test_basic_fields(self) -> None:
        ev = RewardEvent(type="terminal", reward=0.75, source="TestRewardFunc")
        assert ev.type == "terminal"
        assert ev.reward == 0.75
        assert ev.source == "TestRewardFunc"
        assert ev.step is None
        assert ev.ts  # auto-filled ISO timestamp

    def test_dense_event_with_step(self) -> None:
        ev = RewardEvent(type="dense", reward=0.5, source="CustomFunc", step=3)
        assert ev.step == 3
        assert ev.type == "dense"


# ---------------------------------------------------------------------------
# VerifyResult construction
# ---------------------------------------------------------------------------


class TestVerifyResultConstruction:
    def test_defaults(self) -> None:
        vr = VerifyResult(reward=1.0)
        assert vr.reward == 1.0
        assert vr.items == {}
        assert vr.events == []
        assert vr.error is None

    def test_with_items_and_events(self) -> None:
        ev = RewardEvent(type="terminal", reward=1.0, source="test")
        vr = VerifyResult(
            reward=0.8,
            items={"test": 1.0, "match": 0.6},
            events=[ev],
            error=None,
        )
        assert len(vr.events) == 1
        assert vr.items["match"] == 0.6

    def test_with_error(self) -> None:
        vr = VerifyResult(reward=0.0, error="scoring failed")
        assert vr.error == "scoring failed"


# ---------------------------------------------------------------------------
# Rubric: equal weights
# ---------------------------------------------------------------------------


class TestRubricEqualWeights:
    def test_single_func(self) -> None:
        rubric = Rubric(reward_funcs=[ConstRewardFunc(0.8)])
        result = asyncio.run(rubric.score(Path("/unused")))
        assert result.reward == pytest.approx(0.8)
        assert len(result.items) == 1

    def test_multiple_funcs_equal_weight(self) -> None:
        rubric = Rubric(
            reward_funcs=[ConstRewardFunc(1.0), ConstRewardFunc(0.0)],
        )
        result = asyncio.run(rubric.score(Path("/unused")))
        assert result.reward == pytest.approx(0.5)
        assert result.items["ConstRewardFunc"] in (0.0, 1.0)

    def test_empty_rubric(self) -> None:
        rubric = Rubric(reward_funcs=[])
        result = asyncio.run(rubric.score(Path("/unused")))
        assert result.reward == 0.0


# ---------------------------------------------------------------------------
# Rubric: custom weights
# ---------------------------------------------------------------------------


class TestRubricCustomWeights:
    def test_custom_weights(self) -> None:
        rubric = Rubric(
            reward_funcs=[ConstRewardFunc(1.0), ConstRewardFunc(0.5)],
            weights=[0.7, 0.3],
        )
        result = asyncio.run(rubric.score(Path("/unused")))
        assert result.reward == pytest.approx(0.7 * 1.0 + 0.3 * 0.5)

    def test_weight_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            Rubric(
                reward_funcs=[ConstRewardFunc(1.0)],
                weights=[0.5, 0.5],
            )


# ---------------------------------------------------------------------------
# Rubric: error handling
# ---------------------------------------------------------------------------


class TestRubricErrorHandling:
    def test_failing_func_returns_zero_and_reports_error(self) -> None:
        rubric = Rubric(
            reward_funcs=[ConstRewardFunc(1.0), FailingRewardFunc()],
        )
        result = asyncio.run(rubric.score(Path("/unused")))
        assert result.error is not None
        assert "deliberate failure" in result.error
        assert result.items["FailingRewardFunc"] == 0.0
        assert result.items["ConstRewardFunc"] == 1.0

    def test_all_failing_funcs(self) -> None:
        rubric = Rubric(reward_funcs=[FailingRewardFunc()])
        result = asyncio.run(rubric.score(Path("/unused")))
        assert result.reward == 0.0
        assert result.error is not None


# ---------------------------------------------------------------------------
# Rubric: events collection
# ---------------------------------------------------------------------------


class TestRubricEvents:
    def test_events_collected_from_successful_funcs(self) -> None:
        rubric = Rubric(
            reward_funcs=[ConstRewardFunc(0.9), ConstRewardFunc(0.1)],
        )
        result = asyncio.run(rubric.score(Path("/unused")))
        assert len(result.events) == 2
        assert all(ev.type == "terminal" for ev in result.events)
        sources = {ev.source for ev in result.events}
        assert "ConstRewardFunc" in sources


# ---------------------------------------------------------------------------
# TestRewardFunc (backward compat)
# ---------------------------------------------------------------------------


class TestTestRewardFunc:
    def test_reads_reward_txt(self, tmp_path: Path) -> None:
        (tmp_path / "reward.txt").write_text("0.85\n")
        func = TestRewardFunc()
        score = asyncio.run(func.score(tmp_path))
        assert score == pytest.approx(0.85)

    def test_missing_reward_txt(self, tmp_path: Path) -> None:
        func = TestRewardFunc()
        score = asyncio.run(func.score(tmp_path))
        assert score == 0.0

    def test_empty_reward_txt(self, tmp_path: Path) -> None:
        (tmp_path / "reward.txt").write_text("")
        func = TestRewardFunc()
        score = asyncio.run(func.score(tmp_path))
        assert score == 0.0

    def test_invalid_reward_txt(self, tmp_path: Path) -> None:
        (tmp_path / "reward.txt").write_text("not-a-number\n")
        func = TestRewardFunc()
        score = asyncio.run(func.score(tmp_path))
        assert score == 0.0

    def test_multiline_reward_txt(self, tmp_path: Path) -> None:
        (tmp_path / "reward.txt").write_text("1.0\nsome extra info\n")
        func = TestRewardFunc()
        score = asyncio.run(func.score(tmp_path))
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# StringMatchRewardFunc
# ---------------------------------------------------------------------------


class TestStringMatchRewardFunc:
    def test_exact_match(self, tmp_path: Path) -> None:
        (tmp_path / "answer.txt").write_text("hello world")
        func = StringMatchRewardFunc(expected="hello world")
        assert asyncio.run(func.score(tmp_path)) == 1.0

    def test_exact_mismatch(self, tmp_path: Path) -> None:
        (tmp_path / "answer.txt").write_text("Hello World")
        func = StringMatchRewardFunc(expected="hello world")
        assert asyncio.run(func.score(tmp_path)) == 0.0

    def test_fuzzy_match(self, tmp_path: Path) -> None:
        (tmp_path / "answer.txt").write_text("The answer is Hello World!")
        func = StringMatchRewardFunc(expected="hello world", fuzzy=True)
        assert asyncio.run(func.score(tmp_path)) == 1.0

    def test_fuzzy_mismatch(self, tmp_path: Path) -> None:
        (tmp_path / "answer.txt").write_text("something else")
        func = StringMatchRewardFunc(expected="hello world", fuzzy=True)
        assert asyncio.run(func.score(tmp_path)) == 0.0

    def test_missing_answer_txt(self, tmp_path: Path) -> None:
        func = StringMatchRewardFunc(expected="hello")
        assert asyncio.run(func.score(tmp_path)) == 0.0


# ---------------------------------------------------------------------------
# CodeExecRewardFunc
# ---------------------------------------------------------------------------


class TestCodeExecRewardFunc:
    def test_custom_function(self, tmp_path: Path) -> None:
        (tmp_path / "result.txt").write_text("42")

        def scorer(d: Path) -> float:
            return float((d / "result.txt").read_text())

        func = CodeExecRewardFunc(scorer)
        assert asyncio.run(func.score(tmp_path)) == pytest.approx(42.0)

    def test_returns_zero(self, tmp_path: Path) -> None:
        func = CodeExecRewardFunc(lambda d: 0.0)
        assert asyncio.run(func.score(tmp_path)) == 0.0


# ---------------------------------------------------------------------------
# RewardFunc protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_builtins_satisfy_protocol(self) -> None:
        assert isinstance(TestRewardFunc(), RewardFunc)
        assert isinstance(StringMatchRewardFunc(expected="x"), RewardFunc)
        assert isinstance(CodeExecRewardFunc(func=lambda d: 0.0), RewardFunc)
        assert isinstance(LLMJudgeRewardFunc(prompt="rate it"), RewardFunc)

    def test_const_satisfies_protocol(self) -> None:
        assert isinstance(ConstRewardFunc(0.5), RewardFunc)


# ---------------------------------------------------------------------------
# Backward compatibility: default Rubric wraps TestRewardFunc
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_default_rubric_uses_test_reward_func(self, tmp_path: Path) -> None:
        """Tasks without a rubric config get Rubric([TestRewardFunc()])."""
        (tmp_path / "reward.txt").write_text("0.75\n")
        rubric = Rubric(reward_funcs=[TestRewardFunc()])
        result = asyncio.run(rubric.score(tmp_path))
        assert result.reward == pytest.approx(0.75)
        assert "TestRewardFunc" in result.items


# ---------------------------------------------------------------------------
# RunResult.reward_events field
# ---------------------------------------------------------------------------


class TestRunResultRewardEvents:
    def test_default_none(self) -> None:
        from benchflow.models import RunResult

        rr = RunResult(task_name="t")
        assert rr.reward_events is None

    def test_with_events(self) -> None:
        from benchflow.models import RunResult

        ev = RewardEvent(type="terminal", reward=1.0, source="test")
        rr = RunResult(task_name="t", reward_events=[ev])
        assert rr.reward_events is not None
        assert len(rr.reward_events) == 1
        assert rr.reward_events[0].reward == 1.0


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------


class TestReexports:
    def test_reward_types_importable_from_benchflow(self) -> None:
        import benchflow

        assert benchflow.Rubric.__module__ == "benchflow.rewards.rubric"
        assert benchflow.RewardEvent.__module__ == "benchflow.rewards.events"
        assert benchflow.VerifyResult.__module__ == "benchflow.rewards.protocol"
        assert benchflow.TestRewardFunc.__module__ == "benchflow.rewards.builtins"
        assert benchflow.LLMJudgeRewardFunc.__module__ == "benchflow.rewards.builtins"
        assert benchflow.StringMatchRewardFunc.__module__ == "benchflow.rewards.builtins"
        assert benchflow.CodeExecRewardFunc.__module__ == "benchflow.rewards.builtins"
        assert hasattr(benchflow, "RewardFunc")
