"""Tests for the first-class LLM-as-judge verifier (ENG-55).

All tests that touch LLM providers are mocked — no real API calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.rewards.builtins import LLMJudgeRewardFunc
from benchflow.rewards.llm import parse_verdict
from benchflow.rewards.protocol import RewardFunc
from benchflow.rewards.rubric_config import ScoringConfig

# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


class TestParseVerdict:
    def test_code_fenced_json(self) -> None:
        text = '```json\n{"verdict": "pass", "reasoning": "ok"}\n```'
        v = parse_verdict(text)
        assert v["verdict"] == "pass"

    def test_bare_json(self) -> None:
        text = 'Here is my verdict: {"verdict": "fail", "reasoning": "missing"}'
        v = parse_verdict(text)
        assert v["verdict"] == "fail"

    def test_nested_braces(self) -> None:
        text = '{"verdict": "pass", "details": {"a": 1}}'
        v = parse_verdict(text)
        assert v["verdict"] == "pass"

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not parse"):
            parse_verdict("no json here at all")


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: legacy mode
# ---------------------------------------------------------------------------


class TestLLMJudgeLegacy:
    def test_reads_score_file(self, tmp_path: Path) -> None:
        (tmp_path / "llm_judge_score.txt").write_text("0.85\n")
        func = LLMJudgeRewardFunc(prompt="rate it")
        score = asyncio.run(func.score(tmp_path))
        assert score == pytest.approx(0.85)

    def test_missing_score_file(self, tmp_path: Path) -> None:
        func = LLMJudgeRewardFunc(prompt="rate it")
        score = asyncio.run(func.score(tmp_path))
        assert score == 0.0

    def test_invalid_score_file(self, tmp_path: Path) -> None:
        (tmp_path / "llm_judge_score.txt").write_text("not-a-number\n")
        func = LLMJudgeRewardFunc(prompt="rate it")
        score = asyncio.run(func.score(tmp_path))
        assert score == 0.0


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_satisfies_reward_func_protocol(self) -> None:
        assert isinstance(LLMJudgeRewardFunc(prompt="test"), RewardFunc)

    def test_satisfies_with_rubric_path(self, tmp_path: Path) -> None:
        func = LLMJudgeRewardFunc(rubric_path=tmp_path / "rubric.toml")
        assert isinstance(func, RewardFunc)


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: rubric mode (mocked LLM)
# ---------------------------------------------------------------------------

_MOCK_PASS_RESPONSE = '```json\n{"verdict": "pass", "reasoning": "good"}\n```'
_MOCK_FAIL_RESPONSE = '```json\n{"verdict": "fail", "reasoning": "bad"}\n```'


def _make_rubric_toml(tmp_path: Path, content: str) -> Path:
    rubric_file = tmp_path / "rubric.toml"
    rubric_file.write_text(content)
    return rubric_file


class TestLLMJudgeRubricMode:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_single_binary_criterion_pass(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = _MOCK_PASS_RESPONSE
        (tmp_path / "answer.txt").write_text("hello world")

        rubric_path = _make_rubric_toml(
            tmp_path,
            """\
[judge]
model = "claude-sonnet-4-6"

[[criterion]]
description = "Is the answer correct?"
type = "binary"
""",
        )

        func = LLMJudgeRewardFunc(rubric_path=rubric_path)
        score = asyncio.run(func.score(tmp_path))

        assert score == pytest.approx(1.0)
        assert len(func.events) == 1
        assert func.events[0].type == "dense"
        assert func.events[0].reward == 1.0
        mock_judge.assert_called_once()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_single_binary_criterion_fail(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = _MOCK_FAIL_RESPONSE
        (tmp_path / "answer.txt").write_text("wrong answer")

        rubric_path = _make_rubric_toml(
            tmp_path,
            """\
[[criterion]]
description = "Is the answer correct?"
""",
        )

        func = LLMJudgeRewardFunc(rubric_path=rubric_path)
        score = asyncio.run(func.score(tmp_path))

        assert score == pytest.approx(0.0)
        assert func.events[0].reward == 0.0

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_multiple_criteria_weighted_mean(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.side_effect = [_MOCK_PASS_RESPONSE, _MOCK_FAIL_RESPONSE]
        (tmp_path / "output.txt").write_text("some output")

        rubric_path = _make_rubric_toml(
            tmp_path,
            """\
[[criterion]]
description = "Criterion A"
weight = 0.7

[[criterion]]
description = "Criterion B"
weight = 0.3
""",
        )

        func = LLMJudgeRewardFunc(rubric_path=rubric_path)
        score = asyncio.run(func.score(tmp_path))

        # weighted_mean: (1.0 * 0.7 + 0.0 * 0.3) / (0.7 + 0.3)
        assert score == pytest.approx(0.7)
        assert len(func.events) == 2

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_all_pass_aggregation(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.side_effect = [_MOCK_PASS_RESPONSE, _MOCK_FAIL_RESPONSE]
        (tmp_path / "output.txt").write_text("output")

        rubric_path = _make_rubric_toml(
            tmp_path,
            """\
[scoring]
aggregation = "all_pass"

[[criterion]]
description = "A"

[[criterion]]
description = "B"
""",
        )

        func = LLMJudgeRewardFunc(rubric_path=rubric_path)
        score = asyncio.run(func.score(tmp_path))
        assert score == 0.0  # Not all passed

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_any_pass_aggregation(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.side_effect = [_MOCK_PASS_RESPONSE, _MOCK_FAIL_RESPONSE]
        (tmp_path / "output.txt").write_text("output")

        rubric_path = _make_rubric_toml(
            tmp_path,
            """\
[scoring]
aggregation = "any_pass"

[[criterion]]
description = "A"

[[criterion]]
description = "B"
""",
        )

        func = LLMJudgeRewardFunc(rubric_path=rubric_path)
        score = asyncio.run(func.score(tmp_path))
        assert score == 1.0  # At least one passed

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_likert_criterion(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = '{"score": 4, "reasoning": "pretty good"}'
        (tmp_path / "output.txt").write_text("output")

        rubric_path = _make_rubric_toml(
            tmp_path,
            """\
[[criterion]]
description = "How good?"
type = "likert"
points = 5
""",
        )

        func = LLMJudgeRewardFunc(rubric_path=rubric_path)
        score = asyncio.run(func.score(tmp_path))
        # likert: (4 - 1) / (5 - 1) = 0.75
        assert score == pytest.approx(0.75)

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_numeric_criterion(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = '{"score": 70, "reasoning": "decent"}'
        (tmp_path / "output.txt").write_text("output")

        rubric_path = _make_rubric_toml(
            tmp_path,
            """\
[[criterion]]
description = "Rate coverage"
type = "numeric"
min = 0
max = 100
""",
        )

        func = LLMJudgeRewardFunc(rubric_path=rubric_path)
        score = asyncio.run(func.score(tmp_path))
        assert score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: inline criteria
# ---------------------------------------------------------------------------


class TestLLMJudgeInlineCriteria:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_inline_criteria_list(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = _MOCK_PASS_RESPONSE
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[
                {"description": "Is it correct?", "type": "binary"},
            ],
        )
        score = asyncio.run(func.score(tmp_path))
        assert score == pytest.approx(1.0)

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_inline_with_harvey_lab_keys(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """Inline criteria can use Harvey LAB style keys (id, match_criteria)."""
        mock_judge.return_value = _MOCK_PASS_RESPONSE
        (tmp_path / "analysis.txt").write_text("detailed analysis")

        func = LLMJudgeRewardFunc(
            criteria=[
                {
                    "id": "criterion-1",
                    "match_criteria": "Agent identifies the price",
                    "files": ["analysis.txt"],
                },
            ],
        )
        score = asyncio.run(func.score(tmp_path))
        assert score == pytest.approx(1.0)
        assert func.events[0].source == "criterion:criterion-1"


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: auto-discovery
# ---------------------------------------------------------------------------


class TestLLMJudgeAutoDiscovery:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_auto_discovers_rubric_toml(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = _MOCK_PASS_RESPONSE
        (tmp_path / "output.txt").write_text("answer")
        _make_rubric_toml(
            tmp_path,
            """\
[[criterion]]
description = "Works?"
""",
        )

        func = LLMJudgeRewardFunc()
        score = asyncio.run(func.score(tmp_path))
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: error handling
# ---------------------------------------------------------------------------


class TestLLMJudgeErrors:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_judge_error_returns_zero(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.side_effect = RuntimeError("API error")
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[{"description": "test"}],
        )
        score = asyncio.run(func.score(tmp_path))
        assert score == 0.0
        assert len(func.events) == 1
        assert func.events[0].reward == 0.0


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: evaluation details output
# ---------------------------------------------------------------------------


class TestEvaluationDetails:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_writes_evaluation_details(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.side_effect = [_MOCK_PASS_RESPONSE, _MOCK_FAIL_RESPONSE]
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[
                {"description": "A", "id": "a"},
                {"description": "B", "id": "b"},
            ],
        )
        asyncio.run(func.score(tmp_path))

        details_path = tmp_path / "evaluation_details.json"
        assert details_path.exists()
        details = json.loads(details_path.read_text())
        assert details["n_total"] == 2
        assert details["n_passed"] == 1
        assert len(details["results"]) == 2


# ---------------------------------------------------------------------------
# Dense reward events
# ---------------------------------------------------------------------------


class TestDenseRewardEvents:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_per_criterion_events(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.side_effect = [
            _MOCK_PASS_RESPONSE,
            _MOCK_FAIL_RESPONSE,
            _MOCK_PASS_RESPONSE,
        ]
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[
                {"description": "A", "id": "a"},
                {"description": "B", "id": "b"},
                {"description": "C", "id": "c"},
            ],
        )
        asyncio.run(func.score(tmp_path))

        events = func.events
        assert len(events) == 3
        assert all(e.type == "dense" for e in events)
        assert events[0].step == 0
        assert events[1].step == 1
        assert events[2].step == 2
        assert events[0].reward == 1.0
        assert events[1].reward == 0.0
        assert events[2].reward == 1.0
        assert events[0].source == "criterion:a"

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    def test_events_cleared_between_calls(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = _MOCK_PASS_RESPONSE
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[{"description": "A"}],
        )

        asyncio.run(func.score(tmp_path))
        assert len(func.events) == 1

        asyncio.run(func.score(tmp_path))
        assert len(func.events) == 1  # Not accumulated


# ---------------------------------------------------------------------------
# Aggregation helpers (unit tests)
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_weighted_mean(self) -> None:
        results = [
            {"score": 1.0, "weight": 2.0},
            {"score": 0.0, "weight": 1.0},
        ]
        s = ScoringConfig(aggregation="weighted_mean")
        assert LLMJudgeRewardFunc._aggregate(results, s) == pytest.approx(
            2.0 / 3.0
        )

    def test_all_pass_true(self) -> None:
        results = [
            {"score": 1.0, "weight": 1.0},
            {"score": 0.8, "weight": 1.0},
        ]
        s = ScoringConfig(aggregation="all_pass")
        assert LLMJudgeRewardFunc._aggregate(results, s) == 1.0

    def test_all_pass_false(self) -> None:
        results = [
            {"score": 1.0, "weight": 1.0},
            {"score": 0.3, "weight": 1.0},
        ]
        s = ScoringConfig(aggregation="all_pass")
        assert LLMJudgeRewardFunc._aggregate(results, s) == 0.0

    def test_any_pass(self) -> None:
        results = [
            {"score": 0.0, "weight": 1.0},
            {"score": 0.8, "weight": 1.0},
        ]
        s = ScoringConfig(aggregation="any_pass")
        assert LLMJudgeRewardFunc._aggregate(results, s) == 1.0

    def test_threshold_pass(self) -> None:
        results = [
            {"score": 0.8, "weight": 1.0},
            {"score": 0.7, "weight": 1.0},
        ]
        s = ScoringConfig(aggregation="threshold", threshold=0.7)
        assert LLMJudgeRewardFunc._aggregate(results, s) == 1.0

    def test_threshold_fail(self) -> None:
        results = [
            {"score": 0.5, "weight": 1.0},
            {"score": 0.3, "weight": 1.0},
        ]
        s = ScoringConfig(aggregation="threshold", threshold=0.7)
        assert LLMJudgeRewardFunc._aggregate(results, s) == 0.0

    def test_empty_results(self) -> None:
        s = ScoringConfig()
        assert LLMJudgeRewardFunc._aggregate([], s) == 0.0
