"""Tests for the first-class LLM-as-judge verifier (ENG-55).

All tests that touch LLM providers are mocked — no real API calls.
"""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from benchflow.rewards.builtins import LLMJudgeRewardFunc
from benchflow.rewards.llm import (
    JudgeEnvironmentError,
    _call_anthropic,
    _call_google,
    call_judge,
    parse_verdict,
)
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
# call_judge: provider routing and fallback
# ---------------------------------------------------------------------------


class TestCallJudgeProviderFallback:
    async def test_api_failure_surfaces_original_error(self) -> None:
        """A real API failure on the matching provider is raised as-is.

        The cross-provider fallback must NOT advance to a provider whose
        API cannot serve this model name — otherwise the surfaced error is
        a misleading model-not-found error from the wrong provider instead
        of the genuine failure.
        """
        original = RuntimeError("anthropic: invalid x-api-key")
        anthropic_mock = AsyncMock(side_effect=original)
        openai_mock = AsyncMock(return_value="should not be reached")
        google_mock = AsyncMock(return_value="should not be reached")

        with (
            patch("benchflow.rewards.llm._call_anthropic", anthropic_mock),
            patch("benchflow.rewards.llm._call_openai", openai_mock),
            patch("benchflow.rewards.llm._call_google", google_mock),
            pytest.raises(RuntimeError, match="invalid x-api-key") as exc_info,
        ):
            await call_judge("claude-haiku-4-5", "prompt", retries=2)

        # The exact original exception object propagates.
        assert exc_info.value is original
        # The matching provider was retried, but no other provider tried.
        assert anthropic_mock.await_count == 2
        openai_mock.assert_not_awaited()
        google_mock.assert_not_awaited()

    async def test_import_error_falls_through_to_next_provider(self) -> None:
        """A missing SDK (ImportError) still falls through to the next provider."""
        anthropic_mock = AsyncMock(side_effect=ImportError("no anthropic SDK"))
        openai_mock = AsyncMock(return_value="ok from openai")

        with (
            patch("benchflow.rewards.llm._call_anthropic", anthropic_mock),
            patch("benchflow.rewards.llm._call_openai", openai_mock),
        ):
            result = await call_judge("claude-haiku-4-5", "prompt", retries=2)

        assert result == "ok from openai"
        # ImportError is not retried.
        assert anthropic_mock.await_count == 1

    async def test_all_sdks_missing_raises_judge_environment_error(self) -> None:
        """When *every* provider SDK is missing, call_judge raises a
        JudgeEnvironmentError — a distinct, identifiable environment failure,
        not a generic RuntimeError that callers might mistake for a verdict."""
        missing = AsyncMock(side_effect=ImportError("no SDK"))

        with (
            patch("benchflow.rewards.llm._call_anthropic", missing),
            patch("benchflow.rewards.llm._call_openai", missing),
            patch("benchflow.rewards.llm._call_google", missing),
            pytest.raises(JudgeEnvironmentError, match="judge extra"),
        ):
            await call_judge("claude-haiku-4-5", "prompt", retries=2)

    def test_judge_environment_error_is_a_runtime_error(self) -> None:
        """JudgeEnvironmentError stays a RuntimeError subclass so existing
        ``except RuntimeError`` handlers keep working."""
        assert issubclass(JudgeEnvironmentError, RuntimeError)


# ---------------------------------------------------------------------------
# _call_anthropic: content block handling
# ---------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeNonTextBlock:
    """A block with no ``.text`` attribute (e.g. a tool-use block)."""


class _FakeResponse:
    def __init__(self, content: list) -> None:
        self.content = content


class TestCallAnthropicContent:
    def _patch_sdk(self, response: _FakeResponse) -> AbstractContextManager[None]:
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=response)
        fake_anthropic = type(
            "FakeAnthropic",
            (),
            {"AsyncAnthropic": staticmethod(lambda: client)},
        )
        return patch.dict("sys.modules", {"anthropic": fake_anthropic})

    async def test_empty_content_returns_empty_string(self) -> None:
        with self._patch_sdk(_FakeResponse([])):
            result = await _call_anthropic("claude-haiku-4-5", "prompt", 100)
        assert result == ""

    async def test_non_text_first_block_returns_empty_string(self) -> None:
        with self._patch_sdk(_FakeResponse([_FakeNonTextBlock()])):
            result = await _call_anthropic("claude-haiku-4-5", "prompt", 100)
        assert result == ""

    async def test_non_text_block_before_text_block(self) -> None:
        response = _FakeResponse([_FakeNonTextBlock(), _FakeTextBlock("the verdict")])
        with self._patch_sdk(response):
            result = await _call_anthropic("claude-haiku-4-5", "prompt", 100)
        assert result == "the verdict"

    async def test_text_block_returns_text(self) -> None:
        with self._patch_sdk(_FakeResponse([_FakeTextBlock("hello")])):
            result = await _call_anthropic("claude-haiku-4-5", "prompt", 100)
        assert result == "hello"


# ---------------------------------------------------------------------------
# _call_google: text-part handling
# ---------------------------------------------------------------------------


class _FakeGoogleResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class TestCallGoogleContent:
    def _patch_sdk(self, response: _FakeGoogleResponse) -> AbstractContextManager[None]:
        client = AsyncMock()
        client.aio.models.generate_content = AsyncMock(return_value=response)
        fake_genai = type(
            "FakeGenai",
            (),
            {"Client": staticmethod(lambda api_key=None: client)},
        )
        fake_google = type("FakeGoogle", (), {"genai": fake_genai})
        return patch.dict(
            "sys.modules",
            {"google": fake_google, "google.genai": fake_genai},
        )

    async def test_none_text_returns_empty_string(self) -> None:
        """``response.text`` is None (e.g. safety-filtered) -> "" not None."""
        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            self._patch_sdk(_FakeGoogleResponse(None)),
        ):
            result = await _call_google("gemini-2.0-flash", "prompt")
        assert result == ""

    async def test_text_returns_text(self) -> None:
        with (
            patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}),
            self._patch_sdk(_FakeGoogleResponse("the verdict")),
        ):
            result = await _call_google("gemini-2.0-flash", "prompt")
        assert result == "the verdict"


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: legacy mode
# ---------------------------------------------------------------------------


class TestLLMJudgeLegacy:
    async def test_reads_score_file(self, tmp_path: Path) -> None:
        (tmp_path / "llm_judge_score.txt").write_text("0.85\n")
        func = LLMJudgeRewardFunc(prompt="rate it")
        score = await func.score(tmp_path)
        assert score == pytest.approx(0.85)

    async def test_missing_score_file(self, tmp_path: Path) -> None:
        func = LLMJudgeRewardFunc(prompt="rate it")
        score = await func.score(tmp_path)
        assert score == 0.0

    async def test_invalid_score_file(self, tmp_path: Path) -> None:
        (tmp_path / "llm_judge_score.txt").write_text("not-a-number\n")
        func = LLMJudgeRewardFunc(prompt="rate it")
        score = await func.score(tmp_path)
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
    async def test_single_binary_criterion_pass(
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
        score = await func.score(tmp_path)

        assert score == pytest.approx(1.0)
        assert len(func.events) == 1
        assert func.events[0].type == "dense"
        assert func.events[0].reward == 1.0
        mock_judge.assert_called_once()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_single_binary_criterion_fail(
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
        score = await func.score(tmp_path)

        assert score == pytest.approx(0.0)
        assert func.events[0].reward == 0.0

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_multiple_criteria_weighted_mean(
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
        score = await func.score(tmp_path)

        # weighted_mean: (1.0 * 0.7 + 0.0 * 0.3) / (0.7 + 0.3)
        assert score == pytest.approx(0.7)
        assert len(func.events) == 2

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_all_pass_aggregation(
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
        score = await func.score(tmp_path)
        assert score == 0.0  # Not all passed

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_any_pass_aggregation(
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
        score = await func.score(tmp_path)
        assert score == 1.0  # At least one passed

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_likert_criterion(
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
        score = await func.score(tmp_path)
        # likert: (4 - 1) / (5 - 1) = 0.75
        assert score == pytest.approx(0.75)

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_numeric_criterion(
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
        score = await func.score(tmp_path)
        assert score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: inline criteria
# ---------------------------------------------------------------------------


class TestLLMJudgeInlineCriteria:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_inline_criteria_list(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = _MOCK_PASS_RESPONSE
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[
                {"description": "Is it correct?", "type": "binary"},
            ],
        )
        score = await func.score(tmp_path)
        assert score == pytest.approx(1.0)

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_inline_with_harvey_lab_keys(
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
        score = await func.score(tmp_path)
        assert score == pytest.approx(1.0)
        assert func.events[0].source == "criterion:criterion-1"


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: auto-discovery
# ---------------------------------------------------------------------------


class TestLLMJudgeAutoDiscovery:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_auto_discovers_rubric_toml(
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
        score = await func.score(tmp_path)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: error handling
# ---------------------------------------------------------------------------


class TestLLMJudgeErrors:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_judge_error_returns_zero(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """A real API failure (bad key, model not found) is a genuine judge
        outcome — it degrades the criterion to 0.0, not an environment error."""
        mock_judge.side_effect = RuntimeError("API error")
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[{"description": "test"}],
        )
        score = await func.score(tmp_path)
        assert score == 0.0
        assert len(func.events) == 1
        assert func.events[0].reward == 0.0

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_missing_sdk_propagates_not_scored_zero(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """A missing provider SDK (JudgeEnvironmentError) must propagate out of
        score() — the judge never ran, so it is an environment failure, not a
        verdict of 0.0. Scoring it 0.0 would be indistinguishable from a real
        fail and would silently corrupt benchmark results."""
        mock_judge.side_effect = JudgeEnvironmentError(
            "No LLM provider SDK is installed"
        )
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(criteria=[{"description": "test"}])

        with pytest.raises(JudgeEnvironmentError):
            await func.score(tmp_path)


# ---------------------------------------------------------------------------
# LLMJudgeRewardFunc: evaluation details output
# ---------------------------------------------------------------------------


class TestEvaluationDetails:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    async def test_writes_evaluation_details(
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
        await func.score(tmp_path)

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
    async def test_per_criterion_events(
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
        await func.score(tmp_path)

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
    async def test_events_cleared_between_calls(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = _MOCK_PASS_RESPONSE
        (tmp_path / "output.txt").write_text("answer")

        func = LLMJudgeRewardFunc(
            criteria=[{"description": "A"}],
        )

        await func.score(tmp_path)
        assert len(func.events) == 1

        await func.score(tmp_path)
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
        assert LLMJudgeRewardFunc._aggregate(results, s) == pytest.approx(2.0 / 3.0)

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
