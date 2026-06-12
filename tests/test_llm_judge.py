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
    _strip_provider_prefix,
    call_judge,
    parse_verdict,
)
from benchflow.rewards.protocol import RewardFunc
from benchflow.rewards.rubric_config import ScoringConfig

# Verdict parsing


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


# Provider-prefix stripping


class TestStripProviderPrefix:
    def test_anthropic_prefix_stripped(self) -> None:
        assert (
            _strip_provider_prefix("anthropic/claude-haiku-4-5") == "claude-haiku-4-5"
        )

    def test_openai_prefix_stripped(self) -> None:
        assert _strip_provider_prefix("openai/gpt-4o") == "gpt-4o"

    def test_google_prefix_stripped(self) -> None:
        assert _strip_provider_prefix("google/gemini-2.0-flash") == "gemini-2.0-flash"

    def test_gemini_prefix_stripped_to_bare_google_form(self) -> None:
        """Dogfood bug (1): ``gemini/<model>`` validates clean (startswith
        ``gemini`` => supported) but, before this fix, was passed verbatim to
        google-genai and 404'd because the SDK does not accept the slashed
        name. The prefix must resolve to the same bare form ``google/`` does."""
        assert (
            _strip_provider_prefix("gemini/gemini-3.1-flash-lite")
            == "gemini-3.1-flash-lite"
        )
        # And it lands on the same bare model the google/ spelling produces.
        assert _strip_provider_prefix(
            "gemini/gemini-3.1-flash-lite"
        ) == _strip_provider_prefix("google/gemini-3.1-flash-lite")

    def test_unprefixed_gemini_model_unchanged(self) -> None:
        """A bare ``gemini-...`` (no slash) is already the working form."""
        assert (
            _strip_provider_prefix("gemini-3.1-flash-lite") == "gemini-3.1-flash-lite"
        )

    def test_unknown_prefix_left_intact(self) -> None:
        assert (
            _strip_provider_prefix("deepseek/deepseek-chat") == "deepseek/deepseek-chat"
        )


# call_judge: provider routing and fallback


class TestCallJudgeProviderFallback:
    """Guards the fix from PR #319 (rewards-code audit) for cross-provider
    judge fallback."""

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

    async def test_unknown_prefix_import_error_falls_through_to_next_provider(
        self,
    ) -> None:
        """For an *unknown-prefix* model, a missing SDK (ImportError) falls
        through to the next provider — any provider might serve it."""
        anthropic_mock = AsyncMock(side_effect=ImportError("no anthropic SDK"))
        openai_mock = AsyncMock(return_value="ok from openai")

        with (
            patch("benchflow.rewards.llm._call_anthropic", anthropic_mock),
            patch("benchflow.rewards.llm._call_openai", openai_mock),
        ):
            result = await call_judge("deepseek-chat", "prompt", retries=2)

        assert result == "ok from openai"
        # ImportError is not retried.
        assert anthropic_mock.await_count == 1

    async def test_matched_provider_missing_sdk_raises_clear_error(self) -> None:
        """Dogfood bug (2): when the model name confidently selects a provider
        whose judge SDK is not installed, ``call_judge`` raises a CLEAR,
        provider-named error pointing at ``benchflow[judge]`` — it does NOT
        fall through to another provider, which would surface a misleading
        "Missing OPENAI_API_KEY" instead.
        """
        anthropic_mock = AsyncMock(side_effect=ImportError("no anthropic SDK"))
        # OpenAI would have produced the misleading missing-key error.
        openai_mock = AsyncMock(side_effect=RuntimeError("Missing OPENAI_API_KEY"))
        google_mock = AsyncMock(return_value="should not be reached")

        with (
            patch("benchflow.rewards.llm._call_anthropic", anthropic_mock),
            patch("benchflow.rewards.llm._call_openai", openai_mock),
            patch("benchflow.rewards.llm._call_google", google_mock),
            pytest.raises(JudgeEnvironmentError) as exc_info,
        ):
            await call_judge("claude-haiku-4-5", "prompt", retries=2)

        msg = str(exc_info.value)
        assert "anthropic" in msg  # names the actual provider
        assert "benchflow[judge]" in msg  # actionable install fix
        # ImportError is not retried, and no fallback provider is consulted.
        assert anthropic_mock.await_count == 1
        openai_mock.assert_not_awaited()
        google_mock.assert_not_awaited()

    async def test_gemini_provider_missing_sdk_raises_clear_error(self) -> None:
        """Dogfood bugs (1)+(2): a ``gemini/`` model whose google SDK is
        missing raises a clear google-named error rather than falling through
        to anthropic/openai."""
        google_mock = AsyncMock(side_effect=ImportError("no google-genai SDK"))
        anthropic_mock = AsyncMock(return_value="should not be reached")
        openai_mock = AsyncMock(return_value="should not be reached")

        with (
            patch("benchflow.rewards.llm._call_google", google_mock),
            patch("benchflow.rewards.llm._call_anthropic", anthropic_mock),
            patch("benchflow.rewards.llm._call_openai", openai_mock),
            pytest.raises(JudgeEnvironmentError, match="google") as exc_info,
        ):
            await call_judge("gemini/gemini-3.1-flash-lite", "prompt", retries=2)

        assert "benchflow[judge]" in str(exc_info.value)
        anthropic_mock.assert_not_awaited()
        openai_mock.assert_not_awaited()

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

    async def test_unknown_prefix_model_falls_through_after_api_failure(self) -> None:
        """PR #319 follow-up: an unknown-prefix model whose first provider
        fails at the API still reaches the remaining providers.

        ``call_judge`` enters the "try all" branch for an unknown prefix
        (``mistral-large``, ``deepseek-chat``, custom names). The pre-fix
        code did an unconditional ``raise`` after the first provider
        exhausted retries, so OpenAI/Google were never reached and the
        fallback was dead. The fix only raises immediately for a model
        confidently matched to a single provider.
        """
        anthropic_mock = AsyncMock(side_effect=RuntimeError("anthropic 404"))
        openai_mock = AsyncMock(return_value="ok from openai")
        google_mock = AsyncMock(return_value="should not be reached")

        with (
            patch("benchflow.rewards.llm._call_anthropic", anthropic_mock),
            patch("benchflow.rewards.llm._call_openai", openai_mock),
            patch("benchflow.rewards.llm._call_google", google_mock),
        ):
            result = await call_judge("mistral-large", "prompt", retries=2)

        # Anthropic failed and was retried, then OpenAI served the call.
        assert result == "ok from openai"
        assert anthropic_mock.await_count == 2
        assert openai_mock.await_count == 1
        google_mock.assert_not_awaited()

    async def test_unknown_prefix_all_providers_fail_raises_last_error(self) -> None:
        """PR #319 follow-up: when every provider fails at the API for an
        unknown-prefix model, the genuine API error surfaces — not a
        misleading missing-SDK ``JudgeEnvironmentError``."""
        anthropic_err = RuntimeError("anthropic refused")
        google_err = RuntimeError("google refused")

        with (
            patch(
                "benchflow.rewards.llm._call_anthropic",
                AsyncMock(side_effect=anthropic_err),
            ),
            patch(
                "benchflow.rewards.llm._call_openai",
                AsyncMock(side_effect=RuntimeError("openai refused")),
            ),
            patch(
                "benchflow.rewards.llm._call_google",
                AsyncMock(side_effect=google_err),
            ),
            pytest.raises(RuntimeError, match="google refused") as exc_info,
        ):
            await call_judge("deepseek-chat", "prompt", retries=1)

        # The last provider's real error is surfaced, not JudgeEnvironmentError.
        assert exc_info.value is google_err
        assert not isinstance(exc_info.value, JudgeEnvironmentError)

    async def test_known_prefix_still_raises_immediately(self) -> None:
        """PR #319 follow-up: the unknown-prefix fallback fix must not weaken
        known-prefix routing — a ``claude-`` model that fails at the API still
        raises at once without trying OpenAI/Google."""
        original = RuntimeError("anthropic: invalid x-api-key")
        openai_mock = AsyncMock(return_value="should not be reached")

        with (
            patch(
                "benchflow.rewards.llm._call_anthropic",
                AsyncMock(side_effect=original),
            ),
            patch("benchflow.rewards.llm._call_openai", openai_mock),
            pytest.raises(RuntimeError, match="invalid x-api-key"),
        ):
            await call_judge("claude-haiku-4-5", "prompt", retries=1)

        openai_mock.assert_not_awaited()


class TestCallJudgeEnvThreading:
    """PR #314 follow-up: judge credentials are threaded explicitly through
    ``call_judge`` instead of mutating the process-global ``os.environ`` (the
    pre-fix ``_scoped_env``), which is not concurrency-safe under
    ``asyncio.gather``."""

    async def test_env_passed_to_anthropic_client(self) -> None:
        """The ``env`` kwarg's ANTHROPIC_API_KEY reaches the Anthropic client
        as an explicit constructor argument."""
        seen: dict[str, object] = {}

        def fake_anthropic_ctor(*, api_key: str | None = None) -> AsyncMock:
            seen["api_key"] = api_key
            client = AsyncMock()
            client.messages.create = AsyncMock(
                return_value=_FakeResponse([_FakeTextBlock("ok")])
            )
            return client

        fake_anthropic = type(
            "FakeAnthropic",
            (),
            {"AsyncAnthropic": staticmethod(fake_anthropic_ctor)},
        )
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            result = await call_judge(
                "claude-haiku-4-5",
                "prompt",
                retries=1,
                env={"ANTHROPIC_API_KEY": "key-from-verifier-env"},
            )

        assert result == "ok"
        assert seen["api_key"] == "key-from-verifier-env"

    async def test_concurrent_judge_calls_use_isolated_credentials(self) -> None:
        """PR #314 follow-up: two judge calls running concurrently via
        ``asyncio.gather`` each see *their own* credentials.

        With the pre-fix ``_scoped_env`` global mutation, one coroutine could
        restore/pop an env key while another was still mid-call, so the second
        judge saw missing/wrong credentials. Threading the key explicitly
        keeps the two calls fully isolated.
        """
        import asyncio

        observed: list[str | None] = []

        def fake_anthropic_ctor(*, api_key: str | None = None) -> AsyncMock:
            client = AsyncMock()

            async def create(**_kwargs: object) -> _FakeResponse:
                # Record the key, then yield so the other coroutine runs
                # interleaved — exactly the race _scoped_env could not survive.
                observed.append(api_key)
                await asyncio.sleep(0)
                return _FakeResponse([_FakeTextBlock("ok")])

            client.messages.create = create
            return client

        fake_anthropic = type(
            "FakeAnthropic",
            (),
            {"AsyncAnthropic": staticmethod(fake_anthropic_ctor)},
        )
        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            await asyncio.gather(
                call_judge(
                    "claude-haiku-4-5", "p", retries=1, env={"ANTHROPIC_API_KEY": "A"}
                ),
                call_judge(
                    "claude-haiku-4-5", "p", retries=1, env={"ANTHROPIC_API_KEY": "B"}
                ),
            )

        # Both distinct keys were seen — neither call clobbered the other's.
        assert set(observed) == {"A", "B"}


# _call_anthropic: content block handling


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeNonTextBlock:
    """A block with no ``.text`` attribute (e.g. a tool-use block)."""


class _FakeResponse:
    def __init__(self, content: list) -> None:
        self.content = content


class TestCallAnthropicContent:
    """Guards the fix from PR #319 (rewards-code audit) for `_call_anthropic`
    content-block extraction."""

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


# _call_google: text-part handling


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


# LLMJudgeRewardFunc: legacy mode


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


# LLMJudgeRewardFunc: protocol conformance


class TestProtocol:
    def test_satisfies_reward_func_protocol(self) -> None:
        assert isinstance(LLMJudgeRewardFunc(prompt="test"), RewardFunc)

    def test_satisfies_with_rubric_path(self, tmp_path: Path) -> None:
        func = LLMJudgeRewardFunc(rubric_path=tmp_path / "rubric.toml")
        assert isinstance(func, RewardFunc)


# LLMJudgeRewardFunc: rubric mode (mocked LLM)

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
        """Guards the fix from PR #319 (rewards-code audit) for likert scoring."""
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


# LLMJudgeRewardFunc: inline criteria


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


# LLMJudgeRewardFunc: auto-discovery


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


# LLMJudgeRewardFunc: error handling


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


# LLMJudgeRewardFunc: evaluation details output


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


# Dense reward events


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


# Aggregation helpers (unit tests)


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
