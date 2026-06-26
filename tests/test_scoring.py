"""Tests for benchflow._utils.scoring — pure scoring/classification helpers."""

import pytest

from benchflow._utils.scoring import (
    classify_error,
    classify_result_outcome,
    count_result_outcomes,
    extract_reward,
    pass_rate,
    pass_rate_excl_errors,
)


class TestExtractReward:
    """extract_reward(result) -> float | None"""

    def test_normal_reward(self):
        assert extract_reward({"rewards": {"reward": 1.0}}) == 1.0

    def test_zero_reward(self):
        assert extract_reward({"rewards": {"reward": 0.0}}) == 0.0

    def test_partial_reward(self):
        assert extract_reward({"rewards": {"reward": 0.5}}) == 0.5

    def test_no_rewards_key(self):
        assert extract_reward({"error": "timeout"}) is None

    def test_rewards_is_none(self):
        assert extract_reward({"rewards": None}) is None

    def test_empty_dict(self):
        assert extract_reward({}) is None

    def test_missing_reward_in_rewards(self):
        assert extract_reward({"rewards": {}}) is None

    def test_rewards_is_list(self):
        assert extract_reward({"rewards": [1.0]}) is None


class TestClassifyError:
    """classify_error(error) -> str | None"""

    def test_none(self):
        assert classify_error(None) is None

    def test_empty_string(self):
        assert classify_error("") is None

    def test_install_failed(self):
        assert (
            classify_error("Agent claude-agent-acp install failed (rc=1)")
            == "install_failure"
        )

    def test_pipe_closed(self):
        assert classify_error("Agent closed stdout") == "pipe_closed"

    def test_acp_error(self):
        assert classify_error("ACP error: connection refused") == "acp_error"

    def test_timeout(self):
        assert classify_error("Task timed out after 300s") == "timeout"

    def test_wall_clock_budget_timeout(self):
        assert classify_error("Agent prompt exceeded wall-clock budget 5s") == "timeout"

    def test_idle_timeout(self):
        assert (
            classify_error(
                "Agent idle for 600s with no new tool call, message, or thought"
            )
            == "idle_timeout"
        )

    def test_infra_failure(self):
        assert (
            classify_error("Sandbox not found. Please retry later.") == "infra_failure"
        )

    def test_other(self):
        assert classify_error("something unexpected") == "other"

    def test_provider_auth_gemini_403(self):
        """Guards the fix from PR #564 for issue #546: Gemini PERMISSION_DENIED
        should be classified as provider_auth, not acp_error."""
        assert (
            classify_error(
                "ACP error 403: PERMISSION_DENIED: Your API key was reported as leaked."
            )
            == "provider_auth"
        )

    def test_provider_auth_claude_401(self):
        """Guards the fix from PR #564 for issue #546: Claude 401 auth failure
        should be classified as provider_auth."""
        assert (
            classify_error(
                "ACP error -32603: Internal error: Failed to authenticate. "
                "API Error: 401 Invalid bearer token"
            )
            == "provider_auth"
        )

    def test_provider_auth_invalid_api_key(self):
        """Guards the fix from PR #564 for issue #546: invalid-API-key errors
        classify as provider_auth, not acp_error."""
        assert classify_error("ACP error -32001: Invalid API key") == "provider_auth"

    def test_provider_auth_lowercase_and_status_forms(self):
        """Guards PR #564: reviewer-reported auth shapes the original top-level
        string match missed must classify as provider_auth, not acp_error."""
        for err in (
            "ACP error 401: unauthorized",
            "ACP error -32001: invalid api key",
            "ACP error -32603: failed to authenticate",
            "ACP error -32603: Internal error | provider auth failed (HTTP 401)",
        ):
            assert classify_error(err) == "provider_auth", err

    def test_provider_rate_limit_marker(self):
        """Guards PR #653: Bedrock daily caps surface as provider_rate_limit."""
        assert (
            classify_error(
                "ACP error -32603: Internal error | provider rate limited (HTTP 429)"
            )
            == "provider_rate_limit"
        )

    def test_provider_unavailable_marker_is_infra(self):
        """Provider 503s are transient infra, not generic ACP errors."""
        assert (
            classify_error(
                "ACP error -32603: Internal error | provider unavailable (HTTP 503)"
            )
            == "infra_failure"
        )

    def test_provider_rejected_marker_is_permanent(self):
        """Guards #830: a context-window 400 surfaced as a raised ACP error
        classifies as provider_rejected (permanent), not generic acp_error."""
        assert (
            classify_error(
                "ACP error -32603: Internal error | provider rejected request (HTTP 400)"
            )
            == "provider_rejected"
        )

    def test_generic_acp_internal_error_still_retryable(self):
        """Guards PR #564: a bare ACP internal error with no auth signal stays
        acp_error — only a real surfaced 401/403 should flip it to provider_auth."""
        assert classify_error("ACP error -32603: Internal error") == "acp_error"

    def test_provider_auth_rejected_as_invalid(self):
        """The message from _classify_acp_error when subscription auth exists."""
        assert (
            classify_error(
                "GEMINI_API_KEY was rejected as invalid. "
                "Subscription auth credentials exist — unset the env var "
                "to use them: env -u GEMINI_API_KEY <command>"
            )
            == "provider_auth"
        )

    def test_acp_error_non_auth_still_retryable(self):
        """Generic ACP errors without auth markers should stay acp_error."""
        assert classify_error("ACP error -32000: connection refused") == "acp_error"

    def test_install_broad_match_not_used(self):
        """'install' alone should NOT match — only 'install failed'."""
        assert classify_error("installing dependencies") == "other"


class TestClassifyResultOutcome:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            ({"rewards": {"reward": 1.0}}, "passed"),
            ({"rewards": {"reward": 0.0}}, "failed"),
            ({"rewards": {"reward": 0.5}}, "failed"),
            (
                {
                    "rewards": None,
                    "error": "Agent prompt exceeded wall-clock budget 5s",
                    "verifier_error": "verifier crashed: No reward file found",
                },
                "verifier_errored",
            ),
            (
                {
                    "rewards": {"reward": 0.0},
                    "error": None,
                    "verifier_error": "verifier crashed: stale reward rejected",
                },
                "verifier_errored",
            ),
            (
                {"rewards": None, "error": "Agent timed out", "verifier_error": None},
                "errored",
            ),
            ({"rewards": None, "error": None, "verifier_error": None}, "unscored"),
        ],
    )
    def test_outcome_buckets(self, result, expected):
        assert classify_result_outcome(result) == expected

    def test_count_result_outcomes(self):
        assert count_result_outcomes(
            [
                {"rewards": {"reward": 1.0}},
                {"rewards": {"reward": 0.0}},
                {"rewards": None, "error": "timed out"},
                {
                    "rewards": None,
                    "error": "timed out",
                    "verifier_error": "verifier crashed",
                },
            ]
        ) == {
            "passed": 1,
            "failed": 1,
            "errored": 1,
            "verifier_errored": 1,
            "unscored": 0,
        }


class TestPassRate:
    """pass_rate(*, passed, total) -> float"""

    @pytest.mark.parametrize(
        ("passed", "total", "expected"),
        [
            pytest.param(5, 10, 0.5, id="50pct"),
            pytest.param(10, 10, 1.0, id="100pct"),
            pytest.param(0, 10, 0.0, id="0pct"),
            pytest.param(0, 0, 0.0, id="empty"),
        ],
    )
    def test_pass_rate(self, passed, total, expected):
        assert pass_rate(passed=passed, total=total) == expected


class TestPassRateExclErrors:
    """pass_rate_excl_errors(*, passed, failed) -> float"""

    @pytest.mark.parametrize(
        ("passed", "failed", "expected"),
        [
            pytest.param(5, 3, 5 / 8, id="normal"),
            pytest.param(5, 0, 1.0, id="100pct"),
            pytest.param(0, 5, 0.0, id="0pct"),
            pytest.param(0, 0, 0.0, id="empty"),
        ],
    )
    def test_pass_rate_excl_errors(self, passed, failed, expected):
        assert pass_rate_excl_errors(passed=passed, failed=failed) == expected
