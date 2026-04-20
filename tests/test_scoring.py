"""Tests for benchflow._scoring — pure scoring/classification helpers."""

import pytest

from benchflow._scoring import (
    classify_error,
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

    def test_other(self):
        assert classify_error("something unexpected") == "other"

    def test_install_broad_match_not_used(self):
        """'install' alone should NOT match — only 'install failed'."""
        assert classify_error("installing dependencies") == "other"


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
