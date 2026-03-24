"""Tests for job counting logic and result aggregation."""

import json
from pathlib import Path

import pytest

from benchflow.job import JobConfig, JobResult, RetryConfig


class TestRetryConfig:
    def test_should_retry_install_failure(self):
        cfg = RetryConfig()
        assert cfg.should_retry("Agent install failed (rc=1): ...")

    def test_should_retry_pipe_error(self):
        cfg = RetryConfig()
        assert cfg.should_retry("Process closed stdout (rc=None)")

    def test_should_retry_acp_error(self):
        cfg = RetryConfig()
        assert cfg.should_retry("ACP error -32000: Authentication required")

    def test_should_not_retry_timeout(self):
        cfg = RetryConfig()
        assert not cfg.should_retry("Agent timed out after 900.0s")

    def test_should_not_retry_none(self):
        cfg = RetryConfig()
        assert not cfg.should_retry(None)

    def test_disable_install_retry(self):
        cfg = RetryConfig(retry_on_install=False)
        assert not cfg.should_retry("Agent install failed (rc=1): ...")

    def test_zero_retries(self):
        cfg = RetryConfig(max_retries=0)
        # should_retry still returns True for retryable errors,
        # but max_retries=0 means the job loop won't retry
        assert cfg.should_retry("Agent install failed (rc=1): ...")


class TestJobResult:
    def test_score(self):
        r = JobResult(job_name="test", config=JobConfig(), total=10, passed=5)
        assert r.score == 0.5

    def test_score_zero_total(self):
        r = JobResult(job_name="test", config=JobConfig(), total=0)
        assert r.score == 0.0

    def test_score_excl_errors(self):
        r = JobResult(
            job_name="test", config=JobConfig(),
            total=10, passed=5, failed=3, errored=2,
        )
        assert r.score_excl_errors == 5 / 8


class TestJobCounting:
    """Test the counting logic used in Job.run() — extracted as pure functions."""

    def _count(self, all_results: dict[str, dict]) -> dict:
        """Replicate the counting logic from job.py."""
        def _reward(r: dict) -> float | None:
            rewards = r.get("rewards")
            return rewards.get("reward") if rewards else None

        return {
            "passed": sum(1 for r in all_results.values() if _reward(r) == 1.0),
            "failed": sum(1 for r in all_results.values()
                         if _reward(r) is not None and _reward(r) != 1.0),
            "errored": sum(1 for r in all_results.values()
                          if r.get("error") and r.get("rewards") is None),
        }

    def test_basic_counting(self):
        results = {
            "a": {"rewards": {"reward": 1.0}, "error": None},
            "b": {"rewards": {"reward": 0.0}, "error": None},
            "c": {"rewards": None, "error": "timeout"},
        }
        counts = self._count(results)
        assert counts["passed"] == 1
        assert counts["failed"] == 1
        assert counts["errored"] == 1

    def test_all_passed(self):
        results = {
            "a": {"rewards": {"reward": 1.0}, "error": None},
            "b": {"rewards": {"reward": 1.0}, "error": None},
        }
        counts = self._count(results)
        assert counts["passed"] == 2
        assert counts["failed"] == 0
        assert counts["errored"] == 0

    def test_partial_reward_counts_as_failed(self):
        results = {
            "a": {"rewards": {"reward": 0.5}, "error": None},
        }
        counts = self._count(results)
        assert counts["passed"] == 0
        assert counts["failed"] == 1

    def test_error_with_rewards_is_not_errored(self):
        """If rewards exist but there's also an error string, it's not errored."""
        results = {
            "a": {"rewards": {"reward": 0.0}, "error": "some warning"},
        }
        counts = self._count(results)
        assert counts["errored"] == 0
        assert counts["failed"] == 1

    def test_empty_results(self):
        counts = self._count({})
        assert counts["passed"] == 0
        assert counts["failed"] == 0
        assert counts["errored"] == 0
