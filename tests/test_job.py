"""Tests for job counting logic and result aggregation."""

import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock

import pytest

from benchflow.evaluation import (
    Evaluation,
    EvaluationConfig,
    ResumeMismatchError,
    RetryConfig,
)
from benchflow.models import RunResult


class TestRetryConfig:
    def test_should_retry_install_failure(self):
        cfg = RetryConfig()
        assert cfg.should_retry("Agent install failed (rc=1): ...")

    def test_should_retry_pipe_error(self):
        cfg = RetryConfig()
        assert cfg.should_retry("Process closed stdout (rc=None)")

    def test_should_retry_structured_pipe_category(self):
        """Guards PR #561: retry policy can use diagnostic-owned categories."""
        cfg = RetryConfig()
        assert cfg.should_retry(
            "DaytonaPtyProcess: timeout waiting for agent start marker",
            category="pipe_closed",
        )

    def test_should_retry_acp_error(self):
        cfg = RetryConfig()
        assert cfg.should_retry("ACP error -32000: connection refused")

    def test_should_not_retry_provider_auth_gemini_403(self):
        """Guards the fix from PR #564 for issue #546: provider auth failures
        should not be retried — they waste Daytona sandbox quota."""
        cfg = RetryConfig()
        assert not cfg.should_retry(
            "ACP error 403: PERMISSION_DENIED: Your API key was reported as leaked."
        )

    def test_should_not_retry_provider_auth_claude_401(self):
        """Guards the fix from PR #564 for issue #546: Claude 401 auth failures
        must not be retried — they waste Daytona sandbox quota."""
        cfg = RetryConfig()
        assert not cfg.should_retry(
            "ACP error -32603: Internal error: Failed to authenticate. "
            "API Error: 401 Invalid bearer token"
        )

    def test_should_not_retry_sanitized_proxy_auth_marker(self):
        """Guards PR #564: the sanitized "provider auth failed (HTTP 401)"
        marker injected from the proxy trajectory must fail fast, even when the
        top-level ACP error is only a generic internal error."""
        cfg = RetryConfig()
        assert not cfg.should_retry(
            "ACP error -32603: Internal error | provider auth failed (HTTP 401)"
        )

    def test_should_not_retry_provider_rate_limit_sanitized_marker(self):
        """Guards PR #653: Bedrock daily caps must not burn retries."""
        cfg = RetryConfig()
        assert not cfg.should_retry(
            "ACP error -32603: Internal error | provider rate limited (HTTP 429)"
        )

    def test_should_retry_provider_unavailable_sanitized_marker(self):
        """Provider 503s are transient infra and remain retryable."""
        cfg = RetryConfig()
        assert cfg.should_retry(
            "ACP error -32603: Internal error | provider unavailable (HTTP 503)"
        )

    def test_should_not_retry_timeout(self):
        cfg = RetryConfig()
        assert not cfg.should_retry("Agent timed out after 900.0s")

    def test_should_retry_idle_timeout(self):
        """Guards infra retry parity for ACP idle watchdog failures."""
        cfg = RetryConfig()
        assert cfg.should_retry(
            "Agent idle for 600s with no new tool call, message, or thought"
        )

    def test_should_retry_sandbox_infra_failure(self):
        """Guards infra retry parity for transient sandbox/provider failures."""
        cfg = RetryConfig()
        assert cfg.should_retry(
            "Sandbox not found. Please start the environment first."
        )

    def test_should_retry_verifier_timeout(self):
        """Guards retry parity for verifier timeout infra failures."""
        cfg = RetryConfig()
        assert cfg.should_retry_verifier_error("verifier timed out after 60s")

    def test_should_retry_verifier_download_failure(self):
        """Guards retry parity for verifier transport/download failures."""
        cfg = RetryConfig()
        assert cfg.should_retry_verifier_error(
            "verifier crashed: Failed to download verifier directory from sandbox"
        )

    def test_should_not_retry_missing_reward_file(self):
        """Missing reward is a verifier contract failure, not transient infra."""
        cfg = RetryConfig()
        assert not cfg.should_retry_verifier_error(
            "verifier crashed: No reward file found"
        )

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


class TestJobCounting:
    """Test the counting logic used in Evaluation.run() — calls the real extract_reward."""

    def _count(self, all_results: dict[str, dict]) -> dict:
        """Same counting logic as job.py, using the shared extract_reward."""
        from benchflow._utils.scoring import extract_reward

        return {
            "passed": sum(1 for r in all_results.values() if extract_reward(r) == 1.0),
            "failed": sum(
                1
                for r in all_results.values()
                if extract_reward(r) is not None and extract_reward(r) != 1.0
            ),
            "errored": sum(
                1
                for r in all_results.values()
                if r.get("error") and r.get("rewards") is None
            ),
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

    def test_error_and_verifier_error_counted_once(self):
        """A result with BOTH error and verifier_error (rewards=None) must be
        classified into exactly one bucket so the count invariant holds.

        Exercises the real shared classifier: a result with an agent error is
        ``errored``, and ``verifier_errored`` only applies when there is a
        verifier error AND no agent error — ``errored`` takes precedence.
        """
        from benchflow._utils.scoring import classify_result_dict

        result = {
            "rewards": None,
            "error": "agent crashed",
            "verifier_error": "verifier also failed",
        }
        # The real classifier puts a both-errors result in exactly `errored`.
        assert classify_result_dict(result) == "errored"


class TestRunTaskLoop:
    """Tests for Evaluation._run_task — retry loop behavior."""

    @pytest.mark.asyncio
    async def test_exhausts_retries(self, job_factory):
        """SDK called 1 + max_retries times when all attempts fail with retryable error."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
        fail_result = RunResult(
            task_name="task-0", error="Agent install failed (rc=1): boom"
        )
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(return_value=fail_result)

        result = await job._run_task(tasks_dir / "task-0")
        assert job._sdk.run.call_count == 3  # 1 + 2 retries
        assert result.error == "Agent install failed (rc=1): boom"

    @pytest.mark.asyncio
    async def test_non_retryable_exits_immediately(self, job_factory):
        """Non-retryable error (timeout) exits after 1 attempt."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=3)
        timeout_result = RunResult(
            task_name="task-0", error="Agent timed out after 900.0s"
        )
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(return_value=timeout_result)

        result = await job._run_task(tasks_dir / "task-0")
        assert job._sdk.run.call_count == 1
        assert result.error == "Agent timed out after 900.0s"

    @pytest.mark.asyncio
    async def test_idle_timeout_retries_even_with_zero_reward(self, job_factory):
        """Guards idle-timeout infra retry before accepting verifier fallback reward."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
        idle_result = RunResult(
            task_name="task-0",
            error="Agent idle for 600s with no new tool call, message, or thought",
            rewards={"reward": 0.0},
        )
        ok_result = RunResult(task_name="task-0", rewards={"reward": 1.0})
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(side_effect=[idle_result, ok_result])

        result = await job._run_task(tasks_dir / "task-0")

        assert job._sdk.run.call_count == 2
        assert result.rewards == {"reward": 1.0}

    @pytest.mark.asyncio
    async def test_verifier_timeout_retries_then_succeeds(self, job_factory):
        """Guards verifier timeout infra retry before marking task terminal."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
        timeout_result = RunResult(
            task_name="task-0",
            verifier_error="verifier timed out after 60s",
        )
        ok_result = RunResult(task_name="task-0", rewards={"reward": 1.0})
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(side_effect=[timeout_result, ok_result])

        result = await job._run_task(tasks_dir / "task-0")

        assert job._sdk.run.call_count == 2
        assert result.rewards == {"reward": 1.0}

    @pytest.mark.asyncio
    async def test_succeeds_on_retry(self, job_factory):
        """SDK fails once then succeeds — returns success result."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
        fail_result = RunResult(
            task_name="task-0", error="Agent install failed (rc=1): boom"
        )
        ok_result = RunResult(task_name="task-0", rewards={"reward": 1.0})
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(side_effect=[fail_result, ok_result])

        result = await job._run_task(tasks_dir / "task-0")
        assert job._sdk.run.call_count == 2
        assert result.rewards == {"reward": 1.0}

    @pytest.mark.asyncio
    async def test_retries_using_result_error_category(self, job_factory):
        """Guards PR #561: PTY marker timeouts retry via structured category."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
        marker_timeout = RunResult(
            task_name="task-0",
            error="DaytonaPtyProcess: timeout waiting for agent start marker",
            error_category="pipe_closed",
        )
        ok_result = RunResult(task_name="task-0", rewards={"reward": 1.0})
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(side_effect=[marker_timeout, ok_result])

        result = await job._run_task(tasks_dir / "task-0")

        assert job._sdk.run.call_count == 2
        assert result.rewards == {"reward": 1.0}


class TestJobResumeScoped:
    """Tests for Evaluation._get_completed_tasks scoped to job directory.

    Guards ENG-160: orphan retry artifacts and cross-job contamination.
    """

    def _setup_job(self, tmp_path, job_name="my-job"):
        """Create jobs_dir/job_name directory structure."""
        jobs_dir = tmp_path / "jobs"
        job_dir = jobs_dir / job_name
        job_dir.mkdir(parents=True)
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        return jobs_dir, job_dir, tasks_dir

    def _write_result(
        self,
        job_dir,
        task_name,
        rollout_suffix="abc",
        rewards=None,
        error=None,
        verifier_error=None,
    ):
        rollout_dir = job_dir / f"{task_name}__{rollout_suffix}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "task_name": task_name,
            "rewards": rewards,
            "error": error,
            "verifier_error": verifier_error,
        }
        result_file = rollout_dir / "result.json"
        result_file.write_text(json.dumps(data))
        return result_file

    def test_resume_scoped_to_job_dir(self, tmp_path):
        """_get_completed_tasks only finds results inside the current job directory.

        Guards ENG-160: prevents cross-job contamination on resume.
        """
        jobs_dir, job_dir, tasks_dir = self._setup_job(tmp_path, "run-1")
        self._write_result(job_dir, "task-a", rewards={"reward": 1.0})

        # Write a result in a DIFFERENT job directory — must not be found.
        other_job = jobs_dir / "run-2"
        other_job.mkdir()
        self._write_result(
            other_job, "task-b", rollout_suffix="xyz", rewards={"reward": 0.0}
        )

        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir, job_name="run-1")
        completed = job._get_completed_tasks()
        assert "task-a" in completed
        assert "task-b" not in completed

    def test_retry_dedup_picks_latest(self, tmp_path):
        """When retries create multiple result.json for the same task, newest wins.

        Guards ENG-160: orphan retry artifacts deduped by mtime.
        """
        jobs_dir, job_dir, tasks_dir = self._setup_job(tmp_path)
        # First attempt — errored (no reward).
        self._write_result(
            job_dir, "task-a", rollout_suffix="attempt1", error="install failed"
        )
        time.sleep(0.05)
        # Second attempt — succeeded.
        self._write_result(
            job_dir, "task-a", rollout_suffix="attempt2", rewards={"reward": 1.0}
        )

        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir, job_name="my-job")
        completed = job._get_completed_tasks()
        assert "task-a" in completed
        assert completed["task-a"]["rewards"] == {"reward": 1.0}

    def test_corrupt_file_skipped_scoped(self, tmp_path):
        """Corrupt result.json is silently skipped during resume scan."""
        jobs_dir, job_dir, tasks_dir = self._setup_job(tmp_path)
        self._write_result(job_dir, "task-a", rewards={"reward": 1.0})
        corrupt_dir = job_dir / "bad-rollout__zzz"
        corrupt_dir.mkdir()
        (corrupt_dir / "result.json").write_text("{invalid json")

        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir, job_name="my-job")
        completed = job._get_completed_tasks()
        assert "task-a" in completed
        assert len(completed) == 1

    def test_no_rewards_is_incomplete_scoped(self, tmp_path):
        """result.json with rewards=None and no verifier_error is NOT completed."""
        jobs_dir, job_dir, tasks_dir = self._setup_job(tmp_path)
        self._write_result(job_dir, "task-a", error="timeout")

        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir, job_name="my-job")
        completed = job._get_completed_tasks()
        assert len(completed) == 0

    @pytest.mark.asyncio
    async def test_config_mismatch_refuses_scoped(self, tmp_path):
        """Resume detects an agent mismatch from config.json inside the current
        job dir and refuses, rather than blending a different agent's scores.

        Guards ENG-160: config check scoped to job dir, not jobs_dir root.
        """
        jobs_dir, job_dir, tasks_dir = self._setup_job(tmp_path, "my-job")
        result_file = self._write_result(job_dir, "task-a", rewards={"reward": 1.0})
        (result_file.parent / "config.json").write_text(
            json.dumps({"agent": "old-agent"})
        )
        for name in ("task-a", "task-b"):
            (tasks_dir / name).mkdir(exist_ok=True)
            (tasks_dir / name / "task.toml").write_text(
                'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
                "[agent]\ntimeout_sec = 60\n[environment]\n"
            )
        cfg = EvaluationConfig(agent="new-agent")
        job = Evaluation(
            tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=cfg, job_name="my-job"
        )
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(
            return_value=RunResult(task_name="task-b", rewards={"reward": 1.0})
        )
        # The recorded agent ("old-agent") is named, proving detection is scoped
        # to this job dir, and the run refuses instead of mixing results.
        with pytest.raises(ResumeMismatchError, match="old-agent"):
            await job.run()

    def test_empty_job_dir_returns_empty(self, tmp_path):
        """Non-existent job directory returns empty completed dict."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        job = Evaluation(
            tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", job_name="nonexistent"
        )
        assert job._get_completed_tasks() == {}


class TestResolveJobName:
    """Tests for Evaluation._resolve_job_name — auto-detect job_name for resume.

    Guards ENG-160: auto-generated job_name must be stable across resume calls.
    """

    def test_no_existing_dirs_generates_timestamp(self, tmp_path):
        """Fresh jobs_dir with no subdirectories → new timestamp."""
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        name = Evaluation._resolve_job_name(jobs_dir)
        # Timestamp format: YYYY-MM-DD__HH-MM-SS (20 chars)
        assert "__" in name
        assert len(name) == 20

    def test_nonexistent_jobs_dir_generates_timestamp(self, tmp_path):
        """jobs_dir doesn't exist yet → new timestamp."""
        name = Evaluation._resolve_job_name(tmp_path / "nonexistent")
        assert "__" in name

    def test_single_job_dir_reuses_it(self, tmp_path):
        """Exactly one existing job dir → reuse it for stable resume.

        Guards ENG-160: second Evaluation call resumes into same directory.
        """
        jobs_dir = tmp_path / "jobs"
        (jobs_dir / "2026-05-23__12-00-00").mkdir(parents=True)
        name = Evaluation._resolve_job_name(jobs_dir)
        assert name == "2026-05-23__12-00-00"

    def test_multiple_job_dirs_picks_latest(self, tmp_path):
        """Multiple job dirs → pick the most recent (alphabetically last).

        Guards ENG-160: consistent resume target when multiple runs exist.
        """
        jobs_dir = tmp_path / "jobs"
        (jobs_dir / "2026-05-23__10-00-00").mkdir(parents=True)
        (jobs_dir / "2026-05-23__12-00-00").mkdir(parents=True)
        name = Evaluation._resolve_job_name(jobs_dir)
        assert name == "2026-05-23__12-00-00"

    def test_hidden_dirs_ignored(self, tmp_path):
        """Dotfiles/hidden dirs are not treated as job directories."""
        jobs_dir = tmp_path / "jobs"
        (jobs_dir / ".cache").mkdir(parents=True)
        name = Evaluation._resolve_job_name(jobs_dir)
        assert "__" in name  # Generated timestamp, not ".cache"


class TestSummaryInJobDir:
    """Guards ENG-160: summary.json written inside job directory.

    Verifies summary.json is written to both jobs_dir/job_name/summary.json
    (self-contained) and jobs_dir/summary.json (backward compat).
    """

    @pytest.mark.asyncio
    async def test_summary_written_to_job_dir_and_root(self, tmp_path):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "task-0").mkdir()
        (tasks_dir / "task-0" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
            "[agent]\ntimeout_sec = 60\n[environment]\n"
        )
        jobs_dir = tmp_path / "jobs"
        cfg = EvaluationConfig(retry=RetryConfig(max_retries=0))
        job = Evaluation(
            tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=cfg, job_name="test-run"
        )
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(
            return_value=RunResult(task_name="task-0", rewards={"reward": 1.0})
        )
        await job.run()

        # Both paths must exist and contain identical content.
        root_summary = jobs_dir / "summary.json"
        job_summary = jobs_dir / "test-run" / "summary.json"
        assert root_summary.exists()
        assert job_summary.exists()
        assert json.loads(root_summary.read_text()) == json.loads(
            job_summary.read_text()
        )


class TestJobRunOrchestration:
    """Tests for Evaluation.run() orchestration: semaphore overlap, exception catching."""

    def _make_job(self, tmp_path, n_tasks: int, concurrency: int = 2) -> Evaluation:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        for i in range(n_tasks):
            (tasks_dir / f"task-{i}").mkdir()
            (tasks_dir / f"task-{i}" / "task.toml").write_text(
                'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n'
            )
        cfg = EvaluationConfig(
            concurrency=concurrency, retry=RetryConfig(max_retries=0)
        )
        return Evaluation(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)

    @pytest.mark.asyncio
    async def test_concurrency_semaphore_actually_overlaps_at_bound(self, tmp_path):
        """Prove the asyncio.Semaphore at job.py:464 PERMITS concurrency-many
        tasks at once. ``==`` (not ``<=``) — a broken semaphore that serializes
        would still satisfy ``<= concurrency`` vacuously.
        """
        concurrency = 2
        job = self._make_job(tmp_path, n_tasks=5, concurrency=concurrency)

        counter = 0
        max_in_flight = 0
        lock = asyncio.Lock()
        enough_in_flight = asyncio.Event()

        async def fake_sdk_run(*args, **kwargs):
            nonlocal counter, max_in_flight
            async with lock:
                counter += 1
                max_in_flight = max(max_in_flight, counter)
                if counter >= concurrency:
                    enough_in_flight.set()
            # All waiters block here until ``concurrency`` tasks have entered,
            # then asyncio.Event.set() releases them simultaneously. Decrement
            # AFTER the wait so the peak is observable.
            await enough_in_flight.wait()
            async with lock:
                counter -= 1
            return RunResult(task_name="task", rewards={"reward": 1.0})

        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(side_effect=fake_sdk_run)

        await job.run()
        assert max_in_flight == concurrency

    @pytest.mark.asyncio
    async def test_unexpected_sdk_exception_becomes_errored_result(
        self, tmp_path, caplog
    ):
        """Prove the gather(return_exceptions=True) catch branch at job.py:491-502
        catches a non-classified exception, increments ``errored``, and logs.

        The synthesized RunResult(error="Unexpected: ...") is built in-Python
        and never written to result.json (SDK._build_result never runs when
        SDK.run raises), so we assert via EvaluationResult and caplog, not disk.
        """
        job = self._make_job(tmp_path, n_tasks=1, concurrency=1)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(side_effect=RuntimeError("boom"))

        with caplog.at_level(logging.ERROR):
            result = await job.run()

        assert result.errored == 1
        assert any("unexpected exception: boom" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_error_and_verifier_error_does_not_crash_invariant(self, tmp_path):
        """A result with BOTH error and verifier_error (rewards=None) must not
        be double-counted — otherwise the passed+failed+errored+verifier_errored
        == total assertion in Evaluation.run() crashes the whole evaluation.

        Guards the fix from PR #320 for audit Finding 6.
        """
        job = self._make_job(tmp_path, n_tasks=1, concurrency=1)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(
            return_value=RunResult(
                task_name="task-0",
                rewards=None,
                error="agent crashed",
                verifier_error="verifier also failed",
            )
        )

        # Must not raise the "Counting bug" AssertionError.
        result = await job.run()

        assert (
            result.passed + result.failed + result.errored + result.verifier_errored
            == result.total
            == 1
        )
        # Agent error takes precedence: counted as errored, not verifier_errored.
        assert result.errored == 1
        assert result.verifier_errored == 0

    @pytest.mark.asyncio
    async def test_summary_json_includes_usage_aggregation(self, tmp_path):
        job = self._make_job(tmp_path, n_tasks=3, concurrency=1)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(
            side_effect=[
                RunResult(
                    task_name="task-0",
                    rewards={"reward": 1.0},
                    n_input_tokens=100,
                    n_output_tokens=10,
                    n_cache_read_tokens=5,
                    n_cache_creation_tokens=1,
                    total_tokens=116,
                    cost_usd=0.001,
                    usage_source="provider_response",
                    price_source="pricing_table_2026-05",
                ),
                RunResult(
                    task_name="task-1",
                    rewards={"reward": 1.0},
                    n_input_tokens=200,
                    n_output_tokens=20,
                    n_cache_read_tokens=0,
                    n_cache_creation_tokens=4,
                    total_tokens=224,
                    cost_usd=0.002,
                    usage_source="provider_response",
                    price_source="pricing_table_2026-05",
                ),
                RunResult(
                    task_name="task-2",
                    rewards={"reward": 1.0},
                    usage_source="unavailable",
                ),
            ]
        )

        await job.run()

        summary = json.loads((job._jobs_dir / "summary.json").read_text())
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 30
        assert summary["total_cache_read_tokens"] == 5
        assert summary["total_cache_creation_tokens"] == 5
        assert summary["total_tokens"] == 340
        assert summary["total_cost_usd"] == 0.003
        assert summary["avg_cost_per_trial_usd"] == 0.0015
        assert summary["telemetry_coverage"] == 2 / 3

    @pytest.mark.asyncio
    async def test_summary_json_aggregates_tool_calls_and_phase_timing(self, tmp_path):
        """Issue #501: summary.json must aggregate per-rollout n_tool_calls and
        timing so reviewers don't have to inspect each result.json by hand.

        We seed the resume path with three pre-written result.json files that
        carry known n_tool_calls and timing blocks, then assert the aggregates
        match by hand-computation. Using the resume path keeps the test free
        of the rollout writer machinery while still exercising the exact
        ``all_results`` shape that drives ``phase_timing_summary``.
        """
        job = self._make_job(tmp_path, n_tasks=3, concurrency=1)
        job_dir = job._jobs_dir / job._job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        seeded = [
            {
                "task_name": "task-0",
                "n_tool_calls": 4,
                "trajectory_summary": {"steps": 9, "tool_call_steps": 4},
                "timing": {
                    "environment_setup": 1.0,
                    "agent_setup": 2.0,
                    "agent_execution": 10.0,
                    "verifier": 3.0,
                    "total": 16.0,
                },
            },
            {
                "task_name": "task-1",
                "n_tool_calls": 6,
                "trajectory_summary": {"steps": 13, "tool_call_steps": 6},
                "timing": {
                    "environment_setup": 1.5,
                    "agent_setup": 2.5,
                    "agent_execution": 20.0,
                    "verifier": 5.0,
                    "total": 29.0,
                },
            },
            {
                "task_name": "task-2",
                "n_tool_calls": 2,
                "trajectory_summary": {"steps": 5, "tool_call_steps": 2},
                "timing": {
                    "environment_setup": 0.5,
                    "agent_setup": 1.0,
                    "agent_execution": 4.0,
                    "verifier": 1.0,
                    "total": 6.5,
                },
            },
        ]
        for entry in seeded:
            rdir = job_dir / entry["task_name"]
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / "result.json").write_text(
                json.dumps({**entry, "rewards": {"reward": 1.0}}, indent=2)
            )

        await job.run()

        summary = json.loads((job_dir / "summary.json").read_text())
        # Tool-call aggregates — counts every rollout regardless of outcome.
        assert summary["total_tool_calls"] == 12
        assert summary["avg_tool_calls_per_task"] == pytest.approx(4.0)
        assert summary["max_tool_calls_per_task"] == 6
        # Harbor-style trajectory aggregates track every ACP event and the
        # subset of events that carried tool calls.
        assert summary["total_trajectory_steps"] == 27
        assert summary["avg_trajectory_steps_per_task"] == pytest.approx(9.0)
        assert summary["max_trajectory_steps_per_task"] == 13
        assert summary["total_trajectory_tool_call_steps"] == 12
        assert summary["avg_trajectory_tool_call_steps_per_task"] == pytest.approx(4.0)
        assert summary["max_trajectory_tool_call_steps_per_task"] == 6
        assert summary["trajectory_summary_coverage"] == 1.0
        # Phase totals (sums) — hand-computed from the seeded timing blocks.
        assert summary["environment_setup_time_sec"] == 3.0
        assert summary["agent_setup_time_sec"] == 5.5
        assert summary["agent_execution_time_sec"] == 34.0
        assert summary["verifier_time_sec"] == 9.0
        assert summary["total_time_sec"] == 51.5
        # Averages and maxes round-trip with rollout.py's 0.1s precision.
        assert summary["avg_verifier_time_sec"] == pytest.approx(3.0)
        assert summary["max_verifier_time_sec"] == 5.0
        assert summary["max_agent_execution_time_sec"] == 20.0
        # Coverage is 1.0 — every rollout reported a timing block.
        assert summary["timing_coverage"] == 1.0

    @pytest.mark.asyncio
    async def test_summary_json_phase_timing_handles_missing_blocks(self, tmp_path):
        """Mocked rollouts (no timing in RunResult, no rollout_name) must not
        crash phase_timing_summary — they just lower timing_coverage to 0.0
        while tool-call aggregation still works.
        """
        job = self._make_job(tmp_path, n_tasks=2, concurrency=1)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(
            side_effect=[
                RunResult(
                    task_name="task-0",
                    rewards={"reward": 1.0},
                    n_tool_calls=7,
                ),
                RunResult(
                    task_name="task-1",
                    rewards={"reward": 1.0},
                    n_tool_calls=3,
                ),
            ]
        )

        await job.run()

        summary = json.loads((job._jobs_dir / "summary.json").read_text())
        assert summary["total_tool_calls"] == 10
        assert summary["max_tool_calls_per_task"] == 7
        assert summary["timing_coverage"] == 0.0
        # No data → sums collapse to 0.0 and avg/max are null (distinguishable
        # from "ran but cost nothing").
        assert summary["agent_execution_time_sec"] == 0.0
        assert summary["avg_agent_execution_time_sec"] is None
        assert summary["max_agent_execution_time_sec"] is None
