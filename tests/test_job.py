"""Tests for job counting logic and result aggregation."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock

import pytest

from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
from benchflow.models import RunResult


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


class TestJobResume:
    """Tests for Evaluation._get_completed_tasks — resume logic."""

    def _setup_jobs_dir(self, tmp_path):
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir()
        return jobs_dir

    def _write_result(self, jobs_dir, task_name, rewards=None, error=None):
        trial_dir = jobs_dir / f"trial-{task_name}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        data = {"task_name": task_name, "rewards": rewards, "error": error}
        (trial_dir / "result.json").write_text(json.dumps(data))
        return trial_dir

    def test_valid_resume(self, tmp_path):
        jobs_dir = self._setup_jobs_dir(tmp_path)
        self._write_result(jobs_dir, "task-a", rewards={"reward": 1.0})
        self._write_result(jobs_dir, "task-b", rewards={"reward": 0.0})

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir)
        completed = job._get_completed_tasks()
        assert "task-a" in completed
        assert "task-b" in completed
        assert len(completed) == 2

    def test_corrupt_file_skipped(self, tmp_path):
        jobs_dir = self._setup_jobs_dir(tmp_path)
        self._write_result(jobs_dir, "task-a", rewards={"reward": 1.0})
        # Write corrupt JSON
        corrupt_dir = jobs_dir / "trial-corrupt"
        corrupt_dir.mkdir()
        (corrupt_dir / "result.json").write_text("{invalid json")

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir)
        completed = job._get_completed_tasks()
        assert "task-a" in completed
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_config_mismatch_warning(self, tmp_path, caplog):
        """Guards the event-loop flake observed on v0.5-integration@ffef85d."""
        jobs_dir = self._setup_jobs_dir(tmp_path)
        trial_dir = self._write_result(jobs_dir, "task-a", rewards={"reward": 1.0})
        (trial_dir / "config.json").write_text(json.dumps({"agent": "old-agent"}))

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        # Create task dirs so remaining is computed
        (tasks_dir / "task-a").mkdir()
        (tasks_dir / "task-a" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n'
        )
        (tasks_dir / "task-b").mkdir()
        (tasks_dir / "task-b" / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 60\n[agent]\ntimeout_sec = 60\n[environment]\n'
        )
        cfg = EvaluationConfig(agent="new-agent")
        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=cfg)
        # Patch _run_task so run() doesn't actually run anything real
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(
            return_value=RunResult(task_name="task-b", rewards={"reward": 1.0})
        )
        with caplog.at_level(logging.WARNING):
            await job.run()
        assert any("old-agent" in msg for msg in caplog.messages)

    def test_no_rewards_is_incomplete(self, tmp_path):
        """result.json with rewards=None is NOT treated as completed."""
        jobs_dir = self._setup_jobs_dir(tmp_path)
        self._write_result(jobs_dir, "task-a", rewards=None, error="timeout")

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        job = Evaluation(tasks_dir=tasks_dir, jobs_dir=jobs_dir)
        completed = job._get_completed_tasks()
        assert "task-a" not in completed
        assert len(completed) == 0


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
