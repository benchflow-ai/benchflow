"""Tests for verifier failure isolation — verifier_error field, retry, resume, metrics."""

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._models import RunResult
from benchflow._scoring import (
    VERIFIER_FAILED,
    VERIFIER_TIMEOUT,
    classify_verifier_error,
    extract_reward,
)
from benchflow.metrics import BenchmarkMetrics, TaskMetrics


# ---------------------------------------------------------------------------
# classify_verifier_error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_str,expected", [
    (None, None),
    ("", None),
    ("verifier crashed: ImportError", VERIFIER_FAILED),
    ("verifier timed out after 900s", VERIFIER_TIMEOUT),
    ("verifier did something weird", "verifier_other"),
])
def test_classify_verifier_error(input_str, expected):
    assert classify_verifier_error(input_str) == expected


# ---------------------------------------------------------------------------
# RunResult with verifier_error
# ---------------------------------------------------------------------------

class TestRunResultVerifierError:

    def test_success_requires_no_errors(self):
        assert RunResult(task_name="t", rewards={"reward": 1.0}).success is True
        assert RunResult(task_name="t", error="x").success is False
        assert RunResult(task_name="t", verifier_error="x").success is False

    def test_repr_shows_verifier_error(self):
        r = RunResult(task_name="t", verifier_error="verifier timed out after 900s")
        assert "ERROR: verifier timed out after 900s" in repr(r)

    def test_verifier_error_default_none(self):
        r = RunResult(task_name="t", error="install failed (rc=1)")
        assert r.verifier_error is None
        assert r.success is False


# ---------------------------------------------------------------------------
# Result JSON round-trip via _build_result
# ---------------------------------------------------------------------------

class TestResultJson:

    def _build(self, tmp_path, **overrides):
        from benchflow.sdk import SDK
        from datetime import datetime
        defaults = dict(
            task_name="t1", trial_name="trial-1", agent="test",
            agent_name="", model="", n_tool_calls=0, prompts=["x"],
            error=None, verifier_error=None, trajectory=[],
            partial_trajectory=False, rewards={"reward": 1.0},
            started_at=datetime.now(), timing={},
        )
        defaults.update(overrides)
        SDK._build_result(tmp_path, **defaults)
        return json.loads((tmp_path / "result.json").read_text())

    def test_verifier_error_in_json(self, tmp_path):
        data = self._build(tmp_path, verifier_error="verifier crashed: KeyError", rewards=None)
        assert data["verifier_error"] == "verifier crashed: KeyError"
        assert data["error"] is None
        assert data["rewards"] is None

    def test_clean_run_json(self, tmp_path):
        data = self._build(tmp_path)
        assert data["verifier_error"] is None
        assert data["rewards"] == {"reward": 1.0}


# ---------------------------------------------------------------------------
# SDK._verify() integration
# ---------------------------------------------------------------------------

class TestSdkVerify:

    @pytest.fixture
    def verify_harness(self, tmp_path):
        from benchflow.sdk import SDK
        sdk = SDK()
        task = MagicMock()
        task.config.verifier.timeout_sec = 5
        tp = MagicMock()
        tp.verifier_dir = tmp_path / "verifier"
        env = MagicMock()
        return sdk, env, task, tp

    @pytest.mark.asyncio
    async def test_verifier_timeout(self, verify_harness):
        sdk, env, task, tp = verify_harness
        task.config.verifier.timeout_sec = 0.1
        mock_v = MagicMock()
        mock_v.verify = lambda: asyncio.sleep(10)
        timing = {}
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            rewards, verifier_error = await sdk._verify(env, task, tp, timing)
        assert rewards is None
        assert "timed out" in verifier_error
        assert "verifier" in timing

    @pytest.mark.asyncio
    async def test_verifier_crash(self, verify_harness):
        sdk, env, task, tp = verify_harness
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(side_effect=RuntimeError("kaboom"))
        timing = {}
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            rewards, verifier_error = await sdk._verify(env, task, tp, timing)
        assert rewards is None
        assert "crashed" in verifier_error and "kaboom" in verifier_error

    @pytest.mark.asyncio
    async def test_verifier_success(self, verify_harness):
        sdk, env, task, tp = verify_harness
        mock_result = MagicMock()
        mock_result.rewards = {"reward": 1.0}
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(return_value=mock_result)
        timing = {}
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            rewards, verifier_error = await sdk._verify(env, task, tp, timing)
        assert rewards == {"reward": 1.0}
        assert verifier_error is None


# ---------------------------------------------------------------------------
# Job: retry, resume, bounded log, threshold warning
# ---------------------------------------------------------------------------

@pytest.fixture
def job_factory(tmp_path):
    """Create a Job with n task directories and a mocked SDK."""
    from benchflow.job import Job, JobConfig, RetryConfig

    def _make(n_tasks=1, max_retries=0):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir(exist_ok=True)
        for i in range(n_tasks):
            td = tasks_dir / f"task-{i}"
            td.mkdir(exist_ok=True)
            (td / "task.toml").write_text(
                'version = "1.0"\n[verifier]\ntimeout_sec = 60\n'
                '[agent]\ntimeout_sec = 60\n[environment]\n'
            )
        cfg = JobConfig(retry=RetryConfig(max_retries=max_retries))
        job = Job(tasks_dir=tasks_dir, jobs_dir=tmp_path / "jobs", config=cfg)
        return job, tasks_dir
    return _make


class TestRetry:

    @pytest.mark.asyncio
    async def test_verifier_error_is_terminal(self, job_factory):
        """Verifier errors exit after 1 attempt — no retry."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(return_value=RunResult(
            task_name="task-0", verifier_error="verifier crashed: x",
        ))
        result = await job._run_task(tasks_dir / "task-0")
        assert job._sdk.run.call_count == 1
        assert result.verifier_error == "verifier crashed: x"

    @pytest.mark.asyncio
    async def test_agent_error_still_retries(self, job_factory):
        """Agent install errors are retried."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(return_value=RunResult(
            task_name="task-0", error="Agent claude-agent-acp install failed (rc=1)",
        ))
        await job._run_task(tasks_dir / "task-0")
        assert job._sdk.run.call_count == 3  # 1 + 2 retries


class TestResume:

    def test_verifier_errored_is_complete(self, tmp_path, caplog):
        task_dir = tmp_path / "task1" / "trial-1"
        task_dir.mkdir(parents=True)
        (task_dir / "result.json").write_text(json.dumps({
            "task_name": "task1", "rewards": None,
            "error": None, "verifier_error": "verifier timed out after 900s",
        }))
        from benchflow.job import Job, JobConfig
        job = Job(tasks_dir=tmp_path, jobs_dir=tmp_path, config=JobConfig())
        with caplog.at_level(logging.INFO):
            completed = job._get_completed_tasks()
        assert "task1" in completed
        assert any("Skipping verifier-errored task" in m for m in caplog.messages)

    def test_agent_errored_not_complete(self, tmp_path):
        task_dir = tmp_path / "task2" / "trial-1"
        task_dir.mkdir(parents=True)
        (task_dir / "result.json").write_text(json.dumps({
            "task_name": "task2", "rewards": None,
            "error": "install failed", "verifier_error": None,
        }))
        from benchflow.job import Job, JobConfig
        job = Job(tasks_dir=tmp_path, jobs_dir=tmp_path, config=JobConfig())
        assert "task2" not in job._get_completed_tasks()


class TestJobRunLogs:
    """Tests that exercise actual Job.run() and check log output."""

    @pytest.mark.asyncio
    async def test_bounded_log_shows_verifier_error(self, job_factory, caplog):
        job, _ = job_factory(n_tasks=1)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(return_value=RunResult(
            task_name="task-0", verifier_error="verifier crashed: KeyError",
        ))
        with caplog.at_level(logging.INFO):
            await job.run()
        assert any("verifier crashed" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_over_20_pct_threshold_error(self, job_factory, caplog):
        job, _ = job_factory(n_tasks=3)
        call_count = 0
        async def make_result(**kwargs):
            nonlocal call_count
            r = RunResult(task_name=f"task-{call_count}", verifier_error="verifier crashed: x")
            call_count += 1
            return r
        job._sdk = AsyncMock()
        job._sdk.run = make_result
        with caplog.at_level(logging.WARNING):
            await job.run()
        warning_records = [r for r in caplog.records if "had verifier errors" in r.message]
        assert warning_records and warning_records[0].levelno == logging.WARNING
        error_records = [r for r in caplog.records if "Over 20%" in r.message]
        assert error_records and error_records[0].levelno == logging.ERROR

    @pytest.mark.asyncio
    async def test_under_20_pct_no_error(self, job_factory, caplog):
        job, _ = job_factory(n_tasks=5)
        results = [
            RunResult(task_name=f"task-{i}", rewards={"reward": 1.0}) for i in range(4)
        ] + [RunResult(task_name="task-4", verifier_error="verifier crashed: x")]
        idx = 0
        async def make_result(**kwargs):
            nonlocal idx
            r = results[idx]; idx += 1; return r
        job._sdk = AsyncMock()
        job._sdk.run = make_result
        with caplog.at_level(logging.WARNING):
            await job.run()
        assert any("had verifier errors" in r.message for r in caplog.records)
        assert not any("Over 20%" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_summary_json_includes_verifier_errored(self, job_factory):
        job, _ = job_factory(n_tasks=1)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(return_value=RunResult(
            task_name="task-0", verifier_error="verifier crashed: x",
        ))
        await job.run()
        summary = json.loads((job._jobs_dir / "summary.json").read_text())
        assert summary["verifier_errored"] == 1


# ---------------------------------------------------------------------------
# JobResult invariant
# ---------------------------------------------------------------------------

def test_total_invariant():
    from benchflow.job import JobResult, JobConfig
    jr = JobResult(job_name="t", config=JobConfig(), total=4,
                   passed=1, failed=1, errored=1, verifier_errored=1)
    assert jr.passed + jr.failed + jr.errored + jr.verifier_errored == jr.total

def test_double_count_violates_invariant():
    """Both error and verifier_error set would double-count — documents mutual exclusivity."""
    r = {"rewards": None, "error": "x", "verifier_error": "y"}
    errored = 1 if r.get("error") and r.get("rewards") is None else 0
    v_errored = 1 if r.get("verifier_error") else 0
    assert errored + v_errored > 1  # proves double-counting


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_metrics():
    return BenchmarkMetrics(
        benchmark="test", agent="test", model="test",
        tasks=[
            TaskMetrics(task_name="pass1", reward=1.0, n_tool_calls=3, duration_sec=10),
            TaskMetrics(task_name="fail1", reward=0.0, n_tool_calls=5, duration_sec=20),
            TaskMetrics(task_name="err1", reward=None, error="timed out", n_tool_calls=1, duration_sec=5),
            TaskMetrics(task_name="verr1", reward=None, verifier_error="verifier crashed: x", n_tool_calls=100, duration_sec=999),
            TaskMetrics(task_name="verr2", reward=None, verifier_error="verifier timed out after 900s", n_tool_calls=50, duration_sec=500),
        ],
    )


class TestMetricsVerifierError:

    def test_counts(self, sample_metrics):
        assert sample_metrics.verifier_errored == 2
        assert sample_metrics.errored == 1

    def test_error_breakdowns_are_separate(self, sample_metrics):
        assert VERIFIER_FAILED not in sample_metrics.error_breakdown
        bd = sample_metrics.verifier_error_breakdown
        assert bd[VERIFIER_FAILED] == 1
        assert bd[VERIFIER_TIMEOUT] == 1

    def test_averages_exclude_verifier_errored(self, sample_metrics):
        # Only pass1 (3/10) and fail1 (5/20)
        assert sample_metrics.avg_tool_calls == 4.0
        assert sample_metrics.avg_duration == 15.0

    def test_score_excl_errors(self, sample_metrics):
        assert sample_metrics.score_excl_errors == 0.5  # 1 passed / (1+1)

    def test_summary_includes_verifier_fields(self, sample_metrics):
        s = sample_metrics.summary()
        assert s["verifier_errored"] == 2
        assert sorted(s["verifier_errored_tasks"]) == ["verr1", "verr2"]
        assert "verifier_error_breakdown" in s

    def test_collect_metrics_reads_verifier_error(self, tmp_path):
        from benchflow.metrics import collect_metrics
        from datetime import datetime
        task_dir = tmp_path / "task1" / "trial-1"
        task_dir.mkdir(parents=True)
        now = datetime.now().isoformat()
        (task_dir / "result.json").write_text(json.dumps({
            "task_name": "task1", "rewards": None, "error": None,
            "verifier_error": "verifier crashed: KeyError",
            "n_tool_calls": 5, "n_prompts": 1,
            "started_at": now, "finished_at": now,
        }))
        m = collect_metrics(tmp_path)
        assert m.tasks[0].verifier_error == "verifier crashed: KeyError"
        assert m.tasks[0].verifier_errored is True


@pytest.mark.parametrize("reward,error,verifier_error,expected", [
    (None, None, "verifier crashed: x", True),
    (1.0, None, None, False),
    (None, "timed out", None, False),
    (0.0, None, "verifier crashed: x", False),  # reward set → not verifier_errored
])
def test_task_metrics_verifier_errored(reward, error, verifier_error, expected):
    t = TaskMetrics(task_name="t", reward=reward, error=error, verifier_error=verifier_error)
    assert t.verifier_errored is expected
