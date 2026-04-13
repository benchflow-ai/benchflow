"""Tests for verifier failure isolation — verifier_error field, retry, resume, metrics."""

import asyncio
import contextlib
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow._scoring import (
    VERIFIER_FAILED,
    VERIFIER_TIMEOUT,
    classify_verifier_error,
)
from benchflow.metrics import BenchmarkMetrics, TaskMetrics
from benchflow.models import RunResult

# ---------------------------------------------------------------------------
# classify_verifier_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_str,expected",
    [
        (None, None),
        ("", None),
        ("verifier crashed: ImportError", VERIFIER_FAILED),
        ("verifier timed out after 900s", VERIFIER_TIMEOUT),
        ("verifier did something weird", "verifier_other"),
    ],
)
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


# ---------------------------------------------------------------------------
# Result JSON round-trip via _build_result
# ---------------------------------------------------------------------------


class TestResultJson:
    def _build(self, tmp_path, **overrides):
        from datetime import datetime

        from benchflow.sdk import SDK

        defaults = dict(
            task_name="t1",
            trial_name="trial-1",
            agent="test",
            agent_name="",
            model="",
            n_tool_calls=0,
            prompts=["x"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
        )
        defaults.update(overrides)
        SDK._build_result(tmp_path, **defaults)
        return json.loads((tmp_path / "result.json").read_text())

    def test_verifier_error_in_json(self, tmp_path):
        data = self._build(
            tmp_path, verifier_error="verifier crashed: KeyError", rewards=None
        )
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
        task.config.verifier.env = None
        tp = MagicMock()
        tp.verifier_dir = tmp_path / "verifier"
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
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
                "[agent]\ntimeout_sec = 60\n[environment]\n"
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
        job._sdk.run = AsyncMock(
            return_value=RunResult(
                task_name="task-0",
                verifier_error="verifier crashed: x",
            )
        )
        result = await job._run_task(tasks_dir / "task-0")
        assert job._sdk.run.call_count == 1
        assert result.verifier_error == "verifier crashed: x"

    @pytest.mark.asyncio
    async def test_agent_error_still_retries(self, job_factory):
        """Agent install errors are retried."""
        job, tasks_dir = job_factory(n_tasks=1, max_retries=2)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(
            return_value=RunResult(
                task_name="task-0",
                error="Agent claude-agent-acp install failed (rc=1)",
            )
        )
        await job._run_task(tasks_dir / "task-0")
        assert job._sdk.run.call_count == 3  # 1 + 2 retries


class TestResume:
    def test_verifier_errored_is_complete(self, tmp_path, caplog):
        task_dir = tmp_path / "task1" / "trial-1"
        task_dir.mkdir(parents=True)
        (task_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task1",
                    "rewards": None,
                    "error": None,
                    "verifier_error": "verifier timed out after 900s",
                }
            )
        )
        from benchflow.job import Job, JobConfig

        job = Job(tasks_dir=tmp_path, jobs_dir=tmp_path, config=JobConfig())
        with caplog.at_level(logging.INFO):
            completed = job._get_completed_tasks()
        assert "task1" in completed
        assert any("Skipping verifier-errored task" in m for m in caplog.messages)

    def test_agent_errored_not_complete(self, tmp_path):
        task_dir = tmp_path / "task2" / "trial-1"
        task_dir.mkdir(parents=True)
        (task_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task2",
                    "rewards": None,
                    "error": "install failed",
                    "verifier_error": None,
                }
            )
        )
        from benchflow.job import Job, JobConfig

        job = Job(tasks_dir=tmp_path, jobs_dir=tmp_path, config=JobConfig())
        assert "task2" not in job._get_completed_tasks()


class TestJobRunLogs:
    """Tests that exercise actual Job.run() and check log output."""

    @pytest.mark.asyncio
    async def test_bounded_log_shows_verifier_error(self, job_factory, caplog):
        job, _ = job_factory(n_tasks=1)
        job._sdk = AsyncMock()
        job._sdk.run = AsyncMock(
            return_value=RunResult(
                task_name="task-0",
                verifier_error="verifier crashed: KeyError",
            )
        )
        with caplog.at_level(logging.INFO):
            await job.run()
        assert any("verifier crashed" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_over_20_pct_threshold_error(self, job_factory, caplog):
        job, _ = job_factory(n_tasks=3)
        call_count = 0

        async def make_result(**kwargs):
            nonlocal call_count
            r = RunResult(
                task_name=f"task-{call_count}", verifier_error="verifier crashed: x"
            )
            call_count += 1
            return r

        job._sdk = AsyncMock()
        job._sdk.run = make_result
        with caplog.at_level(logging.WARNING):
            await job.run()
        warning_records = [
            r for r in caplog.records if "had verifier errors" in r.message
        ]
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
            r = results[idx]
            idx += 1
            return r

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
        job._sdk.run = AsyncMock(
            return_value=RunResult(
                task_name="task-0",
                verifier_error="verifier crashed: x",
            )
        )
        await job.run()
        summary = json.loads((job._jobs_dir / "summary.json").read_text())
        assert summary["verifier_errored"] == 1


# ---------------------------------------------------------------------------
# JobResult invariant
# ---------------------------------------------------------------------------


def test_total_invariant():
    from benchflow.job import JobConfig, JobResult

    jr = JobResult(
        job_name="t",
        config=JobConfig(),
        total=4,
        passed=1,
        failed=1,
        errored=1,
        verifier_errored=1,
    )
    assert jr.passed + jr.failed + jr.errored + jr.verifier_errored == jr.total


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metrics():
    return BenchmarkMetrics(
        benchmark="test",
        agent="test",
        model="test",
        tasks=[
            TaskMetrics(task_name="pass1", reward=1.0, n_tool_calls=3, duration_sec=10),
            TaskMetrics(task_name="fail1", reward=0.0, n_tool_calls=5, duration_sec=20),
            TaskMetrics(
                task_name="err1",
                reward=None,
                error="timed out",
                n_tool_calls=1,
                duration_sec=5,
            ),
            TaskMetrics(
                task_name="verr1",
                reward=None,
                verifier_error="verifier crashed: x",
                n_tool_calls=100,
                duration_sec=999,
            ),
            TaskMetrics(
                task_name="verr2",
                reward=None,
                verifier_error="verifier timed out after 900s",
                n_tool_calls=50,
                duration_sec=500,
            ),
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
        from datetime import datetime

        from benchflow.metrics import collect_metrics

        task_dir = tmp_path / "task1" / "trial-1"
        task_dir.mkdir(parents=True)
        now = datetime.now().isoformat()
        (task_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "task1",
                    "rewards": None,
                    "error": None,
                    "verifier_error": "verifier crashed: KeyError",
                    "n_tool_calls": 5,
                    "n_prompts": 1,
                    "started_at": now,
                    "finished_at": now,
                }
            )
        )
        m = collect_metrics(tmp_path)
        assert m.tasks[0].verifier_error == "verifier crashed: KeyError"
        assert m.tasks[0].verifier_errored is True


@pytest.mark.parametrize(
    "reward,error,verifier_error,expected",
    [
        (None, None, "verifier crashed: x", True),
        (1.0, None, None, False),
        (None, "timed out", None, False),
        (0.0, None, "verifier crashed: x", False),  # reward set → not verifier_errored
    ],
)
def test_task_metrics_verifier_errored(reward, error, verifier_error, expected):
    t = TaskMetrics(
        task_name="t", reward=reward, error=error, verifier_error=verifier_error
    )
    assert t.verifier_errored is expected


# ---------------------------------------------------------------------------
# Verifier hardening (PR 2)
# ---------------------------------------------------------------------------


class TestVerifierHardening:
    """Pre-verification hardening: pkill, conftest cleanup, env injection."""

    @pytest.fixture
    def hardening_harness(self, tmp_path):
        from benchflow.sdk import SDK

        sdk = SDK()
        task = MagicMock()
        task.config.verifier.timeout_sec = 5
        task.config.verifier.env = None
        tp = MagicMock()
        tp.verifier_dir = tmp_path / "verifier"
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
        return sdk, env, task, tp

    @pytest.mark.asyncio
    async def test_hardening_with_sandbox_user(self, hardening_harness):
        """With sandbox_user: pkill runs first, cleanup runs, env injected."""
        sdk, env, task, tp = hardening_harness
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            await sdk._verify(env, task, tp, {}, sandbox_user="agent")

        exec_cmds = [c[0][0] for c in env.exec.call_args_list]
        # pkill is first call
        assert "pkill -u agent" in exec_cmds[0]
        # /logs/verifier/ wipe-and-recreate runs after pkill and before cleanup
        wipe_idx = next(
            (i for i, c in enumerate(exec_cmds) if "rm -rf /logs/verifier" in c), None
        )
        assert wipe_idx is not None, "rm -rf /logs/verifier not found in exec calls"
        cleanup_idx = next(
            (i for i, c in enumerate(exec_cmds) if "conftest.py" in c), None
        )
        assert cleanup_idx is not None
        assert wipe_idx < cleanup_idx, "wipe must run before CLEANUP_CMD"
        assert any("mkdir -p /logs/verifier" in c for c in exec_cmds)
        # Cleanup runs (conftest, sitecustomize, .pth — all in one command)
        cleanup = [c for c in exec_cmds if "conftest.py" in c]
        assert cleanup and "sitecustomize.py" in cleanup[0] and ".pth" in cleanup[0]
        assert "-not -path '/tests/*'" in cleanup[0]
        # Env injection — all _VERIFIER_ENV keys present
        injected = task.config.verifier.env
        assert (
            injected["PATH"]
            == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        assert "--rootdir=/tests" in injected["PYTEST_ADDOPTS"]
        assert "-p no:cacheprovider" in injected["PYTEST_ADDOPTS"]
        assert injected["PYTHONPATH"] == ""
        assert (
            "PYTHONHOME" not in injected
        )  # see _sandbox.py comment — breaks Py_Initialize
        assert injected["PYTHONDONTWRITEBYTECODE"] == "1"

    @pytest.mark.asyncio
    async def test_hardening_without_sandbox_user(self, hardening_harness):
        """Without sandbox_user: no pkill, cleanup still runs, env still injected."""
        sdk, env, task, tp = hardening_harness
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            await sdk._verify(env, task, tp, {}, sandbox_user=None)

        exec_cmds = [c[0][0] for c in env.exec.call_args_list]
        assert all("pkill" not in c for c in exec_cmds)
        assert any("conftest.py" in c for c in exec_cmds)
        addopts = task.config.verifier.env["PYTEST_ADDOPTS"]
        assert "--rootdir=/tests" in addopts
        assert "-p no:cacheprovider" in addopts

    @pytest.mark.asyncio
    async def test_task_env_overrides_win(self, hardening_harness):
        """Task-level verifier env vars override _VERIFIER_ENV defaults."""
        sdk, env, task, tp = hardening_harness
        task.config.verifier.env = {"PATH": "/custom/bin", "MY_VAR": "hello"}
        mock_v = MagicMock()
        mock_v.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
        with patch("benchflow.sdk.Verifier", return_value=mock_v):
            await sdk._verify(env, task, tp, {})
        injected = task.config.verifier.env
        assert injected["PATH"] == "/custom/bin"
        assert injected["MY_VAR"] == "hello"
        assert injected["PYTHONPATH"] == ""  # non-overridden defaults kept

    @pytest.mark.asyncio
    async def test_verifier_dir_recreated_before_verifier(self):
        """Full wipe-and-recreate runs as root even when sandbox_user is None."""
        from benchflow._sandbox import harden_before_verify

        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
        task = MagicMock()
        task.config.verifier.env = None

        await harden_before_verify(env, task, sandbox_user=None)

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert any("rm -rf /logs/verifier" in c for c in calls)
        assert any("mkdir -p /logs/verifier" in c for c in calls)

    @pytest.mark.asyncio
    async def test_verifier_dir_chmod_777_after_recreate(self):
        """chmod 777 /logs/verifier appears in the wipe-and-recreate command."""
        from benchflow._sandbox import harden_before_verify

        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
        task = MagicMock()
        task.config.verifier.env = None

        await harden_before_verify(env, task, sandbox_user=None)

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert any("chmod 777 /logs/verifier" in c for c in calls)

    # ── Tier 2: build-config snapshot / restore ──────────────────────────────

    @pytest.mark.asyncio
    async def test_build_config_snapshot_writes_manifest_absent(self):
        """Absent file recorded as false in manifest (not __ABSENT__ string — closes G5)."""
        import json

        from benchflow._sandbox import _snapshot_build_config

        results = iter(
            [
                MagicMock(stdout="", stderr="", exit_code=0),  # mkdir
                *[
                    MagicMock(stdout="absent\n", stderr="", exit_code=0)
                    for _ in range(8)
                ],  # 8 files
                MagicMock(stdout="", stderr="", exit_code=0),  # manifest write
            ]
        )
        env = MagicMock()
        env.exec = AsyncMock(side_effect=results)

        await _snapshot_build_config(env, workspace="/testbed")

        # Find the manifest write call and extract the JSON
        calls = [c.args[0] for c in env.exec.call_args_list]
        manifest_call = next(
            c for c in calls if "_SNAPSHOT_MANIFEST" in c or "manifest.json" in c
        )
        # The JSON is echo'd as a quoted arg — extract it
        import shlex as _shlex

        tokens = _shlex.split(manifest_call)
        json_str = tokens[1]  # echo <json> > path
        manifest = json.loads(json_str)
        assert manifest["setup.py"] is False
        assert "__ABSENT__" not in json_str

    @pytest.mark.asyncio
    async def test_build_config_snapshot_writes_manifest_present(self):
        """Present file recorded as true; cp command was issued."""
        import json

        from benchflow._sandbox import _snapshot_build_config

        # setup.py present, rest absent
        side_effects = [
            MagicMock(stdout="", stderr="", exit_code=0),  # mkdir
            MagicMock(stdout="present\n", stderr="", exit_code=0),  # setup.py
            *[MagicMock(stdout="absent\n", stderr="", exit_code=0) for _ in range(7)],
            MagicMock(stdout="", stderr="", exit_code=0),  # manifest write
        ]
        env = MagicMock()
        env.exec = AsyncMock(side_effect=side_effects)

        await _snapshot_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        # cp ran for setup.py
        assert any("cp --preserve=all /testbed/setup.py" in c for c in calls)
        # manifest records true
        manifest_call = next(c for c in calls if "manifest.json" in c)
        import shlex as _shlex

        json_str = _shlex.split(manifest_call)[1]
        assert json.loads(json_str)["setup.py"] is True

    @pytest.mark.asyncio
    async def test_build_config_restore_removes_absent_file(self):
        """Absent entry in manifest → rm -f issued for destination."""
        import json

        from benchflow._sandbox import _restore_build_config

        manifest = {
            f: False
            for f in (
                "setup.py",
                "pyproject.toml",
                "setup.cfg",
                "tox.ini",
                "noxfile.py",
                "hatch.toml",
                "flit.ini",
                "MANIFEST.in",
            )
        }
        env = MagicMock()
        env.exec = AsyncMock(
            side_effect=[
                MagicMock(
                    stdout=json.dumps(manifest), stderr="", exit_code=0
                ),  # cat manifest
                *[
                    MagicMock(stdout="", stderr="", exit_code=0) for _ in range(8)
                ],  # restores
            ]
        )

        await _restore_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert any("rm -f /testbed/setup.py" in c for c in calls)

    @pytest.mark.asyncio
    async def test_build_config_restore_overwrites_agent_modified_file(self):
        """Present entry in manifest → cp + chown root:root + chmod 644 issued."""
        import json

        from benchflow._sandbox import _restore_build_config

        manifest = {
            f: False
            for f in (
                "setup.py",
                "pyproject.toml",
                "setup.cfg",
                "tox.ini",
                "noxfile.py",
                "hatch.toml",
                "flit.ini",
                "MANIFEST.in",
            )
        }
        manifest["setup.py"] = True
        env = MagicMock()
        env.exec = AsyncMock(
            side_effect=[
                MagicMock(stdout=json.dumps(manifest), stderr="", exit_code=0),
                *[MagicMock(stdout="", stderr="", exit_code=0) for _ in range(8)],
            ]
        )

        await _restore_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        setup_call = next(c for c in calls if "setup.py" in c and "cp" in c)
        assert "chown root:root" in setup_call
        assert "chmod 644" in setup_call

    @pytest.mark.asyncio
    async def test_build_config_sentinel_string_in_file_does_not_trigger_delete(self):
        """setup.py containing literal __ABSENT__ is restored, not deleted (regression for G5)."""
        import json

        from benchflow._sandbox import _restore_build_config

        # Manifest says setup.py was present (True) — regardless of file content
        manifest = {
            f: False
            for f in (
                "setup.py",
                "pyproject.toml",
                "setup.cfg",
                "tox.ini",
                "noxfile.py",
                "hatch.toml",
                "flit.ini",
                "MANIFEST.in",
            )
        }
        manifest["setup.py"] = True
        env = MagicMock()
        env.exec = AsyncMock(
            side_effect=[
                MagicMock(stdout=json.dumps(manifest), stderr="", exit_code=0),
                *[MagicMock(stdout="", stderr="", exit_code=0) for _ in range(8)],
            ]
        )

        await _restore_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        setup_call = next(c for c in calls if "setup.py" in c and c != calls[0])
        # Must issue cp, not rm -f
        assert "cp" in setup_call
        assert "rm -f" not in setup_call

    @pytest.mark.asyncio
    async def test_harden_calls_restore_when_workspace_given(self):
        """harden_before_verify calls restore before CLEANUP_CMD when workspace given."""
        import json

        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        manifest = {
            f: False
            for f in (
                "setup.py",
                "pyproject.toml",
                "setup.cfg",
                "tox.ini",
                "noxfile.py",
                "hatch.toml",
                "flit.ini",
                "MANIFEST.in",
            )
        }

        def side_effect(cmd, **kwargs):
            if f"cat {_SNAPSHOT_MANIFEST}" in cmd:
                return MagicMock(stdout=json.dumps(manifest), stderr="", exit_code=0)
            return MagicMock(stdout="", stderr="", exit_code=0)

        env = MagicMock()
        env.exec = AsyncMock(side_effect=side_effect)
        task = MagicMock()
        task.config.verifier.env = None

        await harden_before_verify(env, task, sandbox_user=None, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        restore_idx = next(
            (i for i, c in enumerate(calls) if _SNAPSHOT_MANIFEST in c), None
        )
        cleanup_idx = next((i for i, c in enumerate(calls) if "conftest.py" in c), None)
        assert restore_idx is not None, "restore (cat manifest) not found"
        assert cleanup_idx is not None
        assert restore_idx < cleanup_idx, "restore must run before CLEANUP_CMD"

    @pytest.mark.asyncio
    async def test_harden_skips_restore_when_workspace_none(self):
        """harden_before_verify with workspace=None issues no restore calls."""
        from benchflow._sandbox import _SNAPSHOT_MANIFEST, harden_before_verify

        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
        task = MagicMock()
        task.config.verifier.env = None

        await harden_before_verify(env, task, sandbox_user=None, workspace=None)

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert not any(_SNAPSHOT_MANIFEST in c for c in calls)

    @pytest.mark.asyncio
    async def test_snapshot_dir_created_chmod_700(self):
        """Snapshot dir creation includes chmod 700 (closes G6)."""
        from benchflow._sandbox import _snapshot_build_config

        side_effects = [
            MagicMock(stdout="", stderr="", exit_code=0),  # mkdir + chmod
            *[MagicMock(stdout="absent\n", stderr="", exit_code=0) for _ in range(8)],
            MagicMock(stdout="", stderr="", exit_code=0),  # manifest write
        ]
        env = MagicMock()
        env.exec = AsyncMock(side_effect=side_effects)

        await _snapshot_build_config(env, workspace="/testbed")

        calls = [c.args[0] for c in env.exec.call_args_list]
        assert any("chmod 700" in c and ".benchflow_build_snapshot" in c for c in calls)

    def test_verifier_env_contract(self):
        """VERIFIER_ENV pins every layer of the pytest ini/plugin hardening.

        Static dict inspection — no async harness needed. See
        tmp/lockdown-sandbox_4.md for the threat model. Each assertion guards a
        specific bypass; collapsing them into one test gives a single
        authoritative contract for the env's contents.
        """
        from benchflow._sandbox import VERIFIER_ENV

        env = VERIFIER_ENV
        addopts = env["PYTEST_ADDOPTS"]

        # Closed-set: any new key added to VERIFIER_ENV must be deliberately
        # accounted for here. Catches accidental additions that could weaken
        # the contract (e.g. a stray debug var with sensitive content).
        assert set(env.keys()) == {
            "PATH",
            "PYTEST_ADDOPTS",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONSAFEPATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
        }

        # Layer 1 — block pyproject.toml/pytest.ini/tox.ini/setup.cfg discovery
        assert "-c /dev/null" in addopts
        # Layer 2 — block conftest.py walk-up beyond /tests
        assert "--confcutdir=/tests" in addopts
        # Pre-existing rootdir pin + cache disable
        assert "--rootdir=/tests" in addopts
        assert "-p no:cacheprovider" in addopts

        # Layer 4 — drop implicit '' (cwd) from sys.path (Python 3.11+)
        assert env["PYTHONSAFEPATH"] == "1"

        # Layer 5 — clear image-ENV carryover (zero-downside insurance)
        assert env["PYTHONSTARTUP"] == ""
        assert env["LD_PRELOAD"] == ""
        assert env["LD_LIBRARY_PATH"] == ""

        # Pattern 7 hardening (pre-existing)
        assert env["PYTHONPATH"] == ""
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert (
            env["PATH"]
            == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )

    def test_plugin_autoload_not_disabled(self):
        """Negative guard: PYTEST_DISABLE_PLUGIN_AUTOLOAD must NOT be in VERIFIER_ENV.

        Disabling plugin autoload would break ~94 SkillsBench tasks that use
        pytest-json-ctrf's --ctrf flag. Entry-point plugin injection is already
        blocked structurally (root verifier + system site-packages perms +
        .pth cleanup in CLEANUP_CMD).

        This guards against accidental re-addition. A developer who *intends*
        to add it will (correctly) update this test and the comment in
        _sandbox.py:VERIFIER_ENV at the same time.
        """
        from benchflow._sandbox import VERIFIER_ENV

        assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" not in VERIFIER_ENV

    def test_pythonhome_not_set(self):
        """Negative guard: PYTHONHOME must NOT be in VERIFIER_ENV.

        PYTHONHOME="" is not equivalent to unset — CPython reads the empty
        prefix and aborts during Py_Initialize, breaking any test.sh that
        spawns a fresh interpreter (e.g. `python -m pip install` before pytest).
        """
        from benchflow._sandbox import VERIFIER_ENV

        assert "PYTHONHOME" not in VERIFIER_ENV

    def test_dash_c_devnull_blocks_hostile_pyproject(self, tmp_path):
        """End-to-end: real pytest under `-c /dev/null` ignores agent-written
        `pyproject.toml` in cwd.

        Binds the static `_VERIFIER_ENV` assertions above to actual pytest
        behavior — if pytest ever changes such that `-c /dev/null` stops
        suppressing ini-file discovery, this catches it. Drops a hostile
        `pyproject.toml` referencing a nonexistent plugin into a tmpdir,
        invokes pytest from that cwd, and asserts both directions:

          1. without `-c /dev/null`: hostile config is loaded → pytest errors
             (sanity check that the test setup is meaningful)
          2. with `-c /dev/null`: hostile config is ignored → pytest succeeds
        """
        import os
        import subprocess
        import sys

        plugin_marker = "benchflow_test_nonexistent_plugin_xyz123"
        (tmp_path / "pyproject.toml").write_text(
            f'[tool.pytest.ini_options]\naddopts = "-p {plugin_marker}"\n'
        )
        (tmp_path / "test_dummy.py").write_text("def test_pass():\n    assert True\n")

        # Whitelist parent env (not blacklist): only pass through what pytest
        # genuinely needs. A blacklist of `PYTEST_*` would still leak
        # PYTHONPATH, VIRTUAL_ENV, CI, TOX_*, etc., any of which can change
        # pytest's behavior and let either branch pass for the wrong reason.
        clean_env = {
            k: os.environ[k]
            for k in ("PATH", "HOME", "LANG", "LC_ALL")
            if k in os.environ
        }

        unhardened = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "test_dummy.py"],
            cwd=tmp_path,
            env=clean_env,
            capture_output=True,
            text=True,
        )
        # Must fail, AND must fail because of OUR hostile plugin marker —
        # not because of an unrelated collection / setup error.
        assert unhardened.returncode != 0, (
            "Sanity check failed: hostile pyproject.toml should crash unhardened pytest. "
            f"stdout: {unhardened.stdout}\nstderr: {unhardened.stderr}"
        )
        combined_unhardened = unhardened.stdout + unhardened.stderr
        assert plugin_marker in combined_unhardened, (
            "Sanity check passed for the wrong reason: hostile plugin marker not in output. "
            f"stdout: {unhardened.stdout}\nstderr: {unhardened.stderr}"
        )

        hardened = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-c",
                "/dev/null",
                "--collect-only",
                "test_dummy.py",
            ],
            cwd=tmp_path,
            env=clean_env,
            capture_output=True,
            text=True,
        )
        # Must succeed, AND must have actually collected the dummy test —
        # not silently collected zero items (which also returns 0 with
        # --collect-only on some pytest versions).
        assert hardened.returncode == 0, (
            "-c /dev/null should block hostile pyproject.toml discovery. "
            f"stdout: {hardened.stdout}\nstderr: {hardened.stderr}"
        )
        assert "test_pass" in hardened.stdout, (
            "Hardened branch returned 0 but did not collect test_pass — "
            "the test may have passed for the wrong reason. "
            f"stdout: {hardened.stdout}\nstderr: {hardened.stderr}"
        )
        # And the hostile plugin marker must NOT appear — proves -c /dev/null
        # actually suppressed pyproject.toml loading.
        assert plugin_marker not in hardened.stdout + hardened.stderr, (
            "-c /dev/null did not suppress hostile pyproject.toml — "
            f"plugin marker {plugin_marker!r} leaked into hardened output."
        )


class TestTrajectorySource:
    """trajectory_source and partial_trajectory fields in RunResult and result.json."""

    def _build(self, tmp_path, **overrides):
        from datetime import datetime

        from benchflow.sdk import SDK

        defaults = dict(
            task_name="t1",
            trial_name="trial-1",
            agent="test",
            agent_name="",
            model="",
            n_tool_calls=0,
            prompts=["x"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            trajectory_source=None,
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
        )
        defaults.update(overrides)
        SDK._build_result(tmp_path, **defaults)
        return json.loads((tmp_path / "result.json").read_text())

    @pytest.mark.parametrize(
        "source,partial,expected_source,expected_partial",
        [
            ("acp", False, "acp", False),
            ("scraped", False, "scraped", False),
            ("partial_acp", True, "partial_acp", True),
            (None, False, None, False),
        ],
    )
    def test_trajectory_source_in_result_json(
        self, tmp_path, source, partial, expected_source, expected_partial
    ):
        data = self._build(
            tmp_path, trajectory_source=source, partial_trajectory=partial
        )
        assert data["trajectory_source"] == expected_source
        assert data["partial_trajectory"] == expected_partial


class TestScrapedTrajectoryTrust:
    """Scraped trajectory must NOT overwrite ACP-sourced n_tool_calls.

    These tests exercise the actual SDK.run() codepath by mocking all
    external dependencies and verifying n_tool_calls is never derived
    from agent-writable data.
    """

    @pytest.fixture
    def sdk_run_mocks(self, tmp_path):
        """Mocks for SDK.run() that reach scraping/finally without real containers."""
        from benchflow.sdk import SDK

        sdk = SDK()

        mock_env = AsyncMock()
        mock_env.exec = AsyncMock(
            return_value=MagicMock(stdout="", stderr="", exit_code=0)
        )
        mock_env.stop = AsyncMock()

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        (task_dir / "task.toml").write_text(
            'version = "1.0"\n[verifier]\ntimeout_sec = 5\n'
            "[agent]\ntimeout_sec = 5\n[environment]\n"
        )
        (task_dir / "environment").mkdir()
        (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:22.04\n")
        (task_dir / "instruction.md").write_text("do the thing")

        return sdk, mock_env, task_dir

    @contextlib.contextmanager
    def _patch_sdk_run(self, sdk, mock_env, extra_patches):
        """Apply shared + extra patches for SDK.run() internals."""
        patches = [
            patch("benchflow.sdk._create_environment", return_value=mock_env),
            patch(
                "benchflow.sdk.install_agent",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    credential_files={},
                    home_dirs=[],
                    skill_paths=[],
                    env_mapping={},
                ),
            ),
            patch("benchflow.sdk.write_credential_files", new_callable=AsyncMock),
            patch("benchflow.sdk.deploy_skills", new_callable=AsyncMock),
            *extra_patches,
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    @pytest.mark.asyncio
    async def test_scraped_trajectory_preserves_n_tool_calls(
        self, sdk_run_mocks, caplog
    ):
        """Main path: forged scraped trajectory must NOT overwrite ACP n_tool_calls."""
        sdk, mock_env, task_dir = sdk_run_mocks

        forged = [{"type": "tool_call", "name": f"fake_{i}"} for i in range(100)]
        mock_session = MagicMock()
        mock_session.tool_calls = [MagicMock() for _ in range(5)]
        mock_acp = AsyncMock()
        mock_acp.session = mock_session
        mock_acp.close = AsyncMock()

        with (
            self._patch_sdk_run(
                sdk,
                mock_env,
                [
                    patch(
                        "benchflow.sdk.connect_acp",
                        new_callable=AsyncMock,
                        return_value=(mock_acp, mock_session, "test-agent"),
                    ),
                    patch(
                        "benchflow.sdk.execute_prompts",
                        new_callable=AsyncMock,
                        return_value=([], 5),
                    ),
                    patch(
                        "benchflow.sdk._scrape_agent_trajectory",
                        new_callable=AsyncMock,
                        return_value=forged,
                    ),
                    patch.object(
                        sdk,
                        "_verify",
                        new_callable=AsyncMock,
                        return_value=({"reward": 1.0}, None),
                    ),
                ],
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = await sdk.run(
                task_dir, agent="test-agent", agent_env={"TEST": "1"}, sandbox_user=None
            )

        assert result.n_tool_calls == 5, (
            "ACP n_tool_calls must survive scraping fallback"
        )
        assert result.trajectory_source == "scraped"
        assert len(result.trajectory) == 100
        assert any("UNTRUSTED" in m for m in caplog.messages)

    @pytest.mark.asyncio
    async def test_partial_acp_uses_session_tool_calls(self, sdk_run_mocks):
        """Finally block: partial_acp path gets n_tool_calls from session, not trajectory."""
        sdk, mock_env, task_dir = sdk_run_mocks

        mock_session = MagicMock()
        mock_session.tool_calls = [MagicMock() for _ in range(3)]
        partial_events = [{"type": "tool_call"}] * 7 + [{"type": "message"}] * 3
        mock_acp = AsyncMock()
        mock_acp.session = mock_session
        mock_acp.close = AsyncMock()

        with self._patch_sdk_run(
            sdk,
            mock_env,
            [
                patch(
                    "benchflow.sdk.connect_acp",
                    new_callable=AsyncMock,
                    return_value=(mock_acp, mock_session, "test-agent"),
                ),
                patch(
                    "benchflow.sdk.execute_prompts",
                    new_callable=AsyncMock,
                    side_effect=ConnectionError("lost"),
                ),
                patch(
                    "benchflow.sdk._capture_session_trajectory",
                    return_value=partial_events,
                ),
                patch(
                    "benchflow.sdk._scrape_agent_trajectory",
                    new_callable=AsyncMock,
                    return_value=[],
                ),
            ],
        ):
            result = await sdk.run(
                task_dir, agent="test-agent", agent_env={"TEST": "1"}, sandbox_user=None
            )

        assert result.n_tool_calls == 3, (
            "Must use session.tool_calls, not trajectory count"
        )
        assert result.trajectory_source == "partial_acp"
        assert result.partial_trajectory is True
