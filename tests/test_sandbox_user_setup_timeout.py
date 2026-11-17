"""The sandbox user setup gets 600 s unless the run (or task) names a time
(the lead's conditions for its 600 s default on this line).

On SWE-bench Pro, handing the workspace to the agent's user took longer than
120 s under load on 10 openlibrary and 4 mlflow tasks (on 0.9.x), unscored.
This line gives the setup the run's named ``--sandbox-setup-timeout``, else
the task's ``sandbox.setup_timeout_sec``, else 600 s. These tests
pin it on every entry point, the unscored ending of a setup timeout, and the
record of the limit.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import benchflow as bf
from benchflow._utils.scoring import INFRA_ERROR, classify_error
from benchflow.evaluation import RetryConfig
from benchflow.rollout import Rollout
from benchflow.sandbox.protocol import ExecTimeoutError
from benchflow.task import Task
from benchflow.task.config import (
    DEFAULT_SANDBOX_USER_SETUP_SEC,
    run_sandbox_setup_timeout,
    sandbox_user_setup_timeout_sec,
)
from tests.test_dogfood_dx_b import _bench, _task


def test_the_default_is_600():
    assert DEFAULT_SANDBOX_USER_SETUP_SEC == 600


def _stub_setup(trial, monkeypatch, setup: AsyncMock) -> AsyncMock:
    for name in (
        "install_agent",
        "write_credential_files",
        "upload_subscription_auth",
        "snapshot_build_config",
        "seed_verifier_workspace",
        "deploy_skills",
        "lockdown_paths",
    ):
        if hasattr(trial._planes, name):
            monkeypatch.setattr(trial._planes, name, AsyncMock())
    monkeypatch.setattr(trial._planes, "setup_sandbox_user", setup)
    return setup


# --- every entry point leaves the run's value unnamed by default ------------


class _Captured(Exception):
    pass


@pytest.fixture
def created(monkeypatch):
    """The RolloutConfig every entry point makes its Rollout from; stopped
    there, before any sandbox."""
    seen: list = []

    def init(self, config, *args, **kwargs):
        seen.append(config)
        raise _Captured("stopped before any sandbox")

    monkeypatch.setattr(Rollout, "__init__", init)
    return seen


def _setup_times(seen) -> set[int]:
    # The task folder _task makes declares no sandbox.setup_timeout_sec.
    return {
        sandbox_user_setup_timeout_sec(run_sandbox_setup_timeout(c), None) for c in seen
    }


@pytest.mark.parametrize(
    ("flag", "expected"), [((), 600), (("--sandbox-setup-timeout", "900"), 900)]
)
def test_bench_run(tmp_path, created, flag, expected):
    task = _task(tmp_path)
    result = _bench(
        "run", str(task), "--agent", "oracle", "--sandbox", "docker",
        "--jobs-dir", str(tmp_path / "jobs"), *flag,
    )  # fmt: skip
    assert created, result.output
    assert _setup_times(created) == {expected}


@pytest.mark.parametrize(
    ("line", "expected"), [("", 600), ("sandbox_setup_timeout: 900\n", 900)]
)
def test_bench_run_config(tmp_path, created, line, expected):
    task = _task(tmp_path)
    config = tmp_path / "run.yaml"
    config.write_text(
        f"tasks_dir: {task.parent}\nagent: oracle\nenvironment: docker\n"
        f"jobs_dir: {tmp_path / 'jobs'}\n{line}"
    )
    result = _bench("run", "--config", str(config))
    assert created, result.output
    assert _setup_times(created) == {expected}


def test_bench_branch(tmp_path, created):
    task = _task(tmp_path)
    result = _bench(
        "branch", "--tasks-dir", str(task), "--agent", "oracle", "--sandbox", "docker",
        "--child", "label=a", "--child", "label=b", "--jobs-dir", str(tmp_path / "jobs"),
    )  # fmt: skip
    assert created, result.output
    assert _setup_times(created) == {600}


@pytest.mark.asyncio
async def test_the_sdk(tmp_path, created):
    task = _task(tmp_path)
    with pytest.raises(_Captured):
        await bf.run(bf.RolloutConfig(task_path=task, agent="oracle"))
    assert created
    assert _setup_times(created) == {600}


# --- a setup timeout ends unscored with its reason, and is retried as infra ---


@pytest.mark.asyncio
async def test_a_setup_timeout_ends_unscored_with_its_reason(tmp_path, monkeypatch):
    """Never a 0: no verifier, no rewards, the timeout's text as the error.
    It is an infrastructure error, which the job retries by default."""
    task = _task(tmp_path)
    rollout = Rollout(
        bf.RolloutConfig(task_path=task, agent="oracle", sandbox_user="agent")
    )
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._rollout_name = "setup-timeout"
    rollout._task = Task(task)
    rollout._started_at = datetime.now()
    rollout._agent_cwd = "/app"
    rollout._timeout = 60
    rollout._env = MagicMock()
    rollout._env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
    for name in ("setup", "start", "cleanup"):
        setattr(rollout, name, AsyncMock())
    rollout.verify = AsyncMock()
    _stub_setup(
        rollout,
        monkeypatch,
        AsyncMock(side_effect=ExecTimeoutError("Command timed out after 600 seconds")),
    )
    result = await rollout.run()
    assert "Command timed out after 600 seconds" in (result.error or "")
    assert result.rewards is None
    rollout.verify.assert_not_awaited()
    assert classify_error(result.error) == INFRA_ERROR
    assert RetryConfig().should_retry(result.error) is True


# --- the trial record names the limit the setup had -------------------------


def test_config_json_records_the_user_setup_limit():
    from benchflow.rollout._results import _write_config

    params = inspect.signature(_write_config).parameters
    assert "sandbox_user_setup_timeout" in params
    source = inspect.getsource(Rollout.setup)
    # The rollout records what it gives the setup, not the run's raw value.
    assert "sandbox_user_setup_timeout=sandbox_user_setup_timeout_sec(" in source


@pytest.mark.integration
@pytest.mark.parametrize(
    ("flag", "expected"), [((), 600), (("--sandbox-setup-timeout", "900"), 900)]
)
def test_a_real_trial_records_the_limit_in_config_json(tmp_path, flag, expected):
    """A real oracle trial on Docker: config.json holds the run's value and
    the limit the sandbox user setup had."""
    import shutil
    import subprocess

    if (
        not shutil.which("docker")
        or subprocess.run(
            ["docker", "info"], capture_output=True, check=False
        ).returncode
    ):
        pytest.skip("Docker is unavailable")
    task = Path(__file__).parent / "examples" / "hello-world-task"
    result = _bench(
        "run", str(task), "--agent", "oracle", "--sandbox", "docker",
        "--jobs-dir", str(tmp_path / "jobs"), *flag,
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    (config,) = (tmp_path / "jobs").rglob("config.json")
    recorded = json.loads(config.read_text())
    assert recorded["sandbox_user_setup_timeout"] == expected
    assert recorded["sandbox_setup_timeout"] == (int(flag[1]) if flag else 120)
