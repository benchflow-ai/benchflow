"""The sandbox user setup gets 600 s unless the run names a value (EN-726).

On SWE-bench Pro, handing the workspace to the agent's user (cp -a of root's
home folders, chown -R of the home and the workspace) took longer than the
run's 120 s default under load on 10 openlibrary and 4 mlflow tasks:
ExecTimeoutError, unscored. ``--sandbox-setup-timeout 900`` scored them. The
setup now gets 600 s when the run names no ``--sandbox-setup-timeout``, as on
the rc line (EN-278); a named value passes through as given, 120 included
(``sandbox_setup_timeout_given``, the rc line's rule).
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
from benchflow.evaluation import EvaluationConfig, RetryConfig
from benchflow.rollout import Rollout
from benchflow.sandbox.protocol import ExecTimeoutError
from benchflow.task import Task
from benchflow.task.config import (
    DEFAULT_SANDBOX_USER_SETUP_SEC,
    RUN_SANDBOX_SETUP_TIMEOUT_SEC,
    run_sandbox_setup_timeout,
    sandbox_user_setup_timeout_sec,
)
from tests.test_dogfood_dx_b import _bench, _task
from tests.test_trial_install_agent_timeout import _make_trial


@pytest.mark.parametrize(
    ("run", "expected"),
    [(None, 600), (120, 120), (60, 60), (900, 900), (600, 600)],
)
def test_the_runs_named_value_passes_through_else_600(run, expected):
    assert DEFAULT_SANDBOX_USER_SETUP_SEC == 600
    assert RUN_SANDBOX_SETUP_TIMEOUT_SEC == 120
    assert sandbox_user_setup_timeout_sec(run) == expected


@pytest.mark.parametrize(
    ("value", "given", "named", "setup"),
    [
        (120, None, None, 600),  # a Python config that does not say: the default
        (120, False, None, 600),
        (120, True, 120, 120),  # an explicit 120 is named
        (900, None, 900, 900),
        (60, True, 60, 60),
    ],
)
def test_a_config_says_whether_the_run_named_it(tmp_path, value, given, named, setup):
    task = _task(tmp_path)
    kwargs = {} if given is None else {"sandbox_setup_timeout_given": given}
    for config in (
        bf.RolloutConfig(
            task_path=task, agent="oracle", sandbox_setup_timeout=value, **kwargs
        ),
        EvaluationConfig(sandbox_setup_timeout=value, **kwargs),
    ):
        assert config.sandbox_setup_timeout == value
        assert config.sandbox_setup_timeout_given is (named is not None)
        assert run_sandbox_setup_timeout(config) == named
        assert (
            sandbox_user_setup_timeout_sec(run_sandbox_setup_timeout(config)) == setup
        )


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


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["oracle", "claude-agent-acp"])
@pytest.mark.parametrize(
    ("run", "named", "expected"),
    [(120, None, 600), (120, True, 120), (60, None, 60), (900, None, 900)],
)
async def test_the_rollout_gives_the_setup_the_resolved_time(
    tmp_path, monkeypatch, agent, run, named, expected
):
    trial = _make_trial(tmp_path, agent=agent, sandbox_setup_timeout=run)
    if named is not None:  # the run named it (a flag or a config key)
        trial._config.sandbox_setup_timeout_given = named
    setup = _stub_setup(trial, monkeypatch, AsyncMock(return_value="/home/agent"))
    await trial.install_agent()
    setup.assert_awaited_once()
    assert setup.await_args.kwargs["timeout_sec"] == expected
    # The run's own setting is unchanged: the agent install still gets it.
    assert trial._config.sandbox_setup_timeout == run


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
    return {sandbox_user_setup_timeout_sec(run_sandbox_setup_timeout(c)) for c in seen}


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ((), 600),
        (("--sandbox-setup-timeout", "120"), 120),  # named: as given
        (("--sandbox-setup-timeout", "900"), 900),
    ],
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
    ("line", "expected"),
    [
        ("", 600),
        ("sandbox_setup_timeout: 120\n", 120),
        ("sandbox_setup_timeout: 900\n", 900),
    ],
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


def test_the_flag_over_a_config_file_is_named(tmp_path, created):
    task = _task(tmp_path)
    config = tmp_path / "run.yaml"
    config.write_text(
        f"tasks_dir: {task.parent}\nagent: oracle\nenvironment: docker\n"
        f"jobs_dir: {tmp_path / 'jobs'}\n"
    )
    result = _bench(
        "run", "--config", str(config), "--sandbox-setup-timeout", "120"
    )  # fmt: skip
    assert created, result.output
    assert _setup_times(created) == {120}


@pytest.mark.parametrize(("value", "given"), [(120, True), (120, False), (900, True)])
def test_the_run_config_record_keeps_whether_it_was_named(tmp_path, value, given):
    """rollout_config_to_dict writes the value only when the run named it, so
    a default read back stays unnamed (600), and a named 120 stays 120; a
    shard's config carries the flag to its worker."""
    from benchflow._utils.yaml_loader import (
        rollout_config_from_dict,
        rollout_config_to_dict,
    )
    from benchflow.eval_worker import _evaluation_config

    task = _task(tmp_path)
    config = bf.RolloutConfig(
        task_path=task,
        agent="oracle",
        sandbox_setup_timeout=value,
        sandbox_setup_timeout_given=given,
    )
    raw = rollout_config_to_dict(config)
    assert ("sandbox_setup_timeout" in raw) is given
    back = rollout_config_from_dict(raw, task_path=task)
    assert run_sandbox_setup_timeout(back) == (value if given else None)
    shard = _evaluation_config(
        {"sandbox_setup_timeout": value, "sandbox_setup_timeout_given": given}
    )
    assert run_sandbox_setup_timeout(shard) == (value if given else None)


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
    ("flag", "expected"),
    [
        ((), 600),
        (("--sandbox-setup-timeout", "120"), 120),
        (("--sandbox-setup-timeout", "900"), 900),
    ],
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
    assert recorded["sandbox_setup_timeout_given"] is bool(flag)
