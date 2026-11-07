"""Guards FF-sw-cli-other-C3 against eager env resolution at f29e8174."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.rollout import _run_environment_setup_commands
from benchflow.task import Task
from benchflow.task.env import resolve_env_vars
from benchflow.task.plan_validation import validate_task_plan
from tests.test_check_run_plan import _task

CAPTURE = '[[sandbox.setup_commands]]\ncommand="true"\ncapture_dir="/tmp/capture"\ncapture_dir_b64_env="PY8_CAPTURED_B64"\n'
CONSUMER = '[[sandbox.setup_commands]]\ncommand="true"\n[sandbox.setup_commands.env]\nPAYLOAD="${PY8_CAPTURED_B64}"\n'


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    # S1 e4db9aa7 requires an explicit grant for env templates. Keep these
    # capture-order tests about availability after that independent grant.
    monkeypatch.setenv(
        "BENCHFLOW_ALLOW_TASK_ENV", "PY8_CAPTURED_B64,PY8_REALLY_MISSING"
    )
    monkeypatch.delenv("PY8_CAPTURED_B64", raising=False)


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize("consumer", ["setup", "verifier", "solution"])
def test_preflight_recognizes_prior_capture(tmp_path, backend, consumer):
    """Guards f29e8174: produced values become available after their command."""
    config = CAPTURE + (
        CONSUMER
        if consumer == "setup"
        else f'[{consumer}.env]\nPAYLOAD="${{PY8_CAPTURED_B64}}"\n'
    )
    task = _task(tmp_path / "task", config)
    validate_task_plan(task, sandbox=backend, agent="oracle")
    assert "PY8_CAPTURED_B64" not in os.environ


@pytest.mark.parametrize("command", [["check"], ["tasks", "check"]])
def test_check_accepts_capture_dependency(tmp_path, command):
    """Guards f29e8174 on both supported check spellings."""
    task = _task(tmp_path / "task", CAPTURE + CONSUMER)
    result = CliRunner().invoke(app, [*command, str(task), "--agent", "oracle"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "config",
    [
        CONSUMER + CAPTURE,
        CAPTURE + '[sandbox.setup_commands.env]\nPAYLOAD="${PY8_CAPTURED_B64}"\n',
        '[sandbox.env]\nPAYLOAD="${PY8_CAPTURED_B64}"\n' + CAPTURE,
        CAPTURE + '[verifier.env]\nPAYLOAD="${PY8_REALLY_MISSING}"\n',
    ],
)
def test_capture_does_not_satisfy_earlier_or_unrelated_dependencies(
    tmp_path, monkeypatch, config
):
    """Guards f29e8174: forward/self/early and genuinely missing inputs fail."""
    monkeypatch.delenv("PY8_REALLY_MISSING", raising=False)
    task = _task(tmp_path / "task", config)
    with pytest.raises(ValueError, match="not found"):
        validate_task_plan(task, sandbox="docker", agent="oracle")


@pytest.mark.asyncio
async def test_planned_capture_is_produced_for_runtime_consumers(tmp_path, monkeypatch):
    """Guards f29e8174: planned availability matches the actual capture boundary."""
    task_dir = _task(
        tmp_path / "task",
        CAPTURE
        + CONSUMER
        + '[verifier.env]\nPAYLOAD="${PY8_CAPTURED_B64}"\n[solution.env]\nPAYLOAD="${PY8_CAPTURED_B64}"\n',
    )
    task = Task(task_dir)
    validate_task_plan(task_dir, sandbox="docker", agent="oracle", task=task)

    async def download(source, destination, **kwargs):
        destination.mkdir()
        (destination / "dummy.txt").write_text("dummy marker")

    env = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0)),
        download_dir=download,
    )
    # Record an undo for the direct os.environ write performed by capture.
    monkeypatch.setenv("PY8_CAPTURED_B64", "")
    monkeypatch.delenv("PY8_CAPTURED_B64")
    await _run_environment_setup_commands(env, task)
    payload = os.environ["PY8_CAPTURED_B64"]
    assert payload
    assert env.exec.await_args_list[1].kwargs["env"] == {"PAYLOAD": payload}
    assert resolve_env_vars(task.config.verifier.env) == {"PAYLOAD": payload}
    assert resolve_env_vars(task.config.solution.env) == {"PAYLOAD": payload}
