"""Guards task environment preflight at PY8 base 363d815b."""

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.task.plan_validation import validate_task_plan
from tests.test_check_run_plan import _task


@pytest.fixture(autouse=True)
def allow_fixture_input(monkeypatch):
    """Keep the 363d815b availability contract after S1 e4db9aa7 admission."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", "PY8_MISSING_DATA")


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize(
    "section", ["sandbox.env", "verifier.env", "solution.env", "setup"]
)
def test_active_env_templates_are_resolved_before_any_sandbox(
    tmp_path, monkeypatch, backend, section
):
    """Guards 363d815b: every runtime-resolved active map gets the same gate."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    monkeypatch.delenv("PY8_MISSING_DATA", raising=False)
    config = (
        f'[{section}]\nDATA_ROOT = "${{PY8_MISSING_DATA}}"\n'
        if section != "setup"
        else '[[sandbox.setup_commands]]\ncommand = "true"\n'
        '[sandbox.setup_commands.env]\nDATA_ROOT = "${PY8_MISSING_DATA}"\n'
    )
    task = _task(tmp_path / "task", config)
    with pytest.raises(ValueError, match=r"PY8_MISSING_DATA.*not found"):
        validate_task_plan(task, sandbox=backend, agent="oracle")


@pytest.mark.parametrize("command", [["check"], ["tasks", "check"]])
def test_check_refuses_missing_verifier_env(tmp_path, monkeypatch, command):
    """Guards 363d815b: both CLI check spellings surface the runtime error."""
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    monkeypatch.delenv("PY8_MISSING_DATA", raising=False)
    task = _task(tmp_path / "task", '[verifier.env]\nDATA_ROOT="${PY8_MISSING_DATA}"\n')
    result = CliRunner().invoke(app, [*command, str(task), "--agent", "oracle"])
    assert result.exit_code == 1, result.output
    assert "PY8_MISSING_DATA" in result.stderr


def test_inactive_solution_env_does_not_refuse_an_agent_run(tmp_path, monkeypatch):
    """Control for 363d815b: agent runs do not execute the oracle's env."""
    task = _task(tmp_path / "task", '[solution.env]\nDATA_ROOT="${PY8_MISSING_DATA}"\n')
    monkeypatch.delenv("PY8_MISSING_DATA", raising=False)
    validate_task_plan(task, sandbox="docker", agent="nop")
