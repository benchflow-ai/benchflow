"""Guards Muse role validation at PY8 base 363d815b."""

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from tests.test_check_run_plan import _task


@pytest.mark.parametrize("command", [["check"], ["tasks", "check"]])
@pytest.mark.parametrize("agent", ["muse", "muse-code"])
@pytest.mark.parametrize("user", ["root", "null", "none", "NULL"])
def test_check_applies_muse_role_validator(tmp_path, command, agent, user):
    """Guards 363d815b: Muse root plans fail before login discovery or setup."""
    task = _task(tmp_path / "task")
    result = CliRunner().invoke(
        app, [*command, str(task), "--agent", agent, "--sandbox-user", user]
    )
    assert result.exit_code == 1, result.output
    assert "muse-code requires a non-root sandbox_user" in result.stderr
