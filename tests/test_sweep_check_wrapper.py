"""Guards supported task collection wrappers at PY8 base 363d815b."""

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from tests.test_check_run_plan import _task


@pytest.mark.parametrize("command", [["check"], ["tasks", "check"]])
@pytest.mark.parametrize("wrapped", [False, True])
def test_check_discovers_every_task_in_a_snapshot(tmp_path, command, wrapped):
    """Guards 363d815b: check and eval share collection-root resolution."""
    root = tmp_path / "snapshot"
    parent = root / "tasks" if wrapped else root
    for name in ("task-a", "task-b"):
        _task(parent / name)
    result = CliRunner().invoke(app, [*command, str(root), "--agent", "oracle"])
    assert result.exit_code == 0, result.output
    assert "task-a" in result.output and "task-b" in result.output
