"""Regression: the `continue` / `continue-batch` commands must stay registered.

The RFC diff deleted the `register_continue(app)` call from cli/main.py, which
made these commands unreachable even though continue_cmd.py still defines them.
"""

from typer.testing import CliRunner

from benchflow.cli.main import app


def _command_names() -> set[str]:
    return {cmd.name for cmd in app.registered_commands}


def test_continue_command_is_registered() -> None:
    assert "continue" in _command_names()


def test_continue_batch_command_is_registered() -> None:
    assert "continue-batch" in _command_names()


def test_continue_listed_in_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "continue" in result.output
