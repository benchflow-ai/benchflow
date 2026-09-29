"""`bench --help` and `bench eval --help` list commands in the order the docs use.

Regression test: the first-run path in
the README and getting-started is doctor, then smoke, then run, but `bench
--help` listed `review` before `doctor` and `bench eval --help` listed `smoke`
last. The `tasks` group's help also lacked its final period.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from benchflow.cli.main import app


def _command_names(help_text: str) -> list[str]:
    """Command names from the help panels (option rows start with a dash)."""
    return re.findall(r"^│ ([a-z][\w-]*) ", help_text, flags=re.MULTILINE)


def _help(*args: str) -> str:
    result = CliRunner().invoke(app, [*args, "--help"], env={"COLUMNS": "120"})
    assert result.exit_code == 0, result.output
    return result.output


def test_top_level_help_lists_doctor_first():
    names = _command_names(_help())
    assert names[0] == "doctor"


def test_eval_help_lists_smoke_then_run_first():
    names = _command_names(_help("eval"))
    assert names[:2] == ["smoke", "run"]


def test_tasks_group_help_ends_with_a_period():
    assert "Task authoring commands." in _help()
