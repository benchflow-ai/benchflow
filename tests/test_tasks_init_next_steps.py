"""`bench tasks init` says what to do after scaffolding.

Guards dx/first-run: the scaffold fails on purpose until its [REPLACE: ...]
placeholders are filled, and in the 2026-09-30 first-run walk nothing said
so or named the next command.
"""

from __future__ import annotations

from typer.testing import CliRunner

from benchflow.cli.main import app


def test_tasks_init_names_the_placeholders_and_the_next_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(app, ["tasks", "init", "demo"], terminal_width=200)

    assert result.exit_code == 0, result.output
    assert "Next: replace every [REPLACE: ...] placeholder" in result.output
    assert "\nbench tasks check tasks/demo\n" in result.output
    assert (
        "\nbench eval run --tasks-dir tasks/demo --agent oracle --sandbox docker "
        "--jobs-dir jobs/demo-oracle\n" in result.output
    )


def test_tasks_init_quotes_a_path_with_spaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["tasks", "init", "demo", "--dir", "my tasks"], terminal_width=200
    )

    assert result.exit_code == 0, result.output
    assert "\nbench tasks check 'my tasks/demo'\n" in result.output
