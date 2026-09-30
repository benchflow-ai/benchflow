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
        "--jobs-dir jobs/demo-oracle --fresh\n" in result.output
    )


def test_tasks_init_run_line_is_fresh_so_an_edit_is_not_read_from_a_cached_job():
    """The printed run command is the authoring loop, so it must not resume.

    A plain run resumes its own last job, and a task whose name already has a
    result is reported from that result rather than run again, so an author
    who fixes `oracle/solve.sh` and re-runs the printed line would read the
    verdict of the code they just changed.
    """
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["tasks", "init", "demo"], terminal_width=200)

        assert result.exit_code == 0, result.output
        run_line = next(
            line
            for line in result.output.splitlines()
            if line.startswith("bench eval run ")
        )
        assert "--fresh" in run_line


def test_tasks_init_quotes_a_path_with_spaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        app, ["tasks", "init", "demo", "--dir", "my tasks"], terminal_width=200
    )

    assert result.exit_code == 0, result.output
    assert "\nbench tasks check 'my tasks/demo'\n" in result.output
    # The run line is where quoting is actually hard: two quoted values, and
    # --jobs-dir goes through its own shlex.quote call.
    assert (
        "\nbench eval run --tasks-dir 'my tasks/demo' --agent oracle "
        "--sandbox docker --jobs-dir jobs/demo-oracle --fresh\n" in result.output
    )
