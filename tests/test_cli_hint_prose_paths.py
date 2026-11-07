"""Guard CONF8's builder sweep against unquoted prose paths at e9bcbd89."""

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console

from benchflow.cli import _shared as shared
from tests.test_cli_hint_paths import _argv

PATHS = [
    "run  a",
    "[bold]x[/bold]",
    "a'b",
    "x; printf D2B_HINT;#",
    "x\ny",
    "x\ty",
    "x\u2028y",
]


def _check_word(word, path, placeholder):
    if str(path).isprintable():
        assert _argv("bench " + word) == [str(path)]
    else:
        assert word == placeholder


@pytest.mark.parametrize("folder", PATHS)
@pytest.mark.parametrize("mode", ["finished", "manifest", "matrix-manifest"])
def test_artifact_recovery_paths_are_literals(tmp_path, monkeypatch, folder, mode):
    """Guards CONF8 sibling: controls must not turn a prose path into advice."""
    from benchflow import eval_artifacts
    from benchflow.cli import eval_artifacts as cli
    from benchflow.evaluation import EvaluationConfig

    path = Path(folder)
    output = io.StringIO()
    monkeypatch.setattr(shared, "err_console", Console(file=output, width=40))
    if mode == "finished":
        shared.print_error("D2B_DUMMY", next_step=cli._finished_next(path))
        prefix = "fix the cause above; the trials finished and are kept in "
        suffix = ""
        placeholder = "<job dir>"
    else:
        monkeypatch.setattr(
            eval_artifacts, "write_task_manifest", lambda *a: {"total": 0}
        )
        monkeypatch.setattr(cli, "_apply_source_sidecar", lambda config, path: config)
        plan = SimpleNamespace(
            request=SimpleNamespace(task_manifest_out=path, expected_tasks=1)
        )
        with pytest.raises(typer.Exit):
            if mode == "manifest":
                cli.postprocess_eval_artifacts(
                    plan, tmp_path, EvaluationConfig(), Path("job")
                )
            else:
                cli._write_matrix_task_manifest(plan, tmp_path, EvaluationConfig())
        prefix = ""
        suffix = (
            " lists the selected tasks; fix --include/--exclude or --expected-tasks"
        )
        placeholder = "<manifest file>"
    line = next(
        line.strip().removeprefix("next: ")
        for line in output.getvalue().splitlines()
        if line.strip().startswith("next: ")
    )
    assert line.startswith(prefix) and line.endswith(suffix)
    word = line[len(prefix) : len(line) - len(suffix) if suffix else None]
    _check_word(word, path, placeholder)


@pytest.mark.parametrize("folder", PATHS)
@pytest.mark.parametrize("mode", ["add", "reorder", "replace"])
def test_doctor_path_advice_is_literal(tmp_path, monkeypatch, folder, mode):
    """Guards CONF8 sibling in all PATH advice builders and their Rich sink."""
    from benchflow.cli.doctor import _print_checks
    from tests.test_doctor_bench_path import Machine, _line

    machine = Machine(tmp_path)
    chosen = tmp_path / folder / "bin"
    first = tmp_path / folder / "other"
    if mode == "add":
        check = _line(machine, [], entry=chosen / "bench")
        prefix, suffix = (
            "Add ",
            " to PATH (for a uv tool install: uv tool update-shell)",
        )
        expected = [(chosen, "<bin folder>")]
    else:
        machine.add_rust(first if mode == "reorder" else chosen)
        check = _line(
            machine,
            [first if mode == "reorder" else chosen],
            entry=chosen / "benchflow",
        )
        prefix = "benchflow always runs this install; for bench to run it too, "
        suffix = ": https://www.benchflow.ai/docs/benchflow/getting-started/install#which-bench-runs"
        if mode == "reorder":
            prefix += "put "
            suffix = " on PATH" + suffix
            expected = [(chosen, "<bin folder>"), (first, "<bin folder>")]
        else:
            # The user's own install command (FF-install).
            from benchflow import __version__

            prefix += (
                "run the install command you installed it with again, with "
                f"--force (benchflow=={__version__} and the same extras and "
                "constraints), which replaces "
            )
            expected = [(chosen / "bench", "<bench executable>")]
    output = io.StringIO()
    monkeypatch.setattr(shared, "err_console", Console(file=output, width=40))
    _print_checks([check])
    line = next(
        line.strip().removeprefix("fix: ")
        for line in output.getvalue().splitlines()
        if line.strip().startswith("fix: ")
    )
    assert line.startswith(prefix) and line.endswith(suffix)
    words = line[len(prefix) : -len(suffix)].split(" before ")
    assert len(words) == len(expected)
    for word, (path, placeholder) in zip(words, expected, strict=True):
        _check_word(word, path, placeholder)
