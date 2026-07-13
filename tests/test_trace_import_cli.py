from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.traces.models import ParsedTrace, ToolCall, TraceStep


def test_tasks_generate_help_uses_long_options_only() -> None:
    """Guards ENG-96: task-generation help stays on long options."""
    result = CliRunner().invoke(app, ["tasks", "generate", "--help"])

    assert result.exit_code == 0
    output = click.unstyle(result.output)
    help_tokens = {token for line in output.splitlines() for token in line.split()}
    assert "-o" not in help_tokens
    assert "-p" not in help_tokens
    assert "-f" not in help_tokens
    assert "-n" not in help_tokens
    assert "--output" in output
    assert "--project" in output
    assert "--format" in output
    assert "--limit" in output


@pytest.mark.parametrize(
    ("alias", "value"),
    [
        ("-o", "generated-tasks"),
        ("-p", "benchflow"),
        ("-f", "auto"),
        ("-n", "1"),
    ],
)
def test_tasks_generate_rejects_removed_short_options(
    alias: str, value: str, tmp_path: Path
) -> None:
    """Guards ENG-96: removed task-generation short options stay rejected."""
    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "generate",
            "--from-file",
            str(tmp_path / "traces.jsonl"),
            alias,
            value,
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    output = click.unstyle(result.output)
    assert "No such option" in output
    assert alias in output


def test_tasks_generate_dry_run_uses_generation_filters(
    monkeypatch, tmp_path: Path
) -> None:
    valid_trace = ParsedTrace(
        trace_id="valid-trace",
        session_id="valid",
        steps=[
            TraceStep(role="user", content="Create good.py"),
            TraceStep(
                role="assistant",
                content="Editing good.py",
                tool_calls=[ToolCall(name="Edit", input={"file_path": "good.py"})],
            ),
        ],
        outcome="success",
    )
    no_file_trace = ParsedTrace(
        trace_id="bash-only",
        session_id="bash",
        steps=[
            TraceStep(role="user", content="Run tests"),
            TraceStep(
                role="assistant",
                content="Running tests",
                tool_calls=[ToolCall(name="Bash", input={"command": "pytest"})],
            ),
        ],
        outcome="success",
    )

    def fake_load_file(path: Path, format: str) -> list[ParsedTrace]:
        _ = (path, format)
        return [valid_trace, no_file_trace]

    monkeypatch.setattr("benchflow.cli.trace_import._load_file", fake_load_file)

    result = CliRunner().invoke(
        app,
        [
            "tasks",
            "generate",
            "--from-file",
            str(tmp_path / "traces.jsonl"),
            "--dry-run",
            "--min-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Skipped 1 trace(s)" in result.output
    assert "Parsed Traces (1)" in result.output
    assert "bash-only" not in result.output
