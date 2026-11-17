"""The tables printed at their natural width (`bench results <jobs folder>`'s
listing, `bench eval metrics`) go through the diagnostics redactor: a secret
the CLI knows never shows, wherever it was planted in what they print (a job
folder's name, a setting label, a cause key, a summary.json count), at any
console width. And summary.json's raw fields print only as numbers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.test_headline_score_unscored import _job

# Credential-shaped values the CLI knows as secrets (their variables' names).
ANTHROPIC = "sk-ant-api03-" + "Q7xR2m" * 7
AWS = "AKIA" + "Z3K9WQ7T" * 2
SECRETS = {"ANTHROPIC_API_KEY": ANTHROPIC, "AWS_SECRET_ACCESS_KEY": AWS}


def _planted(tmp_path: Path) -> Path:
    """A job with an unscored trial (so the note, and the sink, are on), a
    secret in its folder's name, in every trial's model (the setting label)
    and in an integration failure's cause key."""
    job, _ = _job(tmp_path, refused=True)
    for result in sorted(job.glob("*/result.json")):
        record = json.loads(result.read_text())
        record["model"] = f"m-{AWS}"
        if record.get("rewards") is None:
            record["error_category"] = "agent_integration"
            record["integration_failure_info"] = {"cause": f"cause-{ANTHROPIC}"}
        result.write_text(json.dumps(record))
        config = result.parent / "config.json"
        data = json.loads(config.read_text()) if config.exists() else {}
        config.write_text(json.dumps({**data, "model": f"m-{AWS}"}))
    return job.rename(job.parent / f"run-{ANTHROPIC}")


@pytest.mark.parametrize("columns", ["40", "80"])
@pytest.mark.parametrize("command", ["results-listing", "eval-metrics"])
def test_no_planted_secret_shows(tmp_path, monkeypatch, columns, command):
    from benchflow.cli.main import app

    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    job = _planted(tmp_path)
    args = (
        ["results", str(job.parent)]
        if command == "results-listing"
        else [
            "eval",
            "metrics",
            str(job),
        ]
    )
    out = CliRunner().invoke(app, args, env={"COLUMNS": columns, **SECRETS})
    assert out.exit_code == 0, out.output
    for value in SECRETS.values():
        assert value not in out.output, out.output
    assert "***" in out.output, out.output


def test_summary_fields_print_only_as_numbers(tmp_path):
    from benchflow.cli.main import app

    job, _ = _job(tmp_path, refused=False)
    summary = json.loads((job / "summary.json").read_text())
    summary.pop("scoring", None)
    summary.update(passed=f"[red]{ANTHROPIC}[/red]", total="x" * 30, score="1/0 [b]")
    (job / "summary.json").write_text(json.dumps(summary))
    out = CliRunner().invoke(app, ["results", str(job.parent)], terminal_width=200)
    assert out.exit_code == 0, out.output
    assert ANTHROPIC not in out.output and "x" * 30 not in out.output, out.output
    assert "?/? (?)" in out.output, out.output
