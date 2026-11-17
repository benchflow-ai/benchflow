"""(a), the lead's condition: a headline score that counts trials
BenchFlow did not score as 0 (benchflow.score.v1) says so on its own line,
beside the number: ``33.3% (counts 1 unscored trial as 0; scored only:
50.0%)``, so the number is never copied without it. A job whose every trial
was scored prints the plain number. summary.json is unchanged.

The job here: one passed, one failed and one trial BenchFlow's helper
refused (sandbox_setup, no reward)."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import benchflow.cli._shared as shared
from benchflow._utils.scoring import score_summary_fields, unscored_note
from benchflow.evaluation import EvaluationConfig, EvaluationResult
from benchflow.models import RolloutResult
from tests.test_python_sdk_load_job import _trial

HELPER = (
    "RuntimeError: Failed to enforce sandbox-user egress firewall: BenchFlow's "
    "helper could not add the rules to main's namespace (status none)"
)
NOTE = "counts 1 unscored trial as 0; scored only: 50.0%"


def _job(root: Path, *, refused: bool) -> tuple[Path, EvaluationResult]:
    """The job's folder (its trials and summary.json, as a run writes them)
    and its EvaluationResult."""
    job = root / "jobs" / "job"
    _trial(job, "passed", reward=1.0)
    _trial(job, "failed", reward=0.0)
    if refused:
        _trial(
            job, "refused", reward=None, error=HELPER, error_category="sandbox_setup"
        )
    rows = [json.loads(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    (job / "summary.json").write_text(json.dumps(score_summary_fields(rows)))
    results = {
        row["task_name"]: RolloutResult.from_dict(
            row, rollout_dir=job / row["rollout_name"]
        )
        for row in rows
    }
    return job, EvaluationResult(
        job_name="job",
        config=EvaluationConfig(),
        total=len(rows),
        passed=1,
        failed=1,
        errored=int(refused),
        elapsed_sec=1.0,
        job_dir=job,
        results=results,
    )


def _line(text: str, pattern: str) -> str:
    """The one line of *text* matching *pattern*, box drawing removed."""
    found = [
        " ".join(re.sub("[─-╿]", " ", line).split())
        for line in text.splitlines()
        if re.search(pattern, line)
    ]
    assert len(found) == 1, (pattern, text)
    return found[0]


def test_the_note_names_both_counts():
    assert unscored_note(2, 0.5) == "counts 2 unscored trials as 0; scored only: 50.0%"
    assert unscored_note(1, None) == (
        "counts 1 unscored trial as 0; scored only: none scored"
    )
    assert unscored_note(0, 0.5) == ""


@pytest.mark.parametrize("refused", [True, False], ids=["refused", "all-scored"])
def test_bench_eval_runs_score_line(tmp_path, monkeypatch, refused):
    """The Score line `bench eval run` prints, on one line even on a narrow
    terminal."""
    job, result = _job(tmp_path, refused=refused)
    console = Console(file=io.StringIO(), width=60)
    monkeypatch.setattr(shared, "console", console)
    shared._report_eval_result(result, job)
    line = _line(console.file.getvalue(), r"Score: \d")
    if refused:
        assert f"Score: 1/3 (33.3%; {NOTE})" in line, line
    else:
        assert "Score: 1/2 (50.0%)" in line, line
        assert "unscored" not in line, line


@pytest.mark.parametrize("refused", [True, False], ids=["refused", "all-scored"])
def test_bench_eval_metrics_score_row(tmp_path, refused):
    """`bench eval metrics`' Score row; its --json is summary.json's figures,
    with no note."""
    from benchflow.cli.main import app

    job, _ = _job(tmp_path, refused=refused)
    out = CliRunner().invoke(app, ["eval", "metrics", str(job)], terminal_width=200)
    assert out.exit_code == 0, out.output
    line = _line(out.output, r"^\W*Score\W+\d+\.\d%")
    if refused:
        assert f"33.3% (1 of 3 trials; {NOTE})" in line, line
    else:
        assert "50.0% (1 of 2 trials)" in line, line
        assert "unscored" not in line, line
    raw = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert raw.exit_code == 0, raw.output
    assert "unscored trial as 0" not in raw.output
    assert json.loads(raw.output)["score"] == ("33.3%" if refused else "50.0%")


@pytest.mark.parametrize("refused", [True, False], ids=["refused", "all-scored"])
def test_bench_results_headline(tmp_path, refused):
    """`bench results <job>`' headline (its attempted figure) and the jobs
    listing `bench results <jobs folder>` prints."""
    from benchflow.cli.main import app

    job, _ = _job(tmp_path, refused=refused)
    out = CliRunner().invoke(app, ["results", str(job)], terminal_width=200)
    assert out.exit_code == 0, out.output
    line = _line(out.output, r"attempted passed")
    listing = CliRunner().invoke(app, ["results", str(job.parent)], terminal_width=200)
    assert listing.exit_code == 0, listing.output
    row = _line(listing.output, r"\d+/\d+ \(\d")
    if refused:
        assert f"1/3 attempted passed (33%; {NOTE})" in line, line
        assert f"1/3 (33.3%; {NOTE})" in row, row
    else:
        assert "1/2 attempted passed (50%)," in line, line
        assert "1/2 (50.0%)" in row, row
        assert "unscored trial" not in line + row, (line, row)


def test_an_evaluation_results_repr(tmp_path):
    """What an SDK caller sees printing the result."""
    _, result = _job(tmp_path / "a", refused=True)
    assert f"score=0.333 ({NOTE}), " in repr(result)
    _, result = _job(tmp_path / "b", refused=False)
    assert "score=0.500, " in repr(result)
