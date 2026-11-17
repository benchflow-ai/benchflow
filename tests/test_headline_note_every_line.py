"""Every line that prints the benchflow.score.v1 headline says, on that line,
how many unscored trials it counts as 0 (the lead's condition, 17:30Z):
`bench results <job>`'s "Score by" line, report.md's totals, the run card's
Score line and a comparison's "Score by" sentence. A job whose trials were
all scored keeps the plain number."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow._utils.scoring import score_summary_fields
from tests.test_headline_score_unscored import NOTE, _job, _line

IDS = ["refused", "all-scored"]


@pytest.mark.parametrize("refused", [True, False], ids=IDS)
def test_bench_results_score_by_line(tmp_path, refused):
    from benchflow.cli.main import app

    job, _ = _job(tmp_path, refused=refused)
    out = CliRunner().invoke(app, ["results", str(job)], terminal_width=200)
    assert out.exit_code == 0, out.output
    line = _line(out.output, r"Score by benchflow")
    if refused:
        assert (
            f"Score by benchflow.score.v1: 33.3% (1 of 3 trials; {NOTE}; 95% interval"
        ) in line, line
    else:
        assert "Score by benchflow.score.v1: 50.0% (1 of 2 trials; 95% interval" in (
            line
        ), line
        assert "unscored trial" not in line, line


@pytest.mark.parametrize("refused", [True, False], ids=IDS)
def test_report_md_totals(tmp_path, refused):
    from benchflow.report import build_report

    job, _ = _job(tmp_path, refused=refused)
    md = build_report([job], "headline")
    if refused:
        assert f"1 passed (33.3%; {NOTE}; 95% interval" in md, md
    else:
        assert "1 passed (50.0%; 95% interval" in md, md
        assert "unscored trial" not in md, md


@pytest.mark.parametrize("refused", [True, False], ids=IDS)
def test_run_card_score_line(tmp_path, refused):
    from benchflow.run_checks import after_run_lines

    job, _ = _job(tmp_path, refused=refused)
    rows = [json.loads(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    lines = after_run_lines({"scoring": score_summary_fields(rows)["scoring"]})
    (score,) = [line for line in lines if line.startswith("  Score:")]
    if refused:
        assert f"1 of 3 trials passed (33.3%; {NOTE}; 95% interval" in score, score
    else:
        assert "1 of 2 trials passed (50.0%; 95% interval" in score, score
        assert "unscored trial" not in score, score


def test_comparison_score_by_sentence(tmp_path):
    a, _ = _job(tmp_path / "a", refused=True)
    b, _ = _job(tmp_path / "b", refused=True)
    md = bf.compare(a, b, on_mismatch="ignore").to_markdown()
    (sentence,) = [line for line in md.splitlines() if "Score by " in line]
    assert sentence.count(f"(1 of 3 trials; {NOTE}; 95% interval") == 2, sentence
