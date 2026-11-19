"""`bench results <jobs folder>`: a Summary cell's unscored-trials note stays on
its number's line whatever the console's width, as `bench eval metrics`'
Score row does; and a summary that can't tell failed from unscored trials
gives no note."""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from typer.testing import CliRunner

import benchflow.cli._shared as shared
from tests.test_headline_score_unscored import NOTE, _job, _line


@pytest.mark.parametrize("width", [40, 60, 80])
def test_the_listing_keeps_the_note_on_the_numbers_line(tmp_path, monkeypatch, width):
    from benchflow.cli.main import app

    job, _ = _job(tmp_path, refused=True)
    # A run's own folder name, as `bench run` writes it.
    job = job.rename(job.parent / "2026-11-16__19-42-04")
    console = Console(file=io.StringIO(), width=width)
    monkeypatch.setattr(shared, "console", console)
    out = CliRunner().invoke(app, ["results", str(job.parent)])
    assert out.exit_code == 0, out.output
    shown = console.file.getvalue()
    row = _line(shown, r"\d+/\d+ \(\d")
    assert f"1/3 (33.3%; {NOTE})" in row, shown


def test_a_summary_without_a_failed_count_gives_no_note():
    from benchflow._utils.scoring import summary_unscored_note

    assert summary_unscored_note({"total": 10, "passed": 9, "score": "90.0%"}) == ""
    assert summary_unscored_note({"total": 10, "passed": 8, "failed": 1}) == (
        "counts 1 unscored trial as 0; scored only: 88.9%"
    )
