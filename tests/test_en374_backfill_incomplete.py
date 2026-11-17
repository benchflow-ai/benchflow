"""A backfill keeps a partial count's marker, and its report says so.

Review of PR #122 (F2b-R122D, addendum, BF1 and BF2), folded into #147:
- the backfill's report listed none of ``trials_usage_incomplete`` /
  ``trials_cost_incomplete`` and its CLI line printed the totals with no
  word of a floor;
- a backfill that priced a trial marked ``usage_incomplete`` (no list price
  at run time) wrote no ``cost_incomplete`` beside the cost, so summary.json
  and the run card said 0 trials with a partial cost where ATIF, the export
  and ``bench results`` said 1.
The trial is R122's cut-off turn (4,620 of 8,340 tokens). Values are made up.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tests.test_en374_usage_incomplete import _cut_off_after_a_reported_turn


@pytest.mark.asyncio
async def test_a_backfill_that_prices_a_partial_count_marks_its_cost(
    tmp_path, monkeypatch
):
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.cli.main import app
    from benchflow.evaluation import RUN_CARD_USAGE_FIELDS, Evaluation

    rollout, trial = await _cut_off_after_a_reported_turn(tmp_path, monkeypatch)
    rollout._build_result()
    job = trial.parent
    path = trial / "result.json"
    data = json.loads(path.read_text())
    data["rewards"] = {"reward": 0.25}  # a cut-off turn the verifier still scored
    agent = data["agent_result"]
    assert agent["usage_incomplete"] is True
    # No list price for the model at run time: tokens, no cost.
    for key in ("cost_incomplete", "cost_source", "price_source", "price_table_date"):
        agent.pop(key, None)
    agent["cost_usd"] = None
    path.write_text(json.dumps(data))
    summary = {"job_name": job.name, **usage_summary({trial.name: data})}
    (job / "summary.json").write_text(json.dumps(summary))
    Evaluation._log_run_card_after(
        SimpleNamespace(_run_card={}),
        job,
        {},
        None,
        usage={key: summary.get(key) for key in RUN_CARD_USAGE_FIELDS},
    )

    out = CliRunner().invoke(app, ["backfill-cost", str(job)], env={"COLUMNS": "300"})
    assert out.exit_code == 0, out.output

    after = json.loads(path.read_text())["agent_result"]
    assert after["cost_usd"] == pytest.approx(0.0234)
    assert after["usage_incomplete"] is True
    assert after["cost_incomplete"] is True
    summary = json.loads((job / "summary.json").read_text())
    card = json.loads((job / "run-card.json").read_text())["after_run"]["usage"]
    for record in (summary, card):
        assert (
            record["trials_usage_incomplete"],
            record["trials_cost_incomplete"],
        ) == (
            1,
            1,
        )
    report = json.loads((job / "backfill-report.json").read_text())["job"]
    assert report["summary_before"]["trials_usage_incomplete"] == 1
    assert report["summary_after"]["trials_cost_incomplete"] == 1
    line = " ".join(out.stdout.split())
    assert (
        "+ 1 trial(s) counted only some of their calls, so their tokens and cost "
        "are a floor"
    ) in line
