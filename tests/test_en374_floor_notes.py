"""Every surface that prints a cost total says when it is a floor.

Review of PR #147 (F2b-R147B, 2026-11-14): #122 marks a trial whose count
missed calls (``usage_incomplete``, ``cost_incomplete``), and some readers
still printed its cost as whole:
- L1: ``bench backfill-cost`` left summary.json's ``spend_all_attempts`` as
  it was before the backfill priced a trial;
- L2: ``bench eval run``'s closing ``Cost:`` line, the live progress footer
  and the job export's ``cost_usd`` had no flag, and the closing lines did
  not show the budget's ``cost_cap_note``;
- N1: no test went through a real stopped attempt's receipt;
- N3: ``bench results`` and the backfill line called a trial whose cost is
  unknown "a floor" too.

Review of PR #209 (F2b-R209, 2026-11-14):
- L1: the report.md every job writes printed its cost with no flag;
- N1: the export's new field was labelled minor 1.2 in a version 2 schema;
- N2: nothing pinned that a cut-off trial with no cost is no floor, in the
  closing line or the export;
- N3: the export called summary.json's ``trials_cost_incomplete``
  ``cost_incomplete_trials``, and read a minor-1 document as 0;
- N4: telemetry.md said ``spend_all_attempts`` agrees with ``budget.spent``,
  which a backfill does not update.
Every value here is made up.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from benchflow.models import RolloutResult

REASON = "a turn ended without its usage report"
FLOOR = "counted only some of their calls, so the total is a floor"


def _trial(
    job: Path,
    name: str,
    cost: float | None,
    *,
    incomplete: bool = False,
    model: str = "claude-opus-5-5",
) -> RolloutResult:
    folder = job / f"{name}__abcd1234"
    folder.mkdir(parents=True)
    agent = {
        "n_input_tokens": 1000,
        "n_output_tokens": 100,
        "total_tokens": 1100,
        "cost_usd": cost,
        "usage_source": "end_of_turn",
        "cost_source": None if cost is None else "list_price_estimate",
    }
    if incomplete:
        agent.update(usage_incomplete=True, usage_incomplete_reason=REASON)
        if cost is not None:
            agent["cost_incomplete"] = True
    payload = {
        "task_name": name,
        "rollout_name": folder.name,
        "rewards": {"reward": 1.0},
        "agent_name": "claude-agent-acp",
        "model": model,
        "started_at": "2026-11-14T08:00:00+00:00",
        "finished_at": "2026-11-14T08:01:00+00:00",
        "agent_result": agent,
    }
    (folder / "result.json").write_text(json.dumps(payload))
    return RolloutResult.from_dict(payload, rollout_dir=folder)


def _plain(text: str) -> str:
    return " ".join(re.sub("[─-╿]", " ", text).split())


def test_the_closing_cost_line_says_the_total_is_a_floor(tmp_path, monkeypatch):
    import benchflow.cli._shared as shared

    job = tmp_path / "jobs" / "run"
    job.mkdir(parents=True)
    results = {
        "one": _trial(job, "one", 0.01),
        "two": _trial(job, "two", 0.02, incomplete=True),
        # Cut off with no cost: its tokens are a floor, the total's USD is not
        # (F2b-R209 N2).
        "three": _trial(job, "three", None, incomplete=True),
    }
    rec = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_outcomes(SimpleNamespace(results=results), job)
    assert (
        "Cost: $0.03, 3,300 tokens (1 of 3 trials reported no cost); "
        f"1 trial(s) {FLOOR}"
    ) in _plain(rec.file.getvalue())


def test_the_closing_lines_show_the_budget_s_cap_note(monkeypatch):
    import benchflow.cli._shared as shared

    note = (
        "--max-cost-usd compared the counted part only: 1 attempt counted only "
        "some of their calls, so spent.cost_usd is a floor"
    )
    rec = Console(file=io.StringIO(), width=240)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_eval_result(
        SimpleNamespace(
            passed=1,
            total=1,
            errored=0,
            verifier_errored=0,
            score=1.0,
            job_name="j",
            budget={"stopped": False, "cost_cap_note": note},
        )
    )
    assert f"Budget: {note}" in _plain(rec.file.getvalue())


def test_the_live_footer_marks_a_floor_total():
    from benchflow.cli._live_progress import LiveEvalProgress

    def rendered(dashboard) -> str:
        out = Console(file=io.StringIO(), width=200)
        out.print(dashboard.__rich__())
        return out.file.getvalue()

    def result(name, cost, incomplete=False):
        return RolloutResult(
            task_name=name,
            rewards={"reward": 1.0},
            total_tokens=1100,
            cost_usd=cost,
            usage_source="provider_response",
            usage_incomplete=incomplete,
        )

    dash = LiveEvalProgress(
        Console(), label="bench", agent="claude", model="opus", sandbox="docker"
    )
    dash.on_plan(2, 0, 2)
    dash.on_result("one", result("one", 0.01))
    # The final screen's cost grammar: a floor leads with ≥.
    assert "$0.0100" in rendered(dash) and "≥" not in rendered(dash)
    dash.on_result("two", result("two", 0.02, incomplete=True))
    assert "≥$0.0300" in rendered(dash)


def test_the_job_export_counts_its_floor_costs(tmp_path):
    import benchflow as bf
    from benchflow.job_export import SCHEMA_MINOR

    job = tmp_path / "job"
    _trial(job, "one", 0.01)
    _trial(job, "two", 0.02, incomplete=True)
    _trial(job, "three", None, incomplete=True)
    document = bf.load_job(job).to_json_dict()
    assert document["cost_usd"] == pytest.approx(0.03)
    # The unknown cost is not a floor (F2b-R209 N2), and the count has
    # summary.json's name (N3).
    assert document["trials_cost_incomplete"] == 1
    assert "cost_incomplete_trials" not in document
    assert document["schema_minor"] == SCHEMA_MINOR["job"] == 2
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/reference/schemas/benchflow-job.v2.schema.json"
        ).read_text()
    )
    assert "trials_cost_incomplete" in schema["properties"]
    assert "cost_incomplete_trials" not in schema["properties"]


@pytest.mark.asyncio
async def test_a_backfill_brings_every_attempt_s_spend_up_to_date(
    tmp_path, monkeypatch
):
    """R147B's L1 input: a trial priced at $0.0100, and R122's cut-off trial
    with its cost removed, which the backfill prices at $0.0234."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.cli.main import app
    from benchflow.evaluation import _spend_all_attempts
    from tests.test_en374_usage_incomplete import _cut_off_after_a_reported_turn

    rollout, trial = await _cut_off_after_a_reported_turn(tmp_path, monkeypatch)
    rollout._build_result()
    job = trial.parent
    path = trial / "result.json"
    data = json.loads(path.read_text())
    data["rewards"] = {"reward": 0.25}
    agent = data["agent_result"]
    for key in ("cost_incomplete", "cost_source", "price_source", "price_table_date"):
        agent.pop(key, None)
    agent["cost_usd"] = None
    path.write_text(json.dumps(data))
    _trial(job, "priced", 0.01)
    rows = {p.parent.name: json.loads(p.read_text()) for p in job.glob("*/result.json")}
    summary = {"job_name": job.name, **usage_summary(rows), **_spend_all_attempts(job)}
    (job / "summary.json").write_text(json.dumps(summary))
    assert summary["spend_all_attempts"]["cost_usd"] == pytest.approx(0.01)

    out = CliRunner().invoke(app, ["backfill-cost", str(job)], env={"COLUMNS": "300"})
    assert out.exit_code == 0, out.output
    after = json.loads((job / "summary.json").read_text())
    spend = after["spend_all_attempts"]
    assert spend == _spend_all_attempts(job)["spend_all_attempts"]
    assert spend["cost_usd"] == pytest.approx(0.0334)
    assert spend["attempts_without_cost"] == 0
    assert spend["attempts_cost_incomplete"] == 1
    report = json.loads((job / "backfill-report.json").read_text())["job"]
    assert report["summary_after"]["spend_all_attempts"]["cost_usd"] == pytest.approx(
        0.0334
    )


@pytest.mark.asyncio
async def test_a_stopped_attempt_s_receipt_keeps_the_partial_count(tmp_path):
    """A real stop inside ``cancelled_attempt_receipt``: the receipt keeps
    ``usage_incomplete`` and its reason, and a resume's budget and the job's
    spend both count the attempt's cost as a floor."""
    from benchflow._utils.evaluation_results import attempt_spend_summary
    from benchflow.budget import (
        CANCELLED_BUDGET_RECORD,
        Budget,
        BudgetGuard,
        cancelled_attempt_receipt,
    )

    usage = {
        "total_tokens": 4620,
        "cost_usd": 0.0234,
        "usage_source": "end_of_turn",
        "usage_incomplete": True,
        "usage_incomplete_reason": REASON,
    }
    entered = asyncio.Event()

    async def attempt():
        with cancelled_attempt_receipt(lambda: tmp_path, lambda: usage):
            entered.set()
            await asyncio.sleep(30)

    task = asyncio.create_task(attempt())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    receipt = json.loads((tmp_path / CANCELLED_BUDGET_RECORD).read_text())
    assert receipt["agent_result"]["usage_incomplete"] is True
    assert receipt["agent_result"]["usage_incomplete_reason"] == REASON
    record = {**receipt, "task_name": "task-0", "_budget_cancelled": True}
    guard = BudgetGuard(Budget(max_cost_usd=5.0))
    guard.seed([record])
    assert guard.spent()["cost_incomplete_trials"] == 1
    spend = attempt_spend_summary([record])["spend_all_attempts"]
    assert spend["attempts_cost_incomplete"] == 1


def test_an_unknown_cost_is_never_called_a_floor(tmp_path):
    from benchflow._utils.result_metadata import floor_note
    from benchflow.cli.main import app

    assert floor_note(2, 2) == "their tokens and cost are a floor"
    assert floor_note(1, 0) == "their tokens are a floor (no cost is known)"
    assert (
        floor_note(2, 1) == "their tokens are a floor, and so is the cost of 1 of them"
    )

    job = tmp_path / "job"
    _trial(job, "priced", 0.01)
    _trial(job, "unknown", None, incomplete=True)
    shown = CliRunner().invoke(app, ["results", str(job)], env={"COLUMNS": "300"})
    plain = _plain(shown.stdout)
    assert (
        "+ 1 trial(s) counted only some of their calls, so their tokens are a floor "
        f"(no cost is known): {REASON}"
    ) in plain
    assert "tokens and cost are a floor" not in plain


def test_the_backfill_line_calls_an_unknown_cost_no_floor(tmp_path):
    """A marked trial the backfill cannot price (no logs, no list price):
    its tokens are a floor, its cost stays unknown."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow.cli.main import app

    job = tmp_path / "job"
    _trial(job, "unknown", None, incomplete=True, model="zq9-fake-unpriced-model")
    rows = {p.parent.name: json.loads(p.read_text()) for p in job.glob("*/result.json")}
    (job / "summary.json").write_text(
        json.dumps({"job_name": job.name, **usage_summary(rows)})
    )
    out = CliRunner().invoke(app, ["backfill-cost", str(job)], env={"COLUMNS": "300"})
    assert out.exit_code == 0, out.output
    line = " ".join(out.stdout.split())
    assert (
        "+ 1 trial(s) counted only some of their calls, so their tokens are a "
        "floor (no cost is known)"
    ) in line


def _floor_job(tmp_path: Path, trials: list[tuple[float | None, bool]], budget=None):
    """A real ``Evaluation`` run whose rollouts write ``_trial``'s records:
    (cost, cut off) per task, task-0 first."""
    from unittest.mock import AsyncMock

    from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig
    from tests.test_en374_budget_caps import _tasks

    run = Evaluation(
        tasks_dir=_tasks(tmp_path, len(trials)),
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(
            agent="claude-agent-acp",
            model="claude-opus-5-5",
            concurrency=1,
            retry=RetryConfig(max_retries=0),
        ),
        budget=budget,
        preflight=False,
    )

    async def fake_run(task_path, _cfg):
        cost, incomplete = trials[int(task_path.name.split("-")[1])]
        job_dir = run._jobs_dir / run._job_name
        return _trial(job_dir, task_path.name, cost, incomplete=incomplete)

    run._run_single_task = AsyncMock(side_effect=fake_run)
    return run


@pytest.mark.asyncio
async def test_the_job_report_says_its_cost_is_a_floor(tmp_path):
    """F2b-R209 L1: report.md, as ``Evaluation.run()`` writes it, for a
    $0.01 trial, a cut-off trial priced at $0.02 and a cut-off trial with no
    cost. Only the priced cut-off trial makes the cost a floor."""
    run = _floor_job(tmp_path, [(0.01, False), (0.02, True), (None, True)])
    await run.run()
    report = (run._jobs_dir / run._job_name / "report.md").read_text()
    assert f"cost $0.03 (1 trial(s) {FLOOR})" in report

    clean = _floor_job(tmp_path / "clean", [(0.01, False), (0.02, False)])
    await clean.run()
    report = (clean._jobs_dir / clean._job_name / "report.md").read_text()
    assert "Tokens 2,200; cost $0.03" in report
    assert "cost $0.03 (" not in report and "floor" not in report


def test_the_export_s_floor_count_is_labelled_minor_2_2():
    """F2b-R209 N1: the job schema is version 2, so the field added at minor
    2 is "(2.2)", in the model, the schema file, files.md and the
    CHANGELOG."""
    from benchflow.job_export import SCHEMA_MINOR, JobExport

    repo = Path(__file__).parents[1]
    label = (
        f"({JobExport.model_fields['schema_version'].default}.{SCHEMA_MINOR['job']})"
    )
    assert label == "(2.2)"
    field = JobExport.model_fields["trials_cost_incomplete"]
    assert field.description.endswith(label)
    schema = json.loads(
        (repo / "docs/reference/schemas/benchflow-job.v2.schema.json").read_text()
    )
    assert schema["properties"]["trials_cost_incomplete"]["description"].endswith(label)
    row = next(
        line
        for line in (repo / "docs/reference/files.md").read_text().splitlines()
        if line.startswith("| `trials_cost_incomplete`")
    )
    assert row.rstrip(" |").endswith(label)
    entry = next(
        line
        for line in (repo / "CHANGELOG.md").read_text().splitlines()
        if "Every cost total says when it is a floor" in line
    )
    assert "`trials_cost_incomplete`" in entry and "minor 2.2" in entry
    assert "1.2" not in entry


def test_a_minor_1_export_does_not_read_as_no_floor(tmp_path):
    """F2b-R209 N3: a document below minor 2 never counted its floor trials,
    so reading one gives None, not 0."""
    import benchflow as bf
    from benchflow.job_export import JobExport

    job = tmp_path / "job"
    _trial(job, "one", 0.01)
    _trial(job, "two", 0.02, incomplete=True)
    document = bf.load_job(job).to_json_dict()
    assert document["trials_cost_incomplete"] == 1
    old = {k: v for k, v in document.items() if k != "trials_cost_incomplete"}
    old["schema_minor"] = 1
    assert JobExport.model_validate(old).trials_cost_incomplete is None
    schema = JobExport.model_json_schema(mode="serialization")
    assert "trials_cost_incomplete" not in schema.get("required", [])
    assert schema["properties"]["trials_cost_incomplete"]["default"] is None


@pytest.mark.asyncio
async def test_a_backfill_moves_every_attempt_s_spend_and_not_the_budget(tmp_path):
    """F2b-R209 N4: a job with a token cap, a $0.01 trial and a cut-off trial
    with no cost. ``bench backfill-cost`` prices the second trial: summary.json's
    ``spend_all_attempts`` follows, and ``budget.spent`` stays what the cap
    saw while the job ran. telemetry.md says so."""
    from benchflow import Budget
    from benchflow.cli.main import app

    run = _floor_job(
        tmp_path, [(0.01, False), (None, True)], budget=Budget(max_tokens=10**9)
    )
    await run.run()
    job = run._jobs_dir / run._job_name
    before = json.loads((job / "summary.json").read_text())
    assert before["budget"]["spent"]["cost_usd"] == pytest.approx(0.01)
    assert before["spend_all_attempts"]["cost_usd"] == pytest.approx(0.01)

    out = CliRunner().invoke(app, ["backfill-cost", str(job)], env={"COLUMNS": "300"})
    assert out.exit_code == 0, out.output
    after = json.loads((job / "summary.json").read_text())
    assert after["spend_all_attempts"]["cost_usd"] > 0.01
    assert after["spend_all_attempts"]["attempts_without_cost"] == 0
    assert after["budget"] == before["budget"]

    telemetry = " ".join(
        (Path(__file__).parents[1] / "docs/scale/telemetry.md").read_text().split()
    )
    assert "so with a budget it agrees with `budget.spent`." not in telemetry
    assert (
        "`bench backfill-cost` updates `spend_all_attempts` with the trials it "
        "prices, not `budget.spent`"
    ) in telemetry
