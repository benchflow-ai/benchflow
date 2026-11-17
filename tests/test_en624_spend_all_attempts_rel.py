"""Every reader of a resumed job agrees with summary.json or states its scope (EN-624, 0.9.x).

A job cancelled mid-trial (Ctrl-C) and resumed holds two attempts of one task:
the stopped one (its result.json says ``interrupted``, beside its
cancellation receipt) and the resumed one, which the readers keep. summary.json
counts both; the readers counted the kept one and did not say so.

The numbers are a real cancel + resume on 0.9.x (claude-agent-acp on a
list-priced model): the kept attempt 33,335 tokens and $0.0001553272, the
stopped one 16,632 and $0.0000770336, both 49,967 and $0.0002323608. No trial
content is kept.

Now summary.json writes ``spend_all_attempts`` (every attempt, the 0.10
line's block); ``bench eval metrics`` takes summary.json's figures; the job
export, ``bench results``, report.md and the closing ``Cost:`` line keep the
kept attempt, say so, and give every attempt's spend beside it. A job whose
attempts are all kept reads as before. Each reader also marks a cost priced
from a count that missed calls (``usage_incomplete``) as a floor, from the
records the run writes and with those fields absent (older records).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import benchflow as bf
import benchflow.cli._shared as shared
from benchflow.cli.main import app
from benchflow.evaluation import EvaluationConfig, EvaluationResult
from benchflow.models import RolloutResult

KEPT = {
    "n_input_tokens": 173,
    "n_output_tokens": 138,
    "n_cache_read_tokens": 33024,
    "n_cache_creation_tokens": 0,
    "total_tokens": 33335,
    "cost_usd": 0.0001553272,
}
STOPPED = {
    "n_input_tokens": 20,
    "n_output_tokens": 100,
    "n_cache_read_tokens": 16512,
    "n_cache_creation_tokens": 0,
    "total_tokens": 16632,
    "cost_usd": 7.70336e-05,
}
PRICED = {
    "usage_source": "proxy",
    "usage_source_legacy": "provider_response",
    "cost_source": "list_price_estimate",
    "price_source": "litellm",
    "price_table_date": "2026-11-04",
    "no_usage_reason": None,
}
EVERY = "every attempt (spend_all_attempts): $0.000232, 49,967 tokens over 1 attempt and 1 stopped attempt"


def _result(job: Path, folder: str, usage: dict, **fields) -> Path:
    trial = job / folder
    trial.mkdir(parents=True)
    payload = {
        "task_name": folder.split("__")[0],
        "rollout_name": folder,
        "agent": "claude-agent-acp",
        "model": "claude-haiku-4-5",
        "rewards": {"reward": 1.0},
        "started_at": "2026-11-14 23:36:17.329200",
        "finished_at": "2026-11-14 23:39:09.656990",
        "agent_result": {**usage, **PRICED},
        **fields,
    }
    (trial / "result.json").write_text(json.dumps(payload))
    return trial


def _receipt(trial: Path, usage: dict | None) -> None:
    receipt = {
        "version": 1,
        "reason": "interrupted",
        "timing": {"total": 25.6, "verifier_sandbox_total": 0},
    }
    if usage is not None:
        receipt["agent_result"] = {**usage, **PRICED}
    (trial / "budget-cancelled.json").write_text(json.dumps(receipt))


def _job(tmp_path: Path, *, stopped: bool = True) -> Path:
    job = tmp_path / "jobs" / "rel-cancel"
    (tmp_path / "tasks" / "cancel").mkdir(parents=True)
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "tasks_dir": str(tmp_path / "tasks")})
    )
    _result(job, "cancel__23af19de", KEPT)
    if stopped:
        old = _result(
            job,
            "cancel__8affcac8",
            STOPPED,
            rewards=None,
            interrupted=True,
            started_at="2026-11-14 23:35:46.000000",
            finished_at="2026-11-14 23:36:12.000000",
        )
        _receipt(old, STOPPED)
    (job / "summary.json").write_text("{}")
    return job


def _cli(*args: str) -> str:
    result = CliRunner().invoke(
        app, list(args), terminal_width=250, env={"COLUMNS": "250"}
    )
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def _metrics(job: Path) -> dict:
    result = CliRunner().invoke(app, ["eval", "metrics", str(job), "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _closing(monkeypatch, job: Path, trials: dict[str, Path]) -> str:
    results = {name: RolloutResult.load(path) for name, path in trials.items()}
    rec = Console(file=io.StringIO(), width=300)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_eval_result(
        EvaluationResult(
            job_name=job.name,
            config=EvaluationConfig(),
            total=len(results),
            passed=len(results),
            elapsed_sec=10.0,
            job_dir=job,
            results=results,
        ),
        job,
    )
    return next(
        " ".join(line.split())
        for line in rec.file.getvalue().splitlines()
        if "Cost:" in line
    )


# ── a real cancel + resume ───────────────────────────────────────────────


def test_summary_json_counts_every_attempt_in_spend_all_attempts(tmp_path):
    from benchflow.evaluation import _spend_all_attempts

    spend = _spend_all_attempts(_job(tmp_path))["spend_all_attempts"]
    assert spend == {
        "attempts": 1,
        "stopped_attempts": 1,
        "cost_usd": 0.0002323608,
        "tokens": 49967,
        "attempts_without_cost": 0,
        "attempts_without_tokens": 0,
        "attempts_without_model_call": 0,
        "stopped_attempts_without_usage": 0,
        "stopped_attempts_without_cost": 0,
        "attempts_cost_incomplete": 0,
    }


def test_spend_all_attempts_agrees_with_the_budgets_spent_and_the_totals(tmp_path):
    """summary.json's total_* (kept + stopped), budget.spent and the new
    block say the same on this job."""
    from benchflow._utils.evaluation_results import usage_summary
    from benchflow._utils.result_paths import interrupted_trial_results
    from benchflow.budget import Budget, BudgetGuard
    from benchflow.evaluation import _recorded_attempts, _spend_all_attempts

    job = _job(tmp_path)
    kept = {"cancel": json.loads((job / "cancel__23af19de/result.json").read_text())}
    totals = usage_summary({**kept, **interrupted_trial_results(job)})
    guard = BudgetGuard(Budget(max_cost_usd=0.25))
    guard.seed(_recorded_attempts(job)[0])
    spend = _spend_all_attempts(job)["spend_all_attempts"]
    assert (totals["total_tokens"], totals["total_cost_usd"]) == (49967, 0.0002323608)
    assert (guard.spent()["tokens"], guard.spent()["cost_usd"]) == (49967, 0.0002323608)
    assert (spend["tokens"], spend["cost_usd"]) == (49967, 0.0002323608)


def test_bench_eval_metrics_agrees_with_summary_json(tmp_path, monkeypatch):
    job = _job(tmp_path)
    metrics = _metrics(job)
    assert (metrics["total_tokens"], metrics["total_cost_usd"]) == (49967, 0.0002323608)
    assert metrics["spend_all_attempts"]["tokens"] == 49967
    # Rich sizes shared.console from the environment, and some give it 80
    # columns whatever COLUMNS says (a dumb terminal: TERM=dumb with
    # FORCE_COLOR), so the Spend row wraps. Pin width and height: Rich
    # ignores a width alone on a dumb terminal.
    rec = Console(file=io.StringIO(), width=250, height=50)
    monkeypatch.setattr(shared, "console", rec)
    _cli("eval", "metrics", str(job))
    assert (
        "Spend (spend_all_attempts) │ $0.000232, 49,967 tokens over 1 attempt and 1 "
        "stopped attempt"
    ) in " ".join(rec.file.getvalue().split())


def test_the_job_export_states_its_scope_with_every_attempts_spend_beside(tmp_path):
    doc = bf.load_job(_job(tmp_path)).to_json_dict()
    assert doc["cost_usd"] == 0.0001553272  # the kept attempt, as documented
    assert doc["trials"][0]["usage"]["total_tokens"] == 33335
    spend = doc["spend_all_attempts"]
    assert (spend["cost_usd"], spend["tokens"], spend["stopped_attempts"]) == (
        0.0002323608,
        49967,
        1,
    )
    assert doc["schema_minor"] == 1 and doc["trials_cost_incomplete"] == 0


def test_bench_results_labels_the_kept_attempt_and_gives_every_attempt(tmp_path):
    out = _cli("results", str(_job(tmp_path)))
    assert f"Cost: each task's kept attempt; {EVERY}" in out, out


def test_report_md_gives_every_attempt_beside_its_cost(tmp_path):
    md = bf.load_job(_job(tmp_path)).to_markdown()
    assert f"- Cost: $0.0002 over 1 rollout(s); {EVERY}" in md, md


def test_the_closing_cost_line_labels_the_kept_attempt(tmp_path, monkeypatch):
    """As a run prints it: from the summary.json it has just written. An
    older summary.json without the block reads as before."""
    from benchflow.evaluation import _spend_all_attempts

    job = _job(tmp_path)
    kept = {"cancel": job / "cancel__23af19de"}
    assert _closing(monkeypatch, job, kept) == "Cost: $0.00, 33,335 tokens"
    (job / "summary.json").write_text(json.dumps(_spend_all_attempts(job)))
    line = _closing(monkeypatch, job, kept)
    assert f"$0.00, 33,335 tokens (each task's kept attempt); {EVERY}" in line, line


def test_one_trial_of_a_job_reads_its_own_tasks_attempts(tmp_path):
    """A job folder read from inside (one trial) counts that trial's task's
    attempts, not the job's; read from its root, every task's."""
    job = _job(tmp_path)
    (tmp_path / "tasks" / "hello").mkdir()
    other = _result(job, "hello__1234abcd", {**KEPT, "total_tokens": 1000})
    whole = bf.load_job(job).to_json_dict()["spend_all_attempts"]
    assert (whole["attempts"], whole["stopped_attempts"], whole["tokens"]) == (
        2,
        1,
        50967,
    )
    alone = bf.load_job(other).to_json_dict()["spend_all_attempts"]
    assert (alone["attempts"], alone["stopped_attempts"], alone["tokens"]) == (
        1,
        0,
        1000,
    )
    assert "every attempt" not in _cli("results", str(other))
    assert _metrics(other)["total_tokens"] == 1000
    kept = bf.load_job(job / "cancel__23af19de").to_json_dict()["spend_all_attempts"]
    assert (kept["stopped_attempts"], kept["tokens"]) == (1, 49967)


# ── a job whose attempts are all kept reads as before ────────────────────


def test_a_job_with_no_stopped_attempt_reads_as_before(tmp_path, monkeypatch):
    job = _job(tmp_path, stopped=False)
    metrics = _metrics(job)
    assert (metrics["total_tokens"], metrics["total_cost_usd"]) == (33335, 0.0001553272)
    assert "Spend" not in _cli("eval", "metrics", str(job))
    out = _cli("results", str(job))
    assert "every attempt" not in out and "kept attempt" not in out
    assert (
        bf.load_job(job).to_markdown().endswith("- Cost: $0.0002 over 1 rollout(s)\n")
    )
    assert bf.load_job(job).to_json_dict()["cost_usd"] == 0.0001553272
    line = _closing(monkeypatch, job, {"cancel": job / "cancel__23af19de"})
    assert line == "Cost: $0.00, 33,335 tokens", line


def test_a_retried_attempt_is_in_the_spend_and_the_kept_one_in_the_cost(tmp_path):
    job = _job(tmp_path, stopped=False)
    _result(
        job,
        "cancel__0a1b2c3d",
        {**STOPPED, "total_tokens": 1000, "cost_usd": 0.01},
        rewards=None,
        error="agent exited 1",
        error_category="agent_error",
        started_at="2026-11-14 23:30:00.000000",
        finished_at="2026-11-14 23:31:00.000000",
    )
    spend = _metrics(job)["spend_all_attempts"]
    assert (spend["attempts"], spend["stopped_attempts"], spend["tokens"]) == (
        2,
        0,
        34335,
    )


# ── the incomplete markers ───────────────────────────────────────────────

PARTIAL = {"usage_incomplete": True, "usage_incomplete_reason": "a turn ended early"}


def _partial_job(tmp_path: Path, marker: dict) -> Path:
    job = tmp_path / "jobs" / "partial"
    (tmp_path / "tasks" / "hello").mkdir(parents=True)
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(
        json.dumps({"schema_version": 1, "tasks_dir": str(tmp_path / "tasks")})
    )
    usage = {**KEPT, "total_tokens": 311000, "cost_usd": 0.645}
    _result(job, "hello__0123abcd", usage)
    trial = _result(job, "hello2__4567cdef", usage)
    payload = json.loads((trial / "result.json").read_text())
    payload["agent_result"].update(marker)
    (trial / "result.json").write_text(json.dumps(payload))
    (job / "summary.json").write_text("{}")
    return job


def test_every_reader_marks_a_cost_from_a_count_that_missed_calls(
    tmp_path, monkeypatch
):
    job = _partial_job(tmp_path, PARTIAL)
    trial = job / "hello2__4567cdef"
    usage = bf.load_trial(trial).to_json_dict()["usage"]
    assert (usage["usage_incomplete"], usage["cost_incomplete"]) == (True, True)
    assert usage["usage_incomplete_reason"] == "a turn ended early"
    assert bf.load_job(job).to_json_dict()["trials_cost_incomplete"] == 1
    out = _cli("results", str(job))
    assert "$0.6450+" in out
    assert (
        "+ 1 trial(s) counted only some of their calls, so their tokens and cost "
        "are a floor: a turn ended early"
    ) in out, out
    assert (
        "1 rollout(s) counted only some of their calls, so the total is a floor"
    ) in bf.load_job(job).to_markdown()
    line = _closing(
        monkeypatch, job, {"hello": job / "hello__0123abcd", "hello2": trial}
    )
    assert (
        "1 trial(s) counted only some of their calls, so the total is a floor" in line
    )
    metrics = _metrics(job)
    assert (metrics["trials_usage_incomplete"], metrics["trials_cost_incomplete"]) == (
        1,
        1,
    )


def test_a_trial_the_run_finished_carries_its_marker_to_summary_json_and_the_line(
    tmp_path, monkeypatch
):
    """The closing line and summary.json read the run's own results, not
    result.json: those carry the marker too."""
    from benchflow._utils.evaluation_results import agent_result_from_rollout
    from benchflow._utils.result_metadata import incomplete_counts

    live = RolloutResult(
        task_name="hello",
        rollout_name="hello__0123abcd",
        rewards={"reward": 1.0},
        total_tokens=311000,
        cost_usd=0.645,
        usage_source="proxy",
        **PARTIAL,
    )
    assert (live.usage_incomplete, live.usage_incomplete_reason) == (
        True,
        "a turn ended early",
    )
    agent = agent_result_from_rollout(live)
    assert (agent["usage_incomplete"], agent["cost_incomplete"]) == (True, True)
    assert incomplete_counts([{"agent_result": agent}]) == {
        "trials_usage_incomplete": 1,
        "trials_cost_incomplete": 1,
    }
    job = tmp_path / "job"
    job.mkdir()
    (job / "summary.json").write_text("{}")
    rec = Console(file=io.StringIO(), width=300)
    monkeypatch.setattr(shared, "console", rec)
    shared._report_eval_result(
        EvaluationResult(
            job_name="job",
            config=EvaluationConfig(),
            total=1,
            passed=1,
            elapsed_sec=1.0,
            job_dir=job,
            results={"hello": live},
        ),
        job,
    )
    assert (
        "1 trial(s) counted only some of their calls, so the total is a floor"
        in rec.file.getvalue()
    )
    whole = RolloutResult(task_name="hello", total_tokens=10, cost_usd=0.1)
    assert "usage_incomplete" not in agent_result_from_rollout(whole)


def test_an_older_record_without_the_marker_reads_whole(tmp_path, monkeypatch):
    job = _partial_job(tmp_path, {})
    trial = job / "hello2__4567cdef"
    usage = bf.load_trial(trial).to_json_dict()["usage"]
    assert usage["usage_incomplete"] is None and usage["cost_incomplete"] is None
    assert bf.load_job(job).to_json_dict()["trials_cost_incomplete"] == 0
    out = _cli("results", str(job))
    assert "$0.6450+" not in out and "floor" not in out
    assert "floor" not in bf.load_job(job).to_markdown()
    line = _closing(
        monkeypatch, job, {"hello": job / "hello__0123abcd", "hello2": trial}
    )
    assert line == "Cost: $1.29, 622,000 tokens", line
    assert _metrics(job)["trials_cost_incomplete"] == 0


@pytest.mark.parametrize("marked", [True, False])
def test_a_stopped_attempts_receipt_marker_makes_the_spend_a_floor(tmp_path, marked):
    """A stopped attempt with no result.json is its receipt; the marker the
    receipt carries (EN-646's receipt fields) makes the spend a floor, and an
    older receipt without it reads whole."""
    from benchflow._utils.result_metadata import spend_note
    from benchflow.evaluation import _spend_all_attempts

    job = _job(tmp_path, stopped=False)
    old = job / "cancel__8affcac8"
    old.mkdir()
    (old / "config.json").write_text(json.dumps({"task_path": "tasks/cancel"}))
    _receipt(old, {**STOPPED, **(PARTIAL if marked else {})})
    spend = _spend_all_attempts(job)["spend_all_attempts"]
    assert (spend["stopped_attempts"], spend["tokens"]) == (1, 49967)
    assert spend["attempts_cost_incomplete"] == int(marked)
    note = spend_note(spend, 1)
    assert note is not None and note.startswith(EVERY)
    assert (
        note.endswith("; 1 attempt counted only some of their calls, so it is a floor")
        is marked
    )


def test_the_exports_validate_against_the_committed_schemas(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    from benchflow import job_export

    schemas = Path(__file__).resolve().parents[1] / "docs/reference/schemas"
    for kind, doc in (
        ("job", bf.load_job(_job(tmp_path)).to_json_dict()),
        (
            "trial",
            bf.load_trial(
                _partial_job(tmp_path / "p", PARTIAL) / "hello2__4567cdef"
            ).to_json_dict(),
        ),
    ):
        schema = json.loads((schemas / job_export.schema_filename(kind)).read_text())
        jsonschema.validate(doc, schema, cls=jsonschema.Draft202012Validator)
        listed = schema["properties"]
        if kind == "job":
            assert {"spend_all_attempts", "trials_cost_incomplete"} <= set(listed)
        else:
            usage = schema["$defs"]["UsageExport"]["properties"]
            assert {"usage_incomplete", "cost_incomplete"} <= set(usage)
