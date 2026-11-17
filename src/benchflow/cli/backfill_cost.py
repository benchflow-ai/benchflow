"""``bench backfill-cost`` — re-derive a finished job's tokens and cost from
the logs its trials left, at no model spend."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._shared import console, exit_if_missing, print_error


def backfill_cost(
    job_dir: Annotated[Path, typer.Argument(help="The job folder to backfill")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Write only the report; change no record"),
    ] = False,
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Where to write the report (default: <JOB_DIR>/backfill-report.json);"
            " each run's is also kept beside it, named by its UTC time",
        ),
    ] = None,
) -> None:
    """Fill the tokens and cost a job's trials lack, from the logs they left.

    Reads each trial's Claude Code session and stream logs, Codex session
    logs and the model proxy's log (zstd-compressed ones too), counts each
    model response once and prices it as a trial is priced at teardown. A
    provider's cost, and any cost already recorded, is kept; a 0 beside
    positive log usage is replaced. A kept cost without the cost labels gets
    them (usage_source, cost_source from the evidence, price_table_date),
    its numbers unchanged. Writes result.json (the earlier values under
    agent_result.backfill.original), trainer/atif.json, results.jsonl, the
    job's summary.json totals and run-card.json's after_run.usage. Run it on
    a copy of the job: a second run changes nothing. The report lists each
    trial as covered, tokens_filled, labelled, already_had_cost, uncovered
    (with the reason) or not_applicable.
    """
    from benchflow.cost_backfill import backfill_job

    exit_if_missing(job_dir, next_step="pass a job folder")
    try:
        found = backfill_job(job_dir, dry_run=dry_run, report_path=report)
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc), next_step="pass a job folder")
        raise typer.Exit(1) from None
    job = found["job"]
    counts = ", ".join(f"{n} {status}" for status, n in job["status_counts"].items())
    before, after = job["before"]["cost_usd"], job["after"]["cost_usd"]
    console.print(
        f"{job['trials']} trial(s): {counts}. Cost of the trials with one: "
        f"{_usd(before)} before, {_usd(after)} after"
        + (" (dry run: nothing written)" if dry_run else "")
        + f". Report: {escape(str(report or Path(found['job_dir']) / 'backfill-report.json'))}"
        f" (this run's: {escape(found['report_file'])})",
        highlight=False,
        soft_wrap=True,
    )
    after = job.get("summary_after") or {}
    partial = after.get("trials_usage_incomplete") or 0
    if partial:
        from benchflow._utils.result_metadata import floor_note

        costed = after.get("trials_cost_incomplete") or 0
        console.print(
            f"  + {partial} trial(s) counted only some of their calls, so "
            f"{floor_note(partial, costed)}",
            highlight=False,
            soft_wrap=True,
        )
    for reason, n in job["reason_counts"].items():
        if reason.startswith("uncovered:"):
            console.print(
                f"  uncovered, {reason.split(':', 1)[1]}: {n}", highlight=False
            )
    copy = job.get("jobs_root_summary") or {}
    if copy.get("state") == "stale":
        console.print(
            f"  {escape(copy['path'])} is stale (outside the job folder, not "
            f"written): {escape(copy['refresh'])}",
            highlight=False,
            soft_wrap=True,
        )


def _usd(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def register_backfill_cost(app: typer.Typer) -> None:
    app.command("backfill-cost")(backfill_cost)
