"""``bench cost`` — recompute rollout cost post hoc from recorded usage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from benchflow.cost import recompute_rollout_cost, update_result_json, write_cost_report
from benchflow.pricing import PRICE_TABLES, price_table


def _find_rollout_dirs(paths: list[Path]) -> list[Path]:
    """Expand each path to rollout dirs (a dir with result.json, or its descendants)."""
    found: list[Path] = []
    for p in paths:
        p = p.expanduser()
        if (p / "result.json").is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(r.parent for r in p.rglob("result.json")))
    return found


def cost_command(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Rollout dir(s), or parent dirs to scan for result.json."),
    ] = None,
    pricing: Annotated[
        str,
        typer.Option(
            "--pricing",
            help="Price table: 'list' (official list prices) or a table version.",
        ),
    ] = "list",
    source: Annotated[
        str,
        typer.Option(
            "--source", help="auto | llm_trajectory | otel_usage | result_aggregate"
        ),
    ] = "auto",
    write: Annotated[
        bool,
        typer.Option(
            "--write", help="Write cost_recompute.json into each rollout dir."
        ),
    ] = False,
    update_result: Annotated[
        bool,
        typer.Option(
            "--update-result",
            help="Also rewrite final_metrics.total_cost_usd in result.json.",
        ),
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the full report(s) as JSON.")
    ] = False,
    list_tables: Annotated[
        bool,
        typer.Option("--list-tables", help="Print the known price tables and exit."),
    ] = False,
) -> None:
    """Recompute total_cost_usd for rollout dirs from per-turn usage and list prices."""
    if list_tables:
        seen = set()
        for name, table in PRICE_TABLES.items():
            if table.version in seen:
                continue
            seen.add(table.version)
            typer.echo(json.dumps({"name": name, **table.to_dict()}, indent=2))
        raise typer.Exit()
    try:
        table = price_table(pricing)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    if source not in ("auto", "llm_trajectory", "otel_usage", "result_aggregate"):
        typer.echo(f"invalid --source {source!r}", err=True)
        raise typer.Exit(2)

    dirs = _find_rollout_dirs(paths or [])
    if not dirs:
        typer.echo("no rollout dirs (result.json) found", err=True)
        raise typer.Exit(1)

    reports = []
    for d in dirs:
        report = recompute_rollout_cost(d, pricing=table, source=source)  # type: ignore[arg-type]
        reports.append(report)
        if write:
            write_cost_report(report)
        if update_result:
            update_result_json(report)
        if not as_json:
            rec = (
                "n/a"
                if report.recorded_cost_usd is None
                else f"{report.recorded_cost_usd:.4f}"
            )
            tot = (
                "UNPRICED"
                if report.total_cost_usd is None
                else f"{report.total_cost_usd:.4f}"
            )
            delta = (
                "" if report.delta_usd is None else f"  delta {report.delta_usd:+.4f}"
            )
            models = ", ".join(m.model for m in report.per_model) or "-"
            typer.echo(
                f"{d}: {tot} USD ({table.version}, {report.source}, {report.n_records} records, {models})"
                f"  recorded {rec}{delta}"
            )
    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in reports], indent=2, default=str))
    if len(reports) > 1 and not as_json:
        priced = [r for r in reports if r.total_cost_usd is not None]
        total = sum(r.total_cost_usd or 0 for r in priced)
        recorded = sum(
            r.recorded_cost_usd or 0 for r in priced if r.recorded_cost_usd is not None
        )
        typer.echo(
            f"TOTAL {len(priced)}/{len(reports)} priced: {total:.2f} USD ({table.version}); recorded {recorded:.2f} USD"
        )


def register_cost(app: typer.Typer) -> None:
    app.command("cost", rich_help_panel="Core")(cost_command)
