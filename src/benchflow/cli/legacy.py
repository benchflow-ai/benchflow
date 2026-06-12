"""Deprecated top-level benchflow commands (hidden in ``--help``).

These predate the 0.3 resource-verb subgroups (``eval``/``environment``/
``agent``) and are kept for backwards compatibility only: ``metrics``,
``view``, and ``cleanup``. Each is ``hidden=True, deprecated=True``.

Registered onto the top-level app by :func:`register_legacy`; ``cli/main.py``
only wires the call. ``cleanup`` resolves the Daytona helpers through the
``benchflow.cli.main`` module so tests that monkeypatch those names keep working.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from benchflow.cli._options import (
    AgentOption,
)
from benchflow.cli._shared import (
    console,
)


def register_legacy(app: typer.Typer) -> None:
    """Attach the deprecated top-level commands to the benchflow app."""

    @app.command(hidden=True, deprecated=True)
    def metrics(
        jobs_dir: Annotated[
            Path,
            typer.Argument(help="Jobs directory to analyze"),
        ],
        benchmark: Annotated[
            str,
            typer.Option("--benchmark", help="Benchmark name"),
        ] = "",
        agent: AgentOption = "",
        model: Annotated[
            str,
            typer.Option("--model", help="Model name"),
        ] = "",
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Output as JSON"),
        ] = False,
    ) -> None:
        """Collect and display metrics from a jobs directory."""
        from benchflow.metrics import collect_metrics

        m = collect_metrics(
            str(jobs_dir), benchmark=benchmark, agent=agent, model=model
        )
        summary = m.summary()

        if output_json:
            console.print(json.dumps(summary, indent=2))
            return

        # Pretty table
        table = Table(title=f"Results: {jobs_dir}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Total", str(summary["total"]))
        table.add_row("Passed", f"[green]{summary['passed']}[/green]")
        table.add_row("Failed", f"[red]{summary['failed']}[/red]")
        table.add_row("Errored", f"[yellow]{summary['errored']}[/yellow]")
        table.add_row("Score", f"[bold]{summary['score']}[/bold]")
        if summary.get("memory_score") is not None:
            scored = (summary.get("memory") or {}).get("scored", 0)
            table.add_row(
                "Memory score",
                f"{summary['memory_score']:.1%} ({scored}/{summary['total']})",
            )
        table.add_row("Avg tool calls", f"{summary['avg_tool_calls']:.1f}")
        table.add_row("Avg duration", f"{summary['avg_duration_sec']:.0f}s")

        console.print(table)

        if summary["passed_tasks"]:
            console.print(
                f"\n[green]Passed:[/green] {', '.join(summary['passed_tasks'])}"
            )
        if summary["errored_tasks"]:
            console.print(
                f"[yellow]Errors:[/yellow] {', '.join(summary['errored_tasks'])}"
            )
        if summary["error_breakdown"]:
            console.print(
                f"[yellow]Error breakdown:[/yellow] {summary['error_breakdown']}"
            )

    @app.command(hidden=True, deprecated=True)
    def view(
        rollout_dir: Annotated[
            Path,
            typer.Argument(help="Rollout or job directory with trajectories"),
        ],
        port: Annotated[int, typer.Option(help="Server port")] = 8888,
    ) -> None:
        """View a trial trajectory in the browser."""
        from benchflow.trajectories.viewer import serve

        serve(str(rollout_dir), port)

    @app.command(hidden=True, deprecated=True)
    def cleanup(
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="List sandboxes without deleting"),
        ] = False,
        max_age_minutes: Annotated[
            int,
            typer.Option("--max-age", help="Delete sandboxes older than N minutes"),
        ] = 1440,
    ) -> None:
        """Clean up orphaned Daytona sandboxes.

        Lists and deletes sandboxes that were left running after eval runs.
        Only affects sandboxes older than --max-age minutes (default 1440 = 24h).
        """
        from benchflow.cli import main as cli_main

        cli_main._cleanup_daytona_sandboxes(
            dry_run=dry_run, max_age_minutes=max_age_minutes
        )
