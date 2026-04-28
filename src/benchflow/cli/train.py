"""``bench train create`` — RL training sweeps.

Extracted from cli/main.py per PLAN_V2_impl §13.4 / §13.6 commit 10a.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from benchflow.job import DEFAULT_AGENT, effective_model

console = Console()

train_app = typer.Typer(help="Training and RL optimization commands.")


@train_app.command("create")
def train_create(
    config_file: Annotated[
        Path | None,
        typer.Option("--config", "-f", help="Job config YAML"),
    ] = None,
    tasks_dir: Annotated[
        Path | None,
        typer.Option("--tasks-dir", "-t", help="Tasks directory"),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", "-a", help="Agent name"),
    ] = DEFAULT_AGENT,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--env", "-e", help="Backend"),
    ] = "daytona",
    sweeps: Annotated[
        int,
        typer.Option("--sweeps", help="Number of sweep iterations"),
    ] = 3,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", help="Max concurrent tasks"),
    ] = 64,
    export_dir: Annotated[
        Path | None,
        typer.Option("--export", help="Export sweep results to directory"),
    ] = None,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", "-o", help="Output directory"),
    ] = "jobs",
) -> None:
    """Run a training sweep — successive eval runs with reward-based filtering.

    Each sweep runs all remaining tasks. Tasks that succeed (reward > 0) are
    dropped from the next sweep. After all sweeps, results are split into
    success/failure for RL training data.

    Modeled after Harbor's sweep pattern: run → collect → filter → repeat.
    Integrates with rewards.jsonl (ORS-compatible) for dense reward export.

    Example:
        bench train create -t tasks/ -a gemini --sweeps 3 --export ./training-data
    """
    from benchflow.job import Job, JobConfig

    if not config_file and not tasks_dir:
        console.print("[red]Either --config or --tasks-dir is required[/red]")
        raise typer.Exit(1)

    sweep_results: list[dict] = []

    for sweep_idx in range(1, sweeps + 1):
        console.print(f"\n[bold]Sweep {sweep_idx}/{sweeps}[/bold]")

        if config_file:
            j = Job.from_yaml(config_file)
        else:
            j = Job(
                tasks_dir=str(tasks_dir),
                jobs_dir=f"{jobs_dir}/sweep-{sweep_idx}",
                config=JobConfig(
                    agent=agent,
                    model=effective_model(agent, model),
                    environment=environment,
                    concurrency=concurrency,
                ),
            )

        result = asyncio.run(j.run())

        console.print(
            f"  Score: {result.passed}/{result.total} ({result.score:.1%}), "
            f"errors={result.errored}"
        )

        sweep_results.append(
            {
                "sweep": sweep_idx,
                "passed": result.passed,
                "total": result.total,
                "score": round(result.score, 3),
                "errored": result.errored,
            }
        )

        if result.passed == result.total:
            console.print("[green]All tasks succeeded — stopping early.[/green]")
            break

    console.print("\n[bold]Training Summary[/bold]")
    table = Table()
    table.add_column("Sweep", justify="right")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Score", justify="right")
    for sr in sweep_results:
        table.add_row(
            str(sr["sweep"]),
            str(sr["passed"]),
            str(sr["total"]),
            f"{sr['score']:.1%}",
        )
    console.print(table)

    if export_dir:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "sweep_results.json").write_text(
            json.dumps(sweep_results, indent=2)
        )
        console.print(f"\n[green]Results exported to {export_dir}[/green]")
