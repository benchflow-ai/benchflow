"""``bench eval {create,list}`` — single-task and batch evaluations.

Extracted from cli/main.py per PLAN_V2_impl §13.4 / §13.6 commit 9.
This file replaces the prior orphaned cli/eval.py — the new home is
the live entry point for ``bench eval create`` and ``bench eval list``.
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

eval_app = typer.Typer(help="Evaluation commands.")


@eval_app.command("create")
def eval_create(
    config_file: Annotated[
        Path | None,
        typer.Option("--config", "-f", help="YAML config file"),
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
        typer.Option("--env", "-e", help="Backend: docker or daytona"),
    ] = "docker",
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", help="Max concurrent tasks"),
    ] = 4,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", "-o", help="Output directory"),
    ] = "jobs",
    sandbox_user: Annotated[
        str | None,
        typer.Option("--sandbox-user", help="Sandbox user (null for root)"),
    ] = "agent",
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills-dir", "-s", help="Skills directory to deploy"),
    ] = None,
) -> None:
    """Run an evaluation — single task or batch."""
    from benchflow.job import Job, JobConfig

    if config_file:
        j = Job.from_yaml(config_file)
        result = asyncio.run(j.run())
        console.print(
            f"\n[bold]Score: {result.passed}/{result.total} "
            f"({result.score:.1%})[/bold], errors={result.errored}"
        )
    elif tasks_dir:
        eff_model = effective_model(agent, model)
        # Smart detection: if tasks_dir has task.toml, it's a single task
        if (tasks_dir / "task.toml").exists():
            from benchflow.trial import Scene, Trial, TrialConfig

            config = TrialConfig(
                task_path=tasks_dir,
                scenes=[Scene.single(agent=agent, model=eff_model)],
                environment=environment,
                sandbox_user=sandbox_user,
                jobs_dir=jobs_dir,
                agent=agent,
                model=eff_model,
                skills_dir=str(skills_dir) if skills_dir else None,
            )

            async def _run():
                trial = await Trial.create(config)
                return await trial.run()

            run_result = asyncio.run(_run())
            reward = (run_result.rewards or {}).get("reward")
            console.print(f"\n[bold]Task:[/bold] {tasks_dir.name}")
            console.print(f"[bold]Agent:[/bold] {agent} ({eff_model or 'no model'})")
            console.print(f"[bold]Reward:[/bold] {reward}")
            console.print(f"[bold]Tool calls:[/bold] {run_result.n_tool_calls}")
            if run_result.error:
                console.print(f"[red]Error:[/red] {run_result.error}")
        else:
            # Directory of tasks — batch run
            j = Job(
                tasks_dir=str(tasks_dir),
                jobs_dir=jobs_dir,
                config=JobConfig(
                    agent=agent,
                    model=eff_model,
                    environment=environment,
                    concurrency=concurrency,
                    sandbox_user=sandbox_user,
                    skills_dir=str(skills_dir) if skills_dir else None,
                ),
            )
            result = asyncio.run(j.run())
            console.print(
                f"\n[bold]Score: {result.passed}/{result.total} "
                f"({result.score:.1%})[/bold], errors={result.errored}"
            )
    else:
        console.print("[red]Either --config or --tasks-dir is required[/red]")
        raise typer.Exit(1)


@eval_app.command("list")
def eval_list(
    jobs_dir: Annotated[
        Path,
        typer.Argument(help="Jobs directory to list"),
    ] = Path("jobs"),
) -> None:
    """List completed evaluations."""
    if not jobs_dir.exists():
        console.print(f"[yellow]No jobs directory: {jobs_dir}[/yellow]")
        return

    table = Table(title="Evaluations")
    table.add_column("Job", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Summary")

    for d in sorted(jobs_dir.iterdir()):
        if not d.is_dir():
            continue
        summary = d / "summary.json"
        if summary.exists():
            data = json.loads(summary.read_text())
            table.add_row(
                d.name,
                str(data.get("total", "?")),
                f"{data.get('passed', '?')}/{data.get('total', '?')} ({data.get('score', '?')})",
            )
        else:
            sub_count = sum(1 for s in d.iterdir() if s.is_dir())
            table.add_row(d.name, str(sub_count), "[dim]no summary[/dim]")

    console.print(table)
