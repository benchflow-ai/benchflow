"""Deprecated CLI commands kept hidden for backwards compatibility.

Extracted from cli/main.py per PLAN_V2_impl §13.6 commit 11. All
commands here are ``hidden=True, deprecated=True``. Each is superseded
by a resource-verb command in another cli/* module:

- ``benchflow job``     → ``benchflow eval create --tasks-dir`` / ``--config``
- ``benchflow agents``  → ``benchflow agent list``
- ``benchflow metrics`` → (no replacement yet — backlog)
- ``benchflow view``    → (viewer; no replacement)
- ``benchflow eval``    → ``benchflow eval create --tasks-dir`` / ``--skills-dir``
- ``benchflow cleanup`` → ``benchflow environment list`` + manual cleanup

When all of these are gone, this file goes too.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from benchflow.job import DEFAULT_AGENT, effective_model

console = Console()


def register(app: typer.Typer) -> None:
    @app.command(hidden=True, deprecated=True)
    def job(
        tasks_dir: Annotated[
            Path | None,
            typer.Option("--tasks-dir", "-t", help="Directory of tasks to run"),
        ] = None,
        config_file: Annotated[
            Path | None,
            typer.Option(
                "--config", "-f", help="YAML config file (Harbor or benchflow format)"
            ),
        ] = None,
        agent: Annotated[
            str,
            typer.Option("--agent", "-a", help="Agent name from registry"),
        ] = DEFAULT_AGENT,
        model: Annotated[
            str | None,
            typer.Option("--model", "-m", help="Model to use"),
        ] = None,
        environment: Annotated[
            str,
            typer.Option("--env", "-e", help="Environment: docker or daytona"),
        ] = "docker",
        concurrency: Annotated[
            int,
            typer.Option("--concurrency", "-c", help="Max concurrent tasks"),
        ] = 4,
        max_retries: Annotated[
            int,
            typer.Option("--retries", help="Max retries per task"),
        ] = 0,
        jobs_dir: Annotated[
            str,
            typer.Option("--jobs-dir", "-o", help="Output directory for results"),
        ] = "jobs",
        skills_dir: Annotated[
            Path | None,
            typer.Option(
                "--skills-dir", "-s", help="Skills directory to deploy into sandbox"
            ),
        ] = None,
    ) -> None:
        """Run all tasks in a directory with concurrency and retries.

        Use --config/-f for YAML config, or --tasks-dir/-t for direct invocation.
        """
        from benchflow.job import Job, JobConfig, RetryConfig

        if config_file:
            j = Job.from_yaml(config_file)
        elif tasks_dir:
            j = Job(
                tasks_dir=str(tasks_dir),
                jobs_dir=jobs_dir,
                config=JobConfig(
                    agent=agent,
                    model=effective_model(agent, model),
                    environment=environment,
                    concurrency=concurrency,
                    retry=RetryConfig(max_retries=max_retries),
                    skills_dir=str(skills_dir) if skills_dir else None,
                ),
            )
        else:
            console.print("[red]Either --tasks-dir or --config is required[/red]")
            raise typer.Exit(1)

        result = asyncio.run(j.run())

        console.print(
            f"\n[bold]Score: {result.passed}/{result.total} "
            f"({result.score:.1%})[/bold], "
            f"errors={result.errored}"
        )

    @app.command(hidden=True, deprecated=True)
    def agents() -> None:
        """List available agents."""
        from benchflow.agents.registry import list_agents

        table = Table(title="Registered Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Description")
        table.add_column("Protocol", style="green")
        table.add_column("Requires", style="yellow")

        for agent in list_agents():
            sub_env = (
                agent.subscription_auth.replaces_env if agent.subscription_auth else None
            )
            requires = [
                f"{e} (or login)" if e == sub_env else e for e in agent.requires_env
            ]
            table.add_row(
                agent.name,
                agent.description,
                agent.protocol,
                ", ".join(requires),
            )

        console.print(table)

    @app.command(hidden=True, deprecated=True)
    def metrics(
        jobs_dir: Annotated[
            Path,
            typer.Argument(help="Jobs directory to analyze"),
        ],
        benchmark: Annotated[
            str,
            typer.Option("--benchmark", "-b", help="Benchmark name"),
        ] = "",
        agent: Annotated[
            str,
            typer.Option("--agent", "-a", help="Agent name"),
        ] = "",
        model: Annotated[
            str,
            typer.Option("--model", "-m", help="Model name"),
        ] = "",
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Output as JSON"),
        ] = False,
    ) -> None:
        """Collect and display metrics from a jobs directory."""
        from benchflow.metrics import collect_metrics

        m = collect_metrics(str(jobs_dir), benchmark=benchmark, agent=agent, model=model)
        summary = m.summary()

        if output_json:
            console.print(json.dumps(summary, indent=2))
            return

        table = Table(title=f"Results: {jobs_dir}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")

        table.add_row("Total", str(summary["total"]))
        table.add_row("Passed", f"[green]{summary['passed']}[/green]")
        table.add_row("Failed", f"[red]{summary['failed']}[/red]")
        table.add_row("Errored", f"[yellow]{summary['errored']}[/yellow]")
        table.add_row("Score", f"[bold]{summary['score']}[/bold]")
        table.add_row("Avg tool calls", f"{summary['avg_tool_calls']:.1f}")
        table.add_row("Avg duration", f"{summary['avg_duration_sec']:.0f}s")

        console.print(table)

        if summary["passed_tasks"]:
            console.print(f"\n[green]Passed:[/green] {', '.join(summary['passed_tasks'])}")
        if summary["errored_tasks"]:
            console.print(f"[yellow]Errors:[/yellow] {', '.join(summary['errored_tasks'])}")
        if summary["error_breakdown"]:
            console.print(
                f"[yellow]Error breakdown:[/yellow] {summary['error_breakdown']}"
            )

    @app.command(hidden=True, deprecated=True)
    def view(
        trial_dir: Annotated[
            Path,
            typer.Argument(help="Trial or job directory with trajectories"),
        ],
        port: Annotated[int, typer.Option(help="Server port")] = 8888,
    ) -> None:
        """View a trial trajectory in the browser."""
        from benchflow.trajectories.viewer import serve

        serve(str(trial_dir), port)

    @app.command(hidden=True, deprecated=True)
    def eval(
        tasks_dir: Annotated[
            Path,
            typer.Option("--tasks-dir", "-t", help="Directory of tasks"),
        ],
        skill: Annotated[
            Path | None,
            typer.Option(
                "--skill", help="Path to SKILL.md (parent dir used as skills_dir)"
            ),
        ] = None,
        skills_dir: Annotated[
            Path | None,
            typer.Option(
                "--skills-dir", "-s", help="Skills directory for agent discovery"
            ),
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
            typer.Option("--env", "-e", help="Environment: docker or daytona"),
        ] = "docker",
        concurrency: Annotated[
            int,
            typer.Option("--concurrency", "-c", help="Max concurrent tasks"),
        ] = 4,
        jobs_dir: Annotated[
            str,
            typer.Option("--jobs-dir", "-o", help="Output directory"),
        ] = "jobs",
    ) -> None:
        """Evaluate a skill against multiple tasks.

        Runs all tasks in --tasks-dir with the given skill and produces a summary.
        Simpler than `benchflow job` — designed for skill evaluation workflows.

        Examples:
            benchflow eval -t tasks/ --skill skills/gws/SKILL.md -a claude-agent-acp -e daytona
            benchflow eval -t tasks/ --skills-dir skills/ -a gemini -e daytona -c 64
        """
        from benchflow.job import Job, JobConfig

        effective_skills = (
            str(skills_dir) if skills_dir else (str(skill.parent) if skill else None)
        )

        j = Job(
            tasks_dir=str(tasks_dir),
            jobs_dir=jobs_dir,
            config=JobConfig(
                agent=agent,
                model=effective_model(agent, model),
                environment=environment,
                concurrency=concurrency,
                skills_dir=effective_skills,
            ),
        )

        result = asyncio.run(j.run())

        console.print("\n[bold]Skill Eval Results[/bold]")
        if skill:
            console.print(f"  Skill: {skill}")
        if skills_dir:
            console.print(f"  Skills dir: {skills_dir}")
        console.print(
            f"  Score: [bold]{result.passed}/{result.total} "
            f"({result.score:.1%})[/bold], errors={result.errored}"
        )
        console.print(f"  Elapsed: {result.elapsed_sec:.0f}s")

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
        try:
            from daytona import Daytona
        except ImportError:
            console.print("[red]daytona SDK not installed[/red]")
            raise typer.Exit(1) from None

        d = Daytona()
        now = datetime.now(UTC)
        page = 1
        total_deleted = 0
        total_found = 0
        total_skipped = 0

        while True:
            result = d.list(page=page, limit=100)
            if not result.items:
                break
            total_found += len(result.items)
            for sb in result.items:
                if not sb.created_at:
                    continue
                created_at = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
                age_minutes = (now - created_at).total_seconds() / 60
                if age_minutes < max_age_minutes:
                    total_skipped += 1
                    if dry_run:
                        console.print(
                            f"  [dim]{sb.id}[/dim] state={sb.state} age={age_minutes:.0f}m [green](skip)[/green]"
                        )
                    continue
                if dry_run:
                    console.print(
                        f"  [dim]{sb.id}[/dim] state={sb.state} age={age_minutes:.0f}m [red](delete)[/red]"
                    )
                else:
                    try:
                        d.delete(sb)
                        total_deleted += 1
                    except Exception as e:
                        console.print(
                            f"  [yellow]Failed to delete {sb.id}: {e}[/yellow]"
                        )
            if len(result.items) < 100:
                break
            page += 1

        if dry_run:
            console.print(
                f"\n[bold]{total_found} sandboxes found, "
                f"{total_found - total_skipped} older than {max_age_minutes}m[/bold] "
                f"(use without --dry-run to delete)"
            )
        else:
            console.print(
                f"\n[bold green]{total_deleted} sandboxes deleted[/bold green] "
                f"({total_skipped} skipped, younger than {max_age_minutes}m)"
            )
