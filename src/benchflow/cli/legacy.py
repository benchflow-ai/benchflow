"""Deprecated top-level benchflow commands (hidden in ``--help``).

These predate the 0.3 resource-verb subgroups (``eval``/``environment``/
``agent``) and are kept for backwards compatibility only: ``job``, ``agents``,
``metrics``, ``view``, ``eval`` (legacy skill eval), and ``cleanup``. Each is
``hidden=True, deprecated=True``.

Registered onto the top-level app by :func:`register_legacy`; ``cli/main.py``
only wires the call. ``cleanup`` resolves the Daytona helpers through the
``benchflow.cli.main`` module so tests that monkeypatch those names keep working.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from benchflow.cli._options import (
    AgentOption,
    ConcurrencyOption,
    JobsDirOption,
    ModelOption,
    SandboxOption,
    SkillModeOption,
)
from benchflow.cli._shared import (
    _REQUIRES_AUTH_NOTE,
    _format_requires,
    _report_eval_result,
    console,
)
from benchflow.evaluation import DEFAULT_AGENT, effective_model
from benchflow.skill_policy import SKILL_MODE_NO_SKILL


def register_legacy(app: typer.Typer) -> None:
    """Attach the deprecated top-level commands to the benchflow app."""

    @app.command(hidden=True, deprecated=True)
    def job(
        tasks_dir: Annotated[
            Path | None,
            typer.Option("--tasks-dir", help="Directory of tasks to run"),
        ] = None,
        config_file: Annotated[
            Path | None,
            typer.Option(
                "--config", help="YAML config file (benchflow or legacy format)"
            ),
        ] = None,
        agent: Annotated[
            str,
            typer.Option("--agent", help="Agent name from registry"),
        ] = DEFAULT_AGENT,
        model: Annotated[
            str | None,
            typer.Option("--model", help="Model to use"),
        ] = None,
        environment: SandboxOption = "docker",
        concurrency: ConcurrencyOption = 4,
        max_retries: Annotated[
            int,
            typer.Option("--retries", help="Max retries per task"),
        ] = 0,
        jobs_dir: JobsDirOption = "jobs",
        skills_dir: Annotated[
            Path | None,
            typer.Option(
                "--skills-dir", help="Skills directory to deploy into sandbox"
            ),
        ] = None,
        skill_mode: SkillModeOption = SKILL_MODE_NO_SKILL,
    ) -> None:
        """Run all tasks in a directory with concurrency and retries.

        Use --config for YAML config, or --tasks-dir for direct invocation.
        """
        from benchflow.evaluation import Evaluation, EvaluationConfig, RetryConfig

        if config_file:
            j = Evaluation.from_yaml(config_file)
        elif tasks_dir:
            j = Evaluation(
                tasks_dir=str(tasks_dir),
                jobs_dir=jobs_dir,
                config=EvaluationConfig(
                    agent=agent,
                    model=effective_model(agent, model),
                    environment=environment,
                    concurrency=concurrency,
                    retry=RetryConfig(max_retries=max_retries),
                    skills_dir=str(skills_dir) if skills_dir else None,
                    skill_mode=skill_mode,
                ),
            )
        else:
            console.print("[red]Either --tasks-dir or --config is required[/red]")
            raise typer.Exit(1)

        result = asyncio.run(j.run())

        _report_eval_result(result)

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
            table.add_row(
                agent.name,
                agent.description,
                agent.protocol,
                _format_requires(agent),
            )

        console.print(table)
        console.print(f"[dim]{_REQUIRES_AUTH_NOTE}[/dim]")

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
    def eval(
        tasks_dir: Annotated[
            Path,
            typer.Option("--tasks-dir", help="Directory of tasks"),
        ],
        skill: Annotated[
            Path | None,
            typer.Option(
                "--skill", help="Path to SKILL.md (parent dir used as skills_dir)"
            ),
        ] = None,
        skills_dir: Annotated[
            Path | None,
            typer.Option("--skills-dir", help="Skills directory for agent discovery"),
        ] = None,
        agent: AgentOption = DEFAULT_AGENT,
        model: ModelOption = None,
        environment: SandboxOption = "docker",
        concurrency: ConcurrencyOption = 4,
        jobs_dir: Annotated[
            str,
            typer.Option("--jobs-dir", help="Output directory"),
        ] = "jobs",
        skill_mode: SkillModeOption = SKILL_MODE_NO_SKILL,
    ) -> None:
        """Evaluate a skill against multiple tasks.

        Runs all tasks in --tasks-dir with the given skill and produces a summary.
        Simpler than `benchflow job` — designed for skill evaluation workflows.

        Examples:
            benchflow eval --tasks-dir tasks/ --skill skills/gws/SKILL.md --agent claude-agent-acp --sandbox daytona
            benchflow eval --tasks-dir tasks/ --skills-dir skills/ --agent gemini --sandbox daytona --concurrency 64
        """
        from benchflow.evaluation import Evaluation, EvaluationConfig

        # Use --skill as skills_dir if --skills-dir not provided
        effective_skills = (
            str(skills_dir) if skills_dir else (str(skill.parent) if skill else None)
        )

        j = Evaluation(
            tasks_dir=str(tasks_dir),
            jobs_dir=jobs_dir,
            config=EvaluationConfig(
                agent=agent,
                model=effective_model(agent, model),
                environment=environment,
                concurrency=concurrency,
                skills_dir=effective_skills,
                skill_mode=skill_mode,
            ),
        )

        result = asyncio.run(j.run())

        # Summary
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
        from benchflow.cli import main as cli_main

        cli_main._cleanup_daytona_sandboxes(
            dry_run=dry_run, max_age_minutes=max_age_minutes
        )
