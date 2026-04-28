"""``bench skills {list,install,eval}`` — skill discovery and evaluation.

Extracted from cli/main.py per PLAN_V2_impl §13.4 / §13.6 commit 8a.
main.py mounts ``skills_app`` via ``app.add_typer(skills_app, name="skills")``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

skills_app = typer.Typer(help="Skill discovery, installation, and evaluation.")


@skills_app.command("list")
def skills_list(
    directory: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Skills directory to scan"),
    ] = None,
) -> None:
    """List discovered skills."""
    from benchflow.skill_registry import DEFAULT_SKILLS_DIR, discover_skills

    search_dirs = (
        [directory]
        if directory
        else [DEFAULT_SKILLS_DIR, Path(".claude/skills"), Path("skills")]
    )
    found = discover_skills(*search_dirs)
    if not found:
        console.print(
            "No skills found. Install with: benchflow skills install owner/repo@skill-name"
        )
        return

    table = Table(title="Discovered Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Description")
    table.add_column("Path", style="dim")

    for s in found:
        table.add_row(s.name, s.version or "-", s.description[:60], str(s.path))

    console.print(table)


@skills_app.command("install", hidden=True, deprecated=True)
def skills_install(
    spec: Annotated[
        str,
        typer.Argument(help="Skill spec (e.g. anthropics/skills@find-skills)"),
    ],
    directory: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Target directory"),
    ] = None,
) -> None:
    """Install a skill from the registry."""
    from benchflow.skill_registry import DEFAULT_SKILLS_DIR, install_skill

    target = directory or DEFAULT_SKILLS_DIR
    result = install_skill(spec, target_dir=target)
    if result:
        console.print(f"[green]Installed:[/green] {result}")
    else:
        console.print(f"[red]Failed to install {spec}[/red]")
        raise typer.Exit(1)


@skills_app.command("eval")
def skills_eval(
    skill_dir: Annotated[
        Path,
        typer.Argument(help="Path to skill directory containing evals/evals.json"),
    ],
    agent: Annotated[
        list[str],
        typer.Option("--agent", "-a", help="Agent(s) to evaluate (repeatable)"),
    ] = ["claude-agent-acp"],
    model: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Model(s) (matched 1:1 with agents)"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--env", "-e", help="Environment: docker or daytona"),
    ] = "docker",
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", "-c", help="Max concurrent tasks"),
    ] = 1,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", "-o", help="Output directory for results"),
    ] = "jobs",
    no_baseline: Annotated[
        bool,
        typer.Option("--no-baseline", help="Skip baseline (without-skill) runs"),
    ] = False,
    export_gepa: Annotated[
        bool,
        typer.Option("--export-gepa", help="Export GEPA-compatible traces"),
    ] = False,
) -> None:
    """Evaluate a skill using its evals/evals.json test cases.

    Generates ephemeral tasks from the skill's eval dataset, runs each agent
    with and without the skill installed, and reports the lift.

    Examples:
        benchflow skills eval ./my-skill/ -a claude-agent-acp
        benchflow skills eval ./my-skill/ -a claude-agent-acp -a codex-acp -e daytona -c 4
        benchflow skills eval ./my-skill/ -a claude-agent-acp --no-baseline --export-gepa
    """
    from benchflow.skill_eval import SkillEvaluator, export_gepa_traces

    if not (skill_dir / "evals" / "evals.json").exists():
        console.print(
            f"[red]No evals/evals.json found in {skill_dir}[/red]\n"
            f"Create one with test cases. See: benchflow skills eval --help"
        )
        raise typer.Exit(1)

    evaluator = SkillEvaluator(skill_dir)
    console.print(
        f"[bold]Skill eval:[/bold] {evaluator.dataset.skill_name} "
        f"({len(evaluator.dataset.cases)} cases)"
    )
    console.print(f"  Agents: {', '.join(agent)}")
    console.print(f"  Environment: {environment}")
    if no_baseline:
        console.print("  [dim]Baseline skipped (--no-baseline)[/dim]")

    result = asyncio.run(
        evaluator.run(
            agents=agent,
            models=model,
            environment=environment,
            jobs_dir=jobs_dir,
            no_baseline=no_baseline,
            concurrency=concurrency,
        )
    )

    table = Table(title=f"Skill Eval: {result.skill_name}")
    table.add_column("Agent", style="cyan")
    table.add_column("Mode", style="dim")
    table.add_column("Score")
    table.add_column("Avg Reward")

    for row in result.summary_table():
        style = "bold green" if row["mode"] == "LIFT" else None
        table.add_row(
            row["agent"], row["mode"], row["score"], row["avg_reward"], style=style
        )

    console.print(table)

    if export_gepa:
        gepa_dir = export_gepa_traces(
            result,
            evaluator.dataset,
            output_dir=f"{jobs_dir}/skill-eval/{result.skill_name}/gepa",
        )
        console.print(f"[green]GEPA traces exported to {gepa_dir}[/green]")
