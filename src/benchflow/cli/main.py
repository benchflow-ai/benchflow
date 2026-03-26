"""benchflow CLI — agent benchmarking framework."""

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

app = typer.Typer(
    name="benchflow",
    help="ACP-native agent benchmarking framework.",
    no_args_is_help=True,
)


@app.command()
def run(
    task_dir: Annotated[
        Path,
        typer.Option("--task-dir", "-t", help="Task directory"),
    ],
    agent: Annotated[
        str,
        typer.Option("--agent", "-a", help="Agent name from registry"),
    ] = "claude-agent-acp",
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model to use"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--env", "-e", help="Environment: docker or daytona"),
    ] = "docker",
    prompt: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt", "-p", help="Prompt(s) to send (default: instruction.md)"
        ),
    ] = None,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", "-o", help="Output directory for results"),
    ] = "jobs",
    agent_env: Annotated[
        list[str] | None,
        typer.Option("--ae", help="Agent env var (KEY=VALUE)"),
    ] = None,
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills-dir", "-s", help="Skills directory to deploy into sandbox"),
    ] = None,
) -> None:
    """Run a single task with an ACP agent."""
    from benchflow.sdk import SDK

    parsed_env: dict[str, str] = {}
    for entry in agent_env or []:
        if "=" not in entry:
            console.print(f"[red]Invalid env var: {entry}[/red]")
            raise typer.Exit(1)
        key, value = entry.split("=", 1)
        parsed_env[key] = value

    sdk = SDK()
    result = asyncio.run(
        sdk.run(
            task_path=task_dir,
            agent=agent,
            model=model,
            prompts=prompt,
            agent_env=parsed_env,
            jobs_dir=jobs_dir,
            environment=environment,
            skills_dir=str(skills_dir) if skills_dir else None,
        )
    )

    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Task:[/green] {result.task_name}")
    console.print(f"[green]Agent:[/green] {result.agent_name}")
    console.print(f"[green]Rewards:[/green] {result.rewards}")
    console.print(f"[green]Tool calls:[/green] {result.n_tool_calls}")


@app.command()
def job(
    tasks_dir: Annotated[
        Path | None,
        typer.Option("--tasks-dir", "-t", help="Directory of tasks to run"),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option("--config", "-f", help="YAML config file (Harbor or benchflow format)"),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", "-a", help="Agent name from registry"),
    ] = "claude-agent-acp",
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
        typer.Option("--skills-dir", "-s", help="Skills directory to deploy into sandbox"),
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
                model=model or "claude-haiku-4-5-20251001",
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


@app.command()
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
            ", ".join(agent.requires_env),
        )

    console.print(table)


@app.command()
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
    table.add_row("Avg tool calls", f"{summary['avg_tool_calls']:.1f}")
    table.add_row("Avg duration", f"{summary['avg_duration_sec']:.0f}s")

    console.print(table)

    if summary["passed_tasks"]:
        console.print(f"\n[green]Passed:[/green] {', '.join(summary['passed_tasks'])}")
    if summary["errored_tasks"]:
        console.print(
            f"[yellow]Errors:[/yellow] {', '.join(summary['errored_tasks'])}"
        )
    if summary["error_breakdown"]:
        console.print(f"[yellow]Error breakdown:[/yellow] {summary['error_breakdown']}")


@app.command()
def view(
    trial_dir: Annotated[
        Path,
        typer.Argument(help="Trial or job directory with trajectories"),
    ],
    port: Annotated[int, typer.Option(help="Server port")] = 8888,
) -> None:
    """View a trial trajectory in the browser."""
    from benchflow.viewer import serve

    serve(str(trial_dir), port)


@app.command()
def eval(
    tasks_dir: Annotated[
        Path,
        typer.Option("--tasks-dir", "-t", help="Directory of tasks"),
    ],
    skill: Annotated[
        Path | None,
        typer.Option("--skill", help="Path to SKILL.md to inject into prompt"),
    ] = None,
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills-dir", "-s", help="Skills directory for agent discovery"),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", "-a", help="Agent name"),
    ] = "claude-agent-acp",
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

    # Build prompts — if --skill provided, prepend to instruction.md
    prompts = None
    if skill:
        skill_content = skill.read_text()
        # Skill content becomes the first prompt context
        prompts = [None]  # Will be replaced with instruction.md + skill prefix

    j = Job(
        tasks_dir=str(tasks_dir),
        jobs_dir=jobs_dir,
        config=JobConfig(
            agent=agent,
            model=model or "claude-haiku-4-5-20251001",
            environment=environment,
            concurrency=concurrency,
            skills_dir=str(skills_dir) if skills_dir else None,
        ),
    )

    result = asyncio.run(j.run())

    # Summary
    console.print(f"\n[bold]Skill Eval Results[/bold]")
    if skill:
        console.print(f"  Skill: {skill}")
    if skills_dir:
        console.print(f"  Skills dir: {skills_dir}")
    console.print(
        f"  Score: [bold]{result.passed}/{result.total} "
        f"({result.score:.1%})[/bold], errors={result.errored}"
    )
    console.print(f"  Elapsed: {result.elapsed_sec:.0f}s")


@app.command()
def skills(
    directory: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Skills directory to scan"),
    ] = None,
    install: Annotated[
        str | None,
        typer.Option("--install", "-i", help="Install skill from skills.sh (e.g. anthropics/skills@find-skills)"),
    ] = None,
) -> None:
    """List or install agent skills."""
    from benchflow.skills import discover_skills, install_skill, DEFAULT_SKILLS_DIR, list_skills_summary

    if install:
        target = directory or DEFAULT_SKILLS_DIR
        result = install_skill(install, target_dir=target)
        if result:
            console.print(f"[green]Installed:[/green] {result}")
        else:
            console.print(f"[red]Failed to install {install}[/red]")
            raise typer.Exit(1)
        return

    search_dirs = [directory] if directory else [DEFAULT_SKILLS_DIR, Path(".claude/skills"), Path("skills")]
    found = discover_skills(*search_dirs)
    if not found:
        console.print("No skills found. Install with: benchflow skills --install owner/repo@skill-name")
        return

    table = Table(title="Discovered Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Description")
    table.add_column("Path", style="dim")

    for s in found:
        table.add_row(s.name, s.version or "-", s.description[:60], str(s.path))

    console.print(table)


if __name__ == "__main__":
    app()
