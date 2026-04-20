"""benchflow CLI — agent benchmarking framework."""

import asyncio
import json
from datetime import UTC
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from benchflow.job import DEFAULT_AGENT, DEFAULT_MODEL

console = Console()

app = typer.Typer(
    name="benchflow",
    help="ACP-native agent benchmarking framework.",
    no_args_is_help=True,
)


@app.command(hidden=True, deprecated=True)
def run(
    task_dir: Annotated[
        Path,
        typer.Option("--task-dir", "-t", help="Task directory"),
    ],
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
        typer.Option(
            "--skills-dir", "-s", help="Skills directory to deploy into sandbox"
        ),
    ] = None,
    sandbox_user: Annotated[
        str | None,
        typer.Option(
            "--sandbox-user",
            help="Run agent as non-root user (default: 'agent'). Pass 'none' for root.",
        ),
    ] = "agent",
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
    # CLI only ever passes plain strings; cast to widen for the SDK's
    # `list[str | None] | None` API (None entries mean "use default").
    result = asyncio.run(
        sdk.run(
            task_path=task_dir,
            agent=agent,
            model=model,
            prompts=cast("list[str | None] | None", prompt),
            agent_env=parsed_env,
            jobs_dir=jobs_dir,
            environment=environment,
            skills_dir=str(skills_dir) if skills_dir else None,
            sandbox_user=sandbox_user,
        )
    )

    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Task:[/green] {result.task_name}")
    console.print(f"[green]Agent:[/green] {result.agent_name}")
    console.print(f"[green]Rewards:[/green] {result.rewards}")
    console.print(f"[green]Tool calls:[/green] {result.n_tool_calls}")


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
                model=model or DEFAULT_MODEL,
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
        console.print(f"[yellow]Errors:[/yellow] {', '.join(summary['errored_tasks'])}")
    if summary["error_breakdown"]:
        console.print(f"[yellow]Error breakdown:[/yellow] {summary['error_breakdown']}")


@app.command(hidden=True, deprecated=True)
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
        typer.Option("--skills-dir", "-s", help="Skills directory for agent discovery"),
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

    # Use --skill as skills_dir if --skills-dir not provided
    effective_skills = (
        str(skills_dir) if skills_dir else (str(skill.parent) if skill else None)
    )

    j = Job(
        tasks_dir=str(tasks_dir),
        jobs_dir=jobs_dir,
        config=JobConfig(
            agent=agent,
            model=model or DEFAULT_MODEL,
            environment=environment,
            concurrency=concurrency,
            skills_dir=effective_skills,
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


skills_app = typer.Typer(help="Skill discovery, installation, and evaluation.")
app.add_typer(skills_app, name="skills")


@skills_app.command("list")
def skills_list(
    directory: Annotated[
        Path | None,
        typer.Option("--dir", "-d", help="Skills directory to scan"),
    ] = None,
) -> None:
    """List discovered skills."""
    from benchflow.skills import DEFAULT_SKILLS_DIR, discover_skills

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
    from benchflow.skills import DEFAULT_SKILLS_DIR, install_skill

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

    # Display results
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

    # GEPA export
    if export_gepa:
        gepa_dir = export_gepa_traces(
            result,
            evaluator.dataset,
            output_dir=f"{jobs_dir}/skill-eval/{result.skill_name}/gepa",
        )
        console.print(f"[green]GEPA traces exported to {gepa_dir}[/green]")


tasks_app = typer.Typer(help="Task authoring commands")
app.add_typer(tasks_app, name="tasks")


@tasks_app.command("init")
def tasks_init(
    name: Annotated[str, typer.Argument(help="Task name")],
    parent_dir: Annotated[
        Path,
        typer.Option("--dir", "-p", help="Parent directory (default: tasks/)"),
    ] = Path("tasks"),
    no_pytest: Annotated[
        bool, typer.Option("--no-pytest", help="Skip pytest template")
    ] = False,
    no_solution: Annotated[
        bool, typer.Option("--no-solution", help="Skip solution template")
    ] = False,
) -> None:
    """Scaffold a new benchmark task."""
    from benchflow.tasks import init_task

    try:
        task_dir = init_task(
            name, parent_dir=parent_dir, no_pytest=no_pytest, no_solution=no_solution
        )
        console.print(f"[green]Created:[/green] {task_dir}/")
        console.print(
            "  task.toml, instruction.md, environment/Dockerfile, tests/test.sh"
        )
        if not no_solution:
            console.print("  solution/solve.sh")
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@tasks_app.command("check")
def tasks_check(
    task_dir: Annotated[Path, typer.Argument(help="Path to task directory")],
) -> None:
    """Validate a task directory structure."""
    from benchflow.tasks import check_task

    issues = check_task(task_dir)
    if not issues:
        console.print(f"[green]✓[/green] {task_dir.name} — valid")
    else:
        console.print(f"[red]✗[/red] {task_dir.name} — {len(issues)} issue(s):")
        for issue in issues:
            console.print(f"  [yellow]→[/yellow] {issue}")
        raise typer.Exit(1)


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
    from datetime import datetime

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
            # Daytona's created_at is an ISO-8601 string (with optional Z suffix)
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
                    console.print(f"  [yellow]Failed to delete {sb.id}: {e}[/yellow]")
        if len(result.items) < 100:
            break
        page += 1

    if dry_run:
        console.print(
            f"\n[bold]{total_found} sandboxes found, {total_found - total_skipped} older than {max_age_minutes}m[/bold] (use without --dry-run to delete)"
        )
    else:
        console.print(
            f"\n[bold green]{total_deleted} sandboxes deleted[/bold green] ({total_skipped} skipped, younger than {max_age_minutes}m)"
        )


# ── Resource-verb subgroups (0.3 CLI) ────────────────────────────────────────

agent_app = typer.Typer(help="Agent management commands.")
app.add_typer(agent_app, name="agent")


@agent_app.command("list")
def agent_list() -> None:
    """List all registered agents."""
    from benchflow.agents.registry import list_agents

    table = Table(title="Registered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Protocol", style="green")
    table.add_column("Requires", style="yellow")

    for a in list_agents():
        sub_env = a.subscription_auth.replaces_env if a.subscription_auth else None
        requires = [f"{e} (or login)" if e == sub_env else e for e in a.requires_env]
        table.add_row(a.name, a.description, a.protocol, ", ".join(requires))

    console.print(table)


@agent_app.command("show")
def agent_show(
    name: Annotated[str, typer.Argument(help="Agent name")],
) -> None:
    """Show details for a registered agent."""
    from benchflow.agents.registry import AGENTS

    cfg = AGENTS.get(name)
    if not cfg:
        console.print(f"[red]Unknown agent: {name}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]{cfg.name}[/bold]")
    console.print(f"  Description: {cfg.description}")
    console.print(f"  Protocol:    {cfg.protocol}")
    console.print(f"  Launch:      {cfg.launch_cmd}")
    console.print(f"  Requires:    {', '.join(cfg.requires_env) or '(none)'}")
    if cfg.subscription_auth:
        console.print(
            f"  Auth:        subscription via {cfg.subscription_auth.detect_file}"
        )


eval_app = typer.Typer(help="Evaluation commands.")
app.add_typer(eval_app, name="eval")


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
) -> None:
    """Run an evaluation — batch of tasks with scoring."""
    from benchflow.job import Job, JobConfig

    if config_file:
        j = Job.from_yaml(config_file)
    elif tasks_dir:
        j = Job(
            tasks_dir=str(tasks_dir),
            jobs_dir=jobs_dir,
            config=JobConfig(
                agent=agent,
                model=model or DEFAULT_MODEL,
                environment=environment,
                concurrency=concurrency,
                sandbox_user=sandbox_user,
            ),
        )
    else:
        console.print("[red]Either --config or --tasks-dir is required[/red]")
        raise typer.Exit(1)

    result = asyncio.run(j.run())
    console.print(
        f"\n[bold]Score: {result.passed}/{result.total} "
        f"({result.score:.1%})[/bold], errors={result.errored}"
    )


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


train_app = typer.Typer(help="Training and RL optimization commands.")
app.add_typer(train_app, name="train")


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
                    model=model or DEFAULT_MODEL,
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


env_app = typer.Typer(help="Environment management commands.")
app.add_typer(env_app, name="environment")


@env_app.command("create")
def environment_create(
    task_dir: Annotated[
        Path,
        typer.Argument(help="Task directory with task.toml + environment/Dockerfile"),
    ],
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="Backend: docker or daytona"),
    ] = "daytona",
) -> None:
    """Create an environment from a task directory (does not start it)."""
    from benchflow.runtime import Environment

    env = Environment.from_task(task_dir, backend=backend)
    console.print(f"[green]Environment created:[/green] {env}")
    console.print(f"  Task:    {env.task_path}")
    console.print(f"  Backend: {env.backend}")
    console.print(
        "  Use [cyan]bench environment start[/cyan] to launch, or pass to [cyan]bf.run()[/cyan]"
    )


@env_app.command("list")
def environment_list() -> None:
    """List active Daytona sandboxes."""
    from datetime import datetime

    try:
        from daytona import Daytona
    except ImportError:
        console.print("[red]daytona SDK not installed[/red]")
        raise typer.Exit(1) from None

    d = Daytona()
    table = Table(title="Active Sandboxes")
    table.add_column("ID", style="cyan")
    table.add_column("State", style="green")
    table.add_column("Age")
    table.add_column("Target")

    page = 1
    now = datetime.now(UTC)
    total = 0
    while True:
        result = d.list(page=page, limit=50)
        if not result.items:
            break
        for sb in result.items:
            total += 1
            age = ""
            if sb.created_at:
                created = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
                mins = (now - created).total_seconds() / 60
                age = f"{mins:.0f}m"
            target = getattr(sb, "target", "") or ""
            table.add_row(sb.id[:12] + "…", str(sb.state), age, str(target)[:40])
        if len(result.items) < 50:
            break
        page += 1

    console.print(table)
    console.print(f"\n[bold]{total} sandbox(es)[/bold]")


if __name__ == "__main__":
    app()
