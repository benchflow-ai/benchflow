"""benchflow CLI — agent benchmarking framework."""

import asyncio
import json
import logging
import os
from datetime import UTC
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from benchflow._dotenv import load_dotenv_env
from benchflow._utils.config import normalize_sandbox_user
from benchflow.agents.registry import parse_agent_spec
from benchflow.cli.trace_import import register_tasks_generate
from benchflow.evaluation import DEFAULT_AGENT, effective_model

# Show progress messages (logger.info) from benchflow internals by default.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

console = Console()

app = typer.Typer(
    name="benchflow",
    help="ACP-native agent benchmarking framework.",
    no_args_is_help=True,
)


def _parse_agent_env(entries: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            console.print(f"[red]Invalid env var: {entry}[/red]")
            raise typer.Exit(1)
        key, value = entry.split("=", 1)
        parsed[key] = value
    return parsed


def _apply_dotenv_to_process_env() -> None:
    """Expose local .env credentials to provider SDKs without overriding env."""
    for key, value in load_dotenv_env().items():
        os.environ.setdefault(key, value)


def _exit_if_run_result_failed(run_result: object) -> None:
    error = getattr(run_result, "error", None)
    verifier_error = getattr(run_result, "verifier_error", None)
    if error:
        console.print(f"[red]Error:[/red] {error}")
    if verifier_error:
        console.print(f"[red]Verifier error:[/red] {verifier_error}")
    if error or verifier_error:
        raise typer.Exit(1)


def _exit_if_evaluation_had_errors(result: object) -> None:
    errored = int(getattr(result, "errored", 0) or 0)
    verifier_errored = int(getattr(result, "verifier_errored", 0) or 0)
    if errored or verifier_errored:
        raise typer.Exit(1)


def _normalize_eval_agent_or_exit(agent_spec: str) -> str:
    protocol, canonical_agent = parse_agent_spec(agent_spec)
    if protocol not in ("acp", "acpx"):
        console.print(f"[red]Unsupported eval agent protocol: {protocol}[/red]")
        raise typer.Exit(1)
    if protocol == "acpx":
        return f"acpx/{canonical_agent}"
    return canonical_agent


def _ensure_daytona_anyio_compat() -> None:
    """Patch the anyio symbol that Daytona 0.176 imports on newer anyio."""
    try:
        import anyio
    except ImportError:
        return

    if hasattr(anyio, "AsyncContextManagerMixin"):
        return

    class _AsyncContextManagerMixin:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            aclose = getattr(self, "aclose", None)
            if aclose is not None:
                await aclose()

    vars(anyio)["AsyncContextManagerMixin"] = _AsyncContextManagerMixin


def _daytona_client_or_exit():
    _ensure_daytona_anyio_compat()
    try:
        from daytona import Daytona
    except ModuleNotFoundError as exc:
        if exc.name == "daytona":
            console.print(
                "[red]daytona SDK not installed[/red]\n"
                "Install it with [cyan]uv sync --extra sandbox-daytona[/cyan]."
            )
        else:
            console.print(f"[red]daytona SDK import failed: {exc}[/red]")
        raise typer.Exit(1) from None
    except Exception as exc:
        console.print(f"[red]daytona SDK import failed: {exc}[/red]")
        raise typer.Exit(1) from None
    return Daytona()


def _cleanup_daytona_sandboxes(dry_run: bool, max_age_minutes: int) -> None:
    """Clean up orphaned Daytona sandboxes."""
    from datetime import datetime

    d = _daytona_client_or_exit()
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


@app.command(hidden=True, deprecated=True)
def run(
    task_dir: Annotated[
        Path | None,
        typer.Argument(help="Local task directory (must contain task.toml)"),
    ] = None,
    source_repo: Annotated[
        str | None,
        typer.Option(
            "--source-repo",
            help="Remote repo as org/repo (e.g. benchflow-ai/skillsbench)",
        ),
    ] = None,
    source_path: Annotated[
        str | None,
        typer.Option(
            "--source-path", help="Subpath within the repo (e.g. tasks/edit-pdf)"
        ),
    ] = None,
    source_ref: Annotated[
        str | None,
        typer.Option("--source-ref", help="Branch or tag to clone (e.g. main)"),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", help="Agent name from registry"),
    ] = DEFAULT_AGENT,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model to use"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--sandbox", help="Sandbox: docker, daytona, or modal"),
    ] = "docker",
    environment_manifest: Annotated[
        Path | None,
        typer.Option(
            "--environment-manifest",
            help="Path to an Environment-plane manifest (environment.toml). "
            "Runs the rollout against the declared stateful environment.",
        ),
    ] = None,
    prompt: Annotated[
        list[str] | None,
        typer.Option("--prompt", help="Prompt(s) to send (default: instruction.md)"),
    ] = None,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", help="Output directory for results"),
    ] = "jobs",
    agent_env: Annotated[
        list[str] | None,
        typer.Option("--agent-env", help="Agent env var (KEY=VALUE)"),
    ] = None,
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills-dir", help="Skills directory to deploy into sandbox"),
    ] = None,
    skill_mode: Annotated[
        str,
        typer.Option(
            "--skill-mode",
            help="Skill mode: default or self-gen",
        ),
    ] = "default",
    skill_creator_dir: Annotated[
        Path | None,
        typer.Option(
            "--skill-creator-dir",
            help="Path to skill-creator or a skills root containing it",
        ),
    ] = None,
    self_gen_no_internet: Annotated[
        bool,
        typer.Option(
            "--self-gen-no-internet",
            help="Disable web tools for the self-generated run",
        ),
    ] = False,
    sandbox_user: Annotated[
        str | None,
        typer.Option(
            "--sandbox-user",
            help="Run agent as non-root user (default: 'agent'). Pass 'none' for root.",
        ),
    ] = "agent",
) -> None:
    """Run a single task with an ACP agent.

    Examples:
        bench run --source-repo benchflow-ai/skillsbench --source-path tasks/edit-pdf
        bench run tasks/edit-pdf --agent gemini --model gemini-3.1-flash-lite-preview
    """
    from benchflow.sdk import SDK

    if source_repo:
        from benchflow._utils.benchmark_repos import resolve_source

        resolved_task_dir = resolve_source(
            source_repo, path=source_path, ref=source_ref
        )
    elif task_dir:
        resolved_task_dir = task_dir
    else:
        console.print("[red]Provide a task directory or --source-repo[/red]")
        raise typer.Exit(1)

    parsed_env = _parse_agent_env(agent_env)
    agent = _normalize_eval_agent_or_exit(agent)
    sandbox_user = normalize_sandbox_user(sandbox_user)

    env_manifest = None
    if environment_manifest is not None:
        from benchflow.environment.manifest import load_manifest

        env_manifest = load_manifest(environment_manifest)

    sdk = SDK()
    # CLI only ever passes plain strings; cast to widen for the SDK's
    # `list[str | None] | None` API (None entries mean "use default").
    result = asyncio.run(
        sdk.run(
            task_path=resolved_task_dir,
            agent=agent,
            model=model,
            prompts=cast("list[str | None] | None", prompt),
            agent_env=parsed_env,
            jobs_dir=jobs_dir,
            environment=environment,
            environment_manifest=env_manifest,
            skills_dir=str(skills_dir) if skills_dir else None,
            sandbox_user=sandbox_user,
            skill_mode=skill_mode,
            skill_creator_dir=str(skill_creator_dir) if skill_creator_dir else None,
            self_gen_no_internet=self_gen_no_internet,
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
        typer.Option("--tasks-dir", help="Directory of tasks to run"),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option("--config", help="YAML config file (benchflow or legacy format)"),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", help="Agent name from registry"),
    ] = DEFAULT_AGENT,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model to use"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--sandbox", help="Sandbox: docker, daytona, or modal"),
    ] = "docker",
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Max concurrent tasks"),
    ] = 4,
    max_retries: Annotated[
        int,
        typer.Option("--retries", help="Max retries per task"),
    ] = 0,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", help="Output directory for results"),
    ] = "jobs",
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills-dir", help="Skills directory to deploy into sandbox"),
    ] = None,
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
        typer.Option("--benchmark", help="Benchmark name"),
    ] = "",
    agent: Annotated[
        str,
        typer.Option("--agent", help="Agent name"),
    ] = "",
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
    agent: Annotated[
        str,
        typer.Option("--agent", help="Agent name"),
    ] = DEFAULT_AGENT,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--sandbox", help="Sandbox: docker, daytona, or modal"),
    ] = "docker",
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Max concurrent tasks"),
    ] = 4,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", help="Output directory"),
    ] = "jobs",
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
        typer.Option("--dir", help="Skills directory to scan"),
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
        typer.Option("--dir", help="Target directory"),
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
        list[str] | None,
        typer.Option("--agent", help="Agent(s) to evaluate (repeatable)"),
    ] = None,
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help="Model(s) (matched 1:1 with agents)"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--sandbox", help="Sandbox: docker, daytona, or modal"),
    ] = "docker",
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Max concurrent tasks"),
    ] = 1,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", help="Output directory for results"),
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
        benchflow skills eval ./my-skill/ --agent claude-agent-acp
        benchflow skills eval ./my-skill/ --agent claude-agent-acp --agent codex-acp --sandbox daytona --concurrency 4
        benchflow skills eval ./my-skill/ --agent claude-agent-acp --no-baseline --export-gepa
    """
    from benchflow.skill_eval import SkillEvaluator, export_gepa_traces

    if agent is None:
        agent = ["claude-agent-acp"]
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

register_tasks_generate(tasks_app)


@tasks_app.command("init")
def tasks_init(
    name: Annotated[str, typer.Argument(help="Task name")],
    parent_dir: Annotated[
        Path,
        typer.Option("--dir", help="Parent directory (default: tasks/)"),
    ] = Path("tasks"),
    no_pytest: Annotated[
        bool, typer.Option("--no-pytest", help="Skip pytest template")
    ] = False,
    no_solution: Annotated[
        bool, typer.Option("--no-solution", help="Skip solution template")
    ] = False,
) -> None:
    """Scaffold a new benchmark task."""
    from benchflow._utils.task_authoring import init_task

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
    from benchflow._utils.task_authoring import check_task

    issues = check_task(task_dir)
    if not issues:
        console.print(f"[green]✓[/green] {task_dir.name} — valid")
    else:
        console.print(f"[red]✗[/red] {task_dir.name} — {len(issues)} issue(s):")
        for issue in issues:
            console.print(f"  [yellow]→[/yellow] {issue}")
        raise typer.Exit(1)


compat_app = typer.Typer(help="Third-party framework compatibility checks.")
app.add_typer(compat_app, name="compat")


@compat_app.command("harbor-registry")
def compat_harbor_registry(
    registry: Annotated[
        str,
        typer.Option(
            "--registry",
            help="Harbor registry JSON URL or local file.",
        ),
    ] = "https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json",
    tasks_per_dataset: Annotated[
        int,
        typer.Option(
            "--tasks-per-dataset",
            help="Number of representative tasks to select per registry dataset.",
            min=1,
        ),
    ] = 2,
    level: Annotated[
        str,
        typer.Option(
            "--level",
            help="Compatibility level to run: inventory or check.",
        ),
    ] = "inventory",
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSONL output path."),
    ] = None,
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", help="Cache directory for sparse clones."),
    ] = Path(".cache/compat/harbor"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Optional cap on selected task refs."),
    ] = None,
) -> None:
    """Inventory or structurally check representative Harbor registry tasks."""
    from benchflow.compat.harbor_registry import (
        check_harbor_registry,
        records_summary,
    )

    try:
        records = check_harbor_registry(
            registry,
            tasks_per_dataset=tasks_per_dataset,
            level=level,
            out=out,
            cache_dir=cache_dir,
            limit=limit,
        )
    except Exception as exc:
        console.print(f"[red]Harbor compatibility check failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    summary = records_summary(records)
    console.print(
        "[bold]Harbor compatibility:[/bold] "
        f"{summary['total']} task refs, "
        f"{summary['pass']} pass, {summary['fail']} fail, "
        f"{summary['blocked']} blocked"
    )
    if out is not None:
        console.print(f"[green]Wrote JSONL report:[/green] {out}")


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
    _cleanup_daytona_sandboxes(dry_run=dry_run, max_age_minutes=max_age_minutes)


# ── Resource-verb subgroups (0.3 CLI) ────────────────────────────────────────

agent_app = typer.Typer(help="Agent management commands.")
app.add_typer(agent_app, name="agent")


@agent_app.command("list")
def agent_list() -> None:
    """List all registered agents."""
    from benchflow.agents.registry import AGENT_ALIASES, list_agents

    # Build reverse map: canonical name -> list of aliases
    reverse_aliases: dict[str, list[str]] = {}
    for alias, canonical in AGENT_ALIASES.items():
        if alias != canonical:
            reverse_aliases.setdefault(canonical, []).append(alias)

    table = Table(title="Registered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Aliases", style="dim")
    table.add_column("Description")
    table.add_column("Protocol", style="green")
    table.add_column("Requires", style="yellow")

    for a in list_agents():
        sub_env = a.subscription_auth.replaces_env if a.subscription_auth else None
        requires = [f"{e} (or login)" if e == sub_env else e for e in a.requires_env]
        aliases = ", ".join(sorted(reverse_aliases.get(a.name, [])))
        table.add_row(a.name, aliases, a.description, a.protocol, ", ".join(requires))

    console.print(table)


@agent_app.command("show")
def agent_show(
    name: Annotated[str, typer.Argument(help="Agent name")],
) -> None:
    """Show details for a registered agent."""
    from benchflow.agents.registry import AGENT_ALIASES, AGENTS

    resolved = AGENT_ALIASES.get(name, name)
    cfg = AGENTS.get(resolved)
    if not cfg:
        console.print(f"[red]Unknown agent: {name}[/red]")
        raise typer.Exit(1)

    # Collect aliases that point to this agent
    aliases = sorted(
        a for a, c in AGENT_ALIASES.items() if c == cfg.name and a != cfg.name
    )

    console.print(f"[bold]{cfg.name}[/bold]")
    if aliases:
        console.print(f"  Aliases:     {', '.join(aliases)}")
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
        typer.Option("--config", help="YAML config file"),
    ] = None,
    tasks_dir: Annotated[
        Path | None,
        typer.Option("--tasks-dir", help="Local tasks directory"),
    ] = None,
    source_repo: Annotated[
        str | None,
        typer.Option(
            "--source-repo",
            help="Remote repo as org/repo (e.g. benchflow-ai/skillsbench)",
        ),
    ] = None,
    source_path: Annotated[
        str | None,
        typer.Option("--source-path", help="Subpath within the repo (e.g. tasks)"),
    ] = None,
    source_ref: Annotated[
        str | None,
        typer.Option("--source-ref", help="Branch or tag to clone (e.g. main)"),
    ] = None,
    source_env: Annotated[
        str | None,
        typer.Option(
            "--source-env",
            help="Hosted environment source (e.g. primeintellect/general-agent)",
        ),
    ] = None,
    source_env_version: Annotated[
        str | None,
        typer.Option("--source-env-version", help="Hosted environment version"),
    ] = None,
    source_env_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--source-env-arg",
            help="Hosted environment arg as KEY=VALUE; repeatable",
        ),
    ] = None,
    source_env_num_examples: Annotated[
        int,
        typer.Option("--source-env-num-examples", help="Number of env examples"),
    ] = 1,
    source_env_rollouts_per_example: Annotated[
        int,
        typer.Option(
            "--source-env-rollouts-per-example",
            help="Rollouts per hosted env example",
        ),
    ] = 1,
    source_env_max_tokens: Annotated[
        int,
        typer.Option("--source-env-max-tokens", help="Max tokens for hosted env run"),
    ] = 1024,
    source_env_temperature: Annotated[
        float,
        typer.Option("--source-env-temperature", help="Temperature for hosted env run"),
    ] = 0.0,
    source_env_sampling_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--source-env-sampling-arg",
            help="Hosted env sampling arg as KEY=VALUE; repeatable (e.g. reasoning_effort=minimal)",
        ),
    ] = None,
    agent: Annotated[
        str,
        typer.Option("--agent", help="Agent name"),
    ] = DEFAULT_AGENT,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option("--sandbox", help="Sandbox: docker, daytona, or modal"),
    ] = "docker",
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Max concurrent tasks"),
    ] = 4,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", help="Output directory"),
    ] = "jobs",
    sandbox_user: Annotated[
        str | None,
        typer.Option("--sandbox-user", help="Sandbox user (null for root)"),
    ] = "agent",
    sandbox_setup_timeout: Annotated[
        int,
        typer.Option(
            "--sandbox-setup-timeout",
            help="Timeout (seconds) for sandbox user setup inside the environment.",
        ),
    ] = 120,
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills-dir", help="Skills directory to deploy"),
    ] = None,
    skill_mode: Annotated[
        str,
        typer.Option(
            "--skill-mode",
            help="Skill mode: default or self-gen",
        ),
    ] = "default",
    skill_creator_dir: Annotated[
        Path | None,
        typer.Option(
            "--skill-creator-dir",
            help="Path to skill-creator or a skills root containing it",
        ),
    ] = None,
    self_gen_no_internet: Annotated[
        bool,
        typer.Option(
            "--self-gen-no-internet",
            help="Disable web tools for the self-generated run",
        ),
    ] = False,
    agent_env: Annotated[
        list[str] | None,
        typer.Option("--agent-env", help="Agent env var (KEY=VALUE)"),
    ] = None,
) -> None:
    """Run an evaluation — single task or batch.

    Sandbox: docker, daytona, or modal.
    """
    from benchflow.evaluation import Evaluation, EvaluationConfig

    _apply_dotenv_to_process_env()
    parsed_env = _parse_agent_env(agent_env)
    sources = [bool(config_file), bool(tasks_dir), bool(source_repo), bool(source_env)]
    if sum(sources) > 1:
        console.print(
            "[red]Choose only one source: --config, --tasks-dir, --source-repo, or --source-env[/red]"
        )
        raise typer.Exit(1)
    agent = _normalize_eval_agent_or_exit(agent)
    sandbox_user = normalize_sandbox_user(sandbox_user)

    if config_file:
        j = Evaluation.from_yaml(config_file)
        j._config.agent = _normalize_eval_agent_or_exit(j._config.agent)
        j._config.model = effective_model(j._config.agent, j._config.model)
        j._config.agent_env = {**j._config.agent_env, **parsed_env}
        j._config.sandbox_user = normalize_sandbox_user(j._config.sandbox_user)
        result = asyncio.run(j.run())
        console.print(
            f"\n[bold]Score: {result.passed}/{result.total} "
            f"({result.score:.1%})[/bold], errors={result.errored}"
        )
        _exit_if_evaluation_had_errors(result)
    elif source_env:
        from benchflow.hosted_env import (
            HostedEnvError,
            HostedEnvRef,
            HostedEnvRunConfig,
            parse_sampling_args,
            parse_source_env_args,
            run_hosted_env,
        )

        if parsed_env:
            console.print(
                "[yellow]--agent-env is for BenchFlow ACP agents; source-env runs inherit the process environment.[/yellow]"
            )
        if environment != "docker":
            console.print(
                f"[yellow]--sandbox {environment!r} is not used by source-env runs; "
                "the hosted Verifiers environment owns its harness/sandbox.[/yellow]"
            )
        if agent != DEFAULT_AGENT:
            console.print(
                f"[dim]source-env records --agent {agent!r}, but executes the model endpoint through Verifiers.[/dim]"
            )

        try:
            ref = HostedEnvRef.parse(source_env, version=source_env_version)
            run_result = run_hosted_env(
                HostedEnvRunConfig(
                    source_env=ref,
                    model=model or "",
                    env_args=parse_source_env_args(source_env_arg),
                    agent=agent,
                    jobs_dir=Path(jobs_dir),
                    concurrency=concurrency,
                    num_examples=source_env_num_examples,
                    rollouts_per_example=source_env_rollouts_per_example,
                    max_tokens=source_env_max_tokens,
                    temperature=source_env_temperature,
                    sampling_args=parse_sampling_args(source_env_sampling_arg),
                )
            )
        except HostedEnvError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        console.print(f"\n[bold]Environment:[/bold] {run_result.source_env.env_uid}")
        console.print(f"[bold]Hub:[/bold] {run_result.source_env.hub_url}")
        console.print(
            f"[bold]Model:[/bold] {run_result.normalized_model}"
            + (
                f" [dim](from {run_result.model})[/dim]"
                if run_result.normalized_model != run_result.model
                else ""
            )
        )
        console.print(f"[bold]Run dir:[/bold] {run_result.run_dir}")
        console.print(f"[bold]Reward:[/bold] {run_result.reward}")
        if run_result.total_tool_calls is not None:
            console.print(f"[bold]Tool calls:[/bold] {run_result.total_tool_calls}")
        if run_result.error:
            console.print(f"[red]Error:[/red] {run_result.error}")
            raise typer.Exit(1)
    elif source_repo:
        from benchflow._utils.benchmark_repos import resolve_source

        resolved_tasks_dir = resolve_source(
            source_repo, path=source_path, ref=source_ref
        )
        eff_model = effective_model(agent, model)
        # Smart detection: if tasks_dir has task.toml, it's a single task
        if (resolved_tasks_dir / "task.toml").exists():
            from benchflow.sdk import SDK

            async def _run():
                return await SDK().run(
                    task_path=resolved_tasks_dir,
                    agent=agent,
                    model=eff_model,
                    job_name=None,
                    rollout_name=None,
                    jobs_dir=jobs_dir,
                    environment=environment,
                    agent_env=parsed_env,
                    skills_dir=str(skills_dir) if skills_dir else None,
                    sandbox_user=sandbox_user,
                    sandbox_setup_timeout=sandbox_setup_timeout,
                    skill_mode=skill_mode,
                    skill_creator_dir=(
                        str(skill_creator_dir) if skill_creator_dir else None
                    ),
                    self_gen_no_internet=self_gen_no_internet,
                )

            run_result = asyncio.run(_run())
            reward = (run_result.rewards or {}).get("reward")
            console.print(f"\n[bold]Task:[/bold] {resolved_tasks_dir.name}")
            console.print(f"[bold]Agent:[/bold] {agent} ({eff_model or 'no model'})")
            console.print(f"[bold]Reward:[/bold] {reward}")
            console.print(f"[bold]Tool calls:[/bold] {run_result.n_tool_calls}")
            _exit_if_run_result_failed(run_result)
        else:
            # Directory of tasks — batch run
            j = Evaluation(
                tasks_dir=str(resolved_tasks_dir),
                jobs_dir=jobs_dir,
                config=EvaluationConfig(
                    agent=agent,
                    model=eff_model,
                    environment=environment,
                    concurrency=concurrency,
                    agent_env=parsed_env,
                    sandbox_user=sandbox_user,
                    sandbox_setup_timeout=sandbox_setup_timeout,
                    skills_dir=str(skills_dir) if skills_dir else None,
                    skill_mode=skill_mode,
                    skill_creator_dir=(
                        str(skill_creator_dir) if skill_creator_dir else None
                    ),
                    self_gen_no_internet=self_gen_no_internet,
                ),
            )
            result = asyncio.run(j.run())
            console.print(
                f"\n[bold]Score: {result.passed}/{result.total} "
                f"({result.score:.1%})[/bold], errors={result.errored}"
            )
            _exit_if_evaluation_had_errors(result)
    elif tasks_dir:
        resolved_tasks_dir = tasks_dir
        eff_model = effective_model(agent, model)
        # Smart detection: if tasks_dir has task.toml, it's a single task
        if (resolved_tasks_dir / "task.toml").exists():
            from benchflow.sdk import SDK

            async def _run():
                return await SDK().run(
                    task_path=resolved_tasks_dir,
                    agent=agent,
                    model=eff_model,
                    job_name=None,
                    rollout_name=None,
                    jobs_dir=jobs_dir,
                    environment=environment,
                    agent_env=parsed_env,
                    skills_dir=str(skills_dir) if skills_dir else None,
                    sandbox_user=sandbox_user,
                    sandbox_setup_timeout=sandbox_setup_timeout,
                    skill_mode=skill_mode,
                    skill_creator_dir=(
                        str(skill_creator_dir) if skill_creator_dir else None
                    ),
                    self_gen_no_internet=self_gen_no_internet,
                )

            run_result = asyncio.run(_run())
            reward = (run_result.rewards or {}).get("reward")
            console.print(f"\n[bold]Task:[/bold] {resolved_tasks_dir.name}")
            console.print(f"[bold]Agent:[/bold] {agent} ({eff_model or 'no model'})")
            console.print(f"[bold]Reward:[/bold] {reward}")
            console.print(f"[bold]Tool calls:[/bold] {run_result.n_tool_calls}")
            _exit_if_run_result_failed(run_result)
        else:
            # Directory of tasks — batch run
            j = Evaluation(
                tasks_dir=str(resolved_tasks_dir),
                jobs_dir=jobs_dir,
                config=EvaluationConfig(
                    agent=agent,
                    model=eff_model,
                    environment=environment,
                    concurrency=concurrency,
                    agent_env=parsed_env,
                    sandbox_user=sandbox_user,
                    sandbox_setup_timeout=sandbox_setup_timeout,
                    skills_dir=str(skills_dir) if skills_dir else None,
                    skill_mode=skill_mode,
                    skill_creator_dir=(
                        str(skill_creator_dir) if skill_creator_dir else None
                    ),
                    self_gen_no_internet=self_gen_no_internet,
                ),
            )
            result = asyncio.run(j.run())
            console.print(
                f"\n[bold]Score: {result.passed}/{result.total} "
                f"({result.score:.1%})[/bold], errors={result.errored}"
            )
            _exit_if_evaluation_had_errors(result)
    else:
        console.print(
            "[red]Provide --config, --tasks-dir, --source-repo, or --source-env[/red]"
        )
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
    table.add_column("Evaluation", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Summary")

    root_summary = jobs_dir / "summary.json"
    if root_summary.exists():
        data = json.loads(root_summary.read_text())
        table.add_row(
            jobs_dir.name,
            str(data.get("total", "?")),
            f"{data.get('passed', '?')}/{data.get('total', '?')} ({data.get('score', '?')})",
        )
        console.print(table)
        return

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


env_app = typer.Typer(help="Environment management commands.")
app.add_typer(env_app, name="environment")


@env_app.command("create")
def environment_create(
    task_dir: Annotated[
        Path,
        typer.Argument(help="Task directory with task.toml + environment/Dockerfile"),
    ],
    sandbox: Annotated[
        str,
        typer.Option("--sandbox", help="Sandbox: docker, daytona, or modal"),
    ] = "daytona",
) -> None:
    """Create an environment from a task directory (does not start it)."""
    from benchflow.runtime import Environment

    env = Environment.from_task(task_dir, sandbox=sandbox)
    console.print(f"[green]Environment created:[/green] {env}")
    console.print(f"  Task:    {env.task_path}")
    console.print(f"  Sandbox: {env.sandbox}")
    console.print(
        "  Use [cyan]bench eval create[/cyan] for CLI runs, or pass to [cyan]bf.run()[/cyan]"
    )


@env_app.command("list")
def environment_list(
    hub: Annotated[
        str | None,
        typer.Option("--hub", help="Hosted environment hub to list"),
    ] = None,
    owner: Annotated[
        str | None,
        typer.Option("--owner", help="Hosted hub owner/namespace filter"),
    ] = None,
    search: Annotated[
        str | None,
        typer.Option("--search", help="Hosted hub search query"),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Maximum hosted hub results"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit raw JSON for hosted hub results"),
    ] = False,
) -> None:
    """List active Daytona sandboxes or hosted hub environments."""
    from datetime import datetime

    if hub:
        if hub != "primeintellect":
            console.print("[red]Only --hub primeintellect is supported today[/red]")
            raise typer.Exit(1)
        from benchflow.hosted_env import HostedEnvError, prime_env_list

        try:
            raw = prime_env_list(owner=owner, search=search, limit=limit)
        except HostedEnvError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None
        if output_json:
            console.print(raw)
            return
        data = json.loads(raw)
        rows = (
            data
            if isinstance(data, list)
            else data.get("environments", data.get("items", []))
        )
        table = Table(title="PrimeIntellect Environments")
        table.add_column("Environment", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Visibility")
        table.add_column("Updated", style="dim")
        for item in rows:
            name = (
                item.get("environment")
                or item.get("fullName")
                or item.get("name")
                or item.get("id")
                or ""
            )
            version = str(item.get("version") or item.get("latestVersion") or "")
            visibility = str(item.get("visibility") or item.get("private") or "")
            updated = str(item.get("updated_at") or item.get("updatedAt") or "")
            table.add_row(name, version, visibility, updated)
        console.print(table)
        return

    d = _daytona_client_or_exit()
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


@env_app.command("show")
def environment_show(
    source_env: Annotated[
        str,
        typer.Argument(help="Hosted environment (e.g. primeintellect/general-agent)"),
    ],
    version: Annotated[
        str | None,
        typer.Option("--version", help="Hosted environment version"),
    ] = None,
) -> None:
    """Show hosted environment metadata."""
    from benchflow.hosted_env import HostedEnvError, HostedEnvRef, prime_env_info

    try:
        ref = HostedEnvRef.parse(source_env, version=version)
        console.print(prime_env_info(ref))
    except HostedEnvError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@env_app.command("inspect")
def environment_inspect(
    source_env: Annotated[
        str,
        typer.Argument(help="Hosted environment (e.g. primeintellect/general-agent)"),
    ],
    version: Annotated[
        str | None,
        typer.Option("--version", help="Hosted environment version"),
    ] = None,
    path: Annotated[
        str,
        typer.Option("--path", help="File inside the hosted environment package"),
    ] = "README.md",
) -> None:
    """Inspect a file from a hosted environment package."""
    from benchflow.hosted_env import HostedEnvError, HostedEnvRef, prime_env_inspect

    try:
        ref = HostedEnvRef.parse(source_env, version=version)
        console.print(prime_env_inspect(ref, path=path))
    except HostedEnvError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@env_app.command("cleanup")
def environment_cleanup(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List sandboxes without deleting"),
    ] = False,
    max_age_minutes: Annotated[
        int,
        typer.Option("--max-age", help="Delete sandboxes older than N minutes"),
    ] = 1440,
) -> None:
    """Clean up orphaned Daytona sandboxes."""
    _cleanup_daytona_sandboxes(dry_run=dry_run, max_age_minutes=max_age_minutes)


if __name__ == "__main__":
    app()
