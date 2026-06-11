"""benchflow CLI — agent benchmarking framework."""

import asyncio
import json
import logging
import os
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import typer
from rich.console import Console
from rich.table import Table

from benchflow import __version__
from benchflow._dotenv import load_dotenv_env
from benchflow._utils.config import (
    DEFAULT_AGENT_IDLE_TIMEOUT_SEC,
    normalize_agent_idle_timeout,
    normalize_reasoning_effort,
    normalize_sandbox_user,
)
from benchflow.agent_router import register_agent_router
from benchflow.agents.registry import parse_agent_spec
from benchflow.cli._options import (
    AgentOption,
    ConcurrencyOption,
    JobsDirOption,
    ModelOption,
    MonitorJobsDirOption,
    SandboxOption,
    SkillModeOption,
)
from benchflow.cli.continue_cmd import register_continue
from benchflow.cli.trace_import import register_tasks_generate
from benchflow.evaluation import DEFAULT_AGENT, effective_model
from benchflow.skill_policy import SKILL_MODE_NO_SKILL
from benchflow.usage_tracking import UsageTrackingConfig

if TYPE_CHECKING:
    from benchflow.evaluation import EvaluationResult

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

# Standalone `benchflow continue <orig-run-folder>` — resume a timed-out run.
register_continue(app)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"benchflow {__version__}")
        raise typer.Exit()


@app.callback()
def _cli_main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
) -> None:
    """ACP-native agent benchmarking framework."""


_PROVIDER_AUTH_MESSAGE = (
    "Provider-prefixed models may use different credentials; Azure Foundry "
    "models use AZURE_API_KEY + AZURE_API_ENDPOINT."
)
_REQUIRES_AUTH_NOTE = (
    "Requires shows native/default agent auth. " + _PROVIDER_AUTH_MESSAGE
)


def _format_requires(agent) -> str:
    sub_env = agent.subscription_auth.replaces_env if agent.subscription_auth else None
    requires = [
        f"{env_var} (or login)" if env_var == sub_env else env_var
        for env_var in agent.requires_env
    ]
    return ", ".join(requires)


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


def _exit_if_evaluation_had_errors(result: object) -> None:
    errored = int(getattr(result, "errored", 0) or 0)
    verifier_errored = int(getattr(result, "verifier_errored", 0) or 0)
    if errored or verifier_errored:
        raise typer.Exit(1)


def _report_eval_result(result: "EvaluationResult") -> None:
    """Print the standard Score/errors summary line for an evaluation result."""
    console.print(
        f"\n[bold]Score: {result.passed}/{result.total} "
        f"({result.score:.1%})[/bold], errors={result.errored}"
    )


def _normalize_eval_agent_or_exit(agent_spec: str) -> str:
    protocol, canonical_agent = parse_agent_spec(agent_spec)
    if protocol not in ("acp", "acpx"):
        console.print(f"[red]Unsupported eval agent protocol: {protocol}[/red]")
        raise typer.Exit(1)
    if protocol == "acpx":
        return f"acpx/{canonical_agent}"
    return canonical_agent


def _daytona_client_or_exit():
    # Canonical sync-client bootstrap (anyio compat + client build) lives in
    # benchflow.sandbox.daytona; reuse it instead of re-deriving it here.
    from benchflow.sandbox.daytona import build_sync_client

    try:
        return build_sync_client()
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


def _cleanup_daytona_sandboxes(dry_run: bool, max_age_minutes: int) -> None:
    """Clean up orphaned Daytona sandboxes (display wrapper over the library reaper)."""
    from benchflow.sandbox.daytona import reap_stale_sandboxes

    d = _daytona_client_or_exit()

    def _show(sb, age_minutes, will_delete):
        verdict = "[red](delete)[/red]" if will_delete else "[green](skip)[/green]"
        if dry_run or not will_delete:
            console.print(
                f"  [dim]{sb.id}[/dim] state={sb.state} age={age_minutes:.0f}m {verdict}"
            )

    counts = reap_stale_sandboxes(
        d,
        max_age_minutes=max_age_minutes,
        failed_max_age_minutes=max_age_minutes,
        dry_run=dry_run,
        on_decision=_show,
    )
    if dry_run:
        console.print(
            f"\n[bold]{counts['found']} sandboxes found, {counts['deleted']} older than {max_age_minutes}m[/bold] (use without --dry-run to delete)"
        )
    else:
        console.print(
            f"\n[bold green]{counts['deleted']} sandboxes deleted[/bold green] ({counts['skipped']} skipped, younger than {max_age_minutes}m)"
        )


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
    environment: SandboxOption = "docker",
    concurrency: ConcurrencyOption = 4,
    max_retries: Annotated[
        int,
        typer.Option("--retries", help="Max retries per task"),
    ] = 0,
    jobs_dir: JobsDirOption = "jobs",
    skills_dir: Annotated[
        Path | None,
        typer.Option("--skills-dir", help="Skills directory to deploy into sandbox"),
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
    environment: SandboxOption = "docker",
    concurrency: ConcurrencyOption = 1,
    jobs_dir: JobsDirOption = "jobs",
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
    no_oracle: Annotated[
        bool,
        typer.Option(
            "--no-oracle",
            "--no-solution",
            help="Skip oracle template",
        ),
    ] = False,
    task_format: Annotated[
        str, typer.Option("--format", help="Task format: legacy or task-md")
    ] = "task-md",
) -> None:
    """Scaffold a new benchmark task."""
    from benchflow._utils.task_authoring import scaffold_task

    try:
        result = scaffold_task(
            name,
            parent_dir=parent_dir,
            no_pytest=no_pytest,
            no_oracle=no_oracle,
            task_format=cast(Literal["legacy", "task-md"], task_format),
        )
        console.print(f"[green]Created:[/green] {result.task_dir}/")
        # List every file actually written, derived from the scaffold itself so
        # the summary can never under-report (e.g. omit verifier/test_outputs.py
        # or verifier/rubrics/verifier.toml, both of which `bench tasks check`
        # validates).
        for rel in result.files:
            console.print(f"  {rel}")
    except (FileExistsError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


@tasks_app.command("check")
def tasks_check(
    task_dir: Annotated[Path, typer.Argument(help="Path to task directory")],
    validation_level: Annotated[
        Literal[
            "schema",
            "structural",
            "runtime-capability",
            "publication-grade",
            "acceptance",
            "acceptance-live",
        ],
        typer.Option(
            "--level",
            help=(
                "Validation level: schema, structural, runtime-capability, "
                "publication-grade, acceptance, or acceptance-live"
            ),
        ),
    ] = "structural",
    sandbox: Annotated[
        str | None,
        typer.Option(
            "--sandbox",
            help="Also validate parsed runtime semantics for docker, daytona, or modal",
        ),
    ] = None,
    report_output: Annotated[
        Path | None,
        typer.Option(
            "--report-output",
            help=(
                "Write the acceptance-live report to this host path instead "
                "of the task-declared report path"
            ),
        ),
    ] = None,
    no_report_write: Annotated[
        bool,
        typer.Option(
            "--no-report-write",
            help=(
                "Validate acceptance-live without writing the declared report "
                "or its .sha256 sidecar (report-only dogfood; leaves the task "
                "package unmodified). Takes precedence over --report-output."
            ),
        ),
    ] = False,
) -> None:
    """Validate a task directory structure."""
    from rich.markup import escape

    from benchflow._utils.task_authoring import check_task

    issues = check_task(
        task_dir,
        sandbox_type=sandbox,
        validation_level=validation_level,
        acceptance_live_report_output=report_output,
        acceptance_live_write_report=not no_report_write,
    )
    if not issues:
        console.print(f"[green]✓[/green] {task_dir.name} — valid ({validation_level})")
    else:
        console.print(f"[red]✗[/red] {task_dir.name} — {len(issues)} issue(s):")
        for issue in issues:
            # Escape Rich markup so literal section names like "[agent]"
            # render verbatim instead of being parsed as styling (#379).
            console.print(f"  [yellow]→[/yellow] {escape(issue)}")
        raise typer.Exit(1)


@tasks_app.command("migrate")
def tasks_migrate(
    task_dir: Annotated[Path, typer.Argument(help="Legacy task directory")],
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing task.md"),
    ] = False,
    remove_legacy: Annotated[
        bool,
        typer.Option(
            "--remove-legacy",
            help=(
                "Delete split files and promote tests/solution aliases after "
                "task.md is verified"
            ),
        ),
    ] = False,
) -> None:
    """Convert task.toml + instruction.md into the unified task.md format."""
    from benchflow._utils.task_authoring import migrate_task_to_task_md

    try:
        result = migrate_task_to_task_md(
            task_dir,
            overwrite=overwrite,
            remove_legacy=remove_legacy,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    console.print(f"[green]Created:[/green] {result.task_md}")
    if result.removed_legacy:
        console.print("  removed task.toml and instruction.md")
        for migrated_dir in result.migrated_legacy_dirs:
            console.print(f"  promoted {migrated_dir}")
    else:
        console.print("  kept task.toml and instruction.md")


@tasks_app.command("normalize")
def tasks_normalize(
    task_dir: Annotated[Path, typer.Argument(help="Task directory with task.md")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write normalized task.md to this path instead of stdout",
        ),
    ] = None,
    write: Annotated[
        bool,
        typer.Option(
            "--write",
            help="Replace task.md in place with the normalized canonical form",
        ),
    ] = False,
) -> None:
    """Expand minimal task.md authoring profiles into canonical task.md."""
    from benchflow._utils.task_authoring import normalize_task_md

    try:
        result = normalize_task_md(task_dir, output_path=output, write=write)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if result.output_path is None:
        typer.echo(result.normalized_text, nl=False)
    else:
        console.print(f"[green]Normalized:[/green] {result.output_path}")


@tasks_app.command("export")
def tasks_export(
    task_dir: Annotated[Path, typer.Argument(help="Task directory to export")],
    output_dir: Annotated[
        Path | None,
        typer.Argument(
            help="Destination split-layout directory (omit with --report-only)",
        ),
    ] = None,
    target: Annotated[
        str,
        typer.Option("--target", help="Compatibility target: harbor or pier"),
    ] = "harbor",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing export directory"),
    ] = False,
    report_only: Annotated[
        bool,
        typer.Option(
            "--report-only",
            help="Print the compatibility loss report without writing files",
        ),
    ] = False,
) -> None:
    """Export a task to a Harbor/Pier split layout with a loss report."""
    from benchflow.task import (
        build_compatibility_export_report,
        export_task_to_split_layout,
    )

    if target not in {"harbor", "pier"}:
        console.print("[red]target must be 'harbor' or 'pier'[/red]")
        raise typer.Exit(1)

    try:
        if report_only:
            report = build_compatibility_export_report(
                task_dir,
                target=cast(Literal["harbor", "pier"], target),
            )
            typer.echo(report.to_json(), nl=False)
            return
        if output_dir is None:
            console.print(
                "[red]Missing output_dir; pass one or use --report-only[/red]"
            )
            raise typer.Exit(1)
        report = export_task_to_split_layout(
            task_dir,
            output_dir,
            target=cast(Literal["harbor", "pier"], target),
            overwrite=overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    console.print(f"[green]Exported:[/green] {output_dir}")
    console.print(f"  target: {report.target}")
    console.print(f"  status: {report.status}")
    console.print(f"  losses: {len(report.losses)}")
    console.print("  report: compatibility/export-report.json")


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


# Resource-verb subgroups (0.3 CLI)

agent_app = typer.Typer(help="Agent management commands.")
app.add_typer(agent_app, name="agent")
register_agent_router(agent_app)


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
        aliases = ", ".join(sorted(reverse_aliases.get(a.name, [])))
        table.add_row(a.name, aliases, a.description, a.protocol, _format_requires(a))

    console.print(table)
    console.print(f"[dim]{_REQUIRES_AUTH_NOTE}[/dim]")


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
    console.print(f"  Requires:    {_format_requires(cfg) or '(none)'}")
    console.print(f"  Provider auth: {_PROVIDER_AUTH_MESSAGE}")
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
        str | None,
        typer.Option("--agent", help="Agent name"),
    ] = None,
    model: ModelOption = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            help="Agent reasoning/thinking effort when the agent exposes one (e.g. max)",
        ),
    ] = None,
    environment: Annotated[
        str | None,
        typer.Option("--sandbox", help="Sandbox: docker, daytona, or modal"),
    ] = None,
    usage_tracking: Annotated[
        str | None,
        typer.Option(
            "--usage-tracking",
            help="Token usage tracking policy: auto, required, or off",
        ),
    ] = None,
    environment_manifest: Annotated[
        Path | None,
        typer.Option(
            "--environment-manifest",
            help=(
                "Path to an Environment-plane manifest (environment.toml). "
                "Applied to every rollout in the batch so the manifest-declared "
                "stateful environment is provisioned, gated on readiness, and "
                "torn down."
            ),
        ),
    ] = None,
    prompt: Annotated[
        list[str] | None,
        typer.Option("--prompt", help="Prompt(s) to send (default: instruction.md)"),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", help="Max concurrent tasks"),
    ] = None,
    build_concurrency: Annotated[
        int | None,
        typer.Option(
            "--build-concurrency",
            help=(
                "Max concurrent docker image builds. Defaults to --concurrency. "
                "Set lower (e.g. 8) when --concurrency is high to avoid "
                "overwhelming the docker daemon with parallel builds."
            ),
        ),
    ] = None,
    worker_concurrency: Annotated[
        int | None,
        typer.Option(
            "--worker-concurrency",
            help=(
                "Run batch eval through isolated worker subprocesses, each with "
                "at most this many concurrent tasks; --concurrency remains the "
                "aggregate target."
            ),
        ),
    ] = None,
    worker_retries: Annotated[
        int,
        typer.Option(
            "--worker-retries",
            help="Retry a crashed worker shard this many times, resuming its jobs_dir.",
        ),
    ] = 1,
    worker_start_stagger_sec: Annotated[
        float,
        typer.Option(
            "--worker-start-stagger-sec",
            help="Seconds to stagger worker starts to avoid Daytona connection storms.",
        ),
    ] = 1.0,
    agent_idle_timeout: Annotated[
        str | None,
        typer.Option(
            "--agent-idle-timeout",
            help=(
                "Abort ACP prompts after this many idle seconds (default: 600). "
                "Pass 0 or 'none' to disable the idle watchdog and fall back to "
                "the task's wall-clock timeout."
            ),
        ),
    ] = None,
    jobs_dir: Annotated[
        str | None,
        typer.Option("--jobs-dir", help="Output directory"),
    ] = None,
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
    skill_mode: SkillModeOption = SKILL_MODE_NO_SKILL,
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
    include: Annotated[
        list[str] | None,
        typer.Option(
            "--include",
            help="Only run these task names; repeatable (e.g. --include jax-computing-basics --include data-to-d3)",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude",
            help="Skip these task names; repeatable (e.g. --exclude quantum-numerical-simulation)",
        ),
    ] = None,
) -> None:
    """Run an evaluation — single task or batch.

    Sandbox: docker, daytona, or modal.
    """
    from benchflow.evaluation import (
        EmptyTaskSelectionError,
        Evaluation,
        EvaluationConfig,
    )

    _apply_dotenv_to_process_env()
    parsed_env = _parse_agent_env(agent_env)
    include_tasks = set(include) if include else set()
    exclude_tasks = set(exclude) if exclude else set()
    sources = [bool(config_file), bool(tasks_dir), bool(source_repo), bool(source_env)]
    if sum(sources) > 1:
        console.print(
            "[red]Choose only one source: --config, --tasks-dir, --source-repo, or --source-env[/red]"
        )
        raise typer.Exit(1)
    if worker_concurrency is not None and not (tasks_dir or source_repo):
        console.print(
            "[red]--worker-concurrency is supported for --tasks-dir and --source-repo batch runs[/red]"
        )
        raise typer.Exit(1)
    if worker_retries < 0:
        console.print("[red]--worker-retries must be >= 0[/red]")
        raise typer.Exit(1)
    if worker_start_stagger_sec < 0:
        console.print("[red]--worker-start-stagger-sec must be >= 0[/red]")
        raise typer.Exit(1)
    eval_agent = (
        _normalize_eval_agent_or_exit(agent) if agent is not None else DEFAULT_AGENT
    )
    eval_environment = environment or "docker"
    eval_prompts = cast("list[str | None] | None", prompt)
    sandbox_user = normalize_sandbox_user(sandbox_user)
    eval_concurrency = concurrency if concurrency is not None else 4
    usage_tracking_overridden = usage_tracking is not None
    try:
        eval_usage_tracking = UsageTrackingConfig(mode=usage_tracking)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid usage tracking config: {exc}[/red]")
        raise typer.Exit(1) from None
    try:
        eval_agent_idle_timeout = normalize_agent_idle_timeout(
            agent_idle_timeout
            if agent_idle_timeout is not None
            else DEFAULT_AGENT_IDLE_TIMEOUT_SEC
        )
    except ValueError as exc:
        console.print(
            f"[red]Invalid --agent-idle-timeout {agent_idle_timeout!r}: {exc}[/red]"
        )
        raise typer.Exit(1) from None
    try:
        eval_reasoning_effort = normalize_reasoning_effort(reasoning_effort)
    except ValueError as exc:
        console.print(
            f"[red]Invalid --reasoning-effort {reasoning_effort!r}: {exc}[/red]"
        )
        raise typer.Exit(1) from None
    output_jobs_dir = jobs_dir or "jobs"

    # Resolve the optional Environment-plane manifest once and reuse across
    # every source branch (config / source_repo / tasks_dir / source_env).
    eval_env_manifest = None
    if environment_manifest is not None:
        from benchflow.environment.manifest import load_manifest

        try:
            eval_env_manifest = load_manifest(environment_manifest)
        except (OSError, ValueError) as exc:
            console.print(
                f"[red]Could not load --environment-manifest {environment_manifest}: {exc}[/red]"
            )
            raise typer.Exit(1) from None

    def _make_eval_config(
        source_provenance: dict[str, Any] | None = None,
    ) -> EvaluationConfig:
        """Build the EvaluationConfig shared by the source-repo and tasks-dir paths."""
        return EvaluationConfig(
            agent=eval_agent,
            model=effective_model(eval_agent, model),
            reasoning_effort=eval_reasoning_effort,
            environment=eval_environment,
            concurrency=eval_concurrency,
            build_concurrency=build_concurrency,
            prompts=eval_prompts,
            agent_idle_timeout=eval_agent_idle_timeout,
            agent_env=parsed_env,
            sandbox_user=sandbox_user,
            sandbox_setup_timeout=sandbox_setup_timeout,
            skills_dir=str(skills_dir) if skills_dir else None,
            skill_mode=skill_mode,
            skill_creator_dir=str(skill_creator_dir) if skill_creator_dir else None,
            self_gen_no_internet=self_gen_no_internet,
            source_provenance=source_provenance,
            include_tasks=include_tasks,
            exclude_tasks=exclude_tasks,
            usage_tracking=eval_usage_tracking,
            environment_manifest=eval_env_manifest,
        )

    def _run_batch_eval(
        resolved_tasks_dir: Path,
        eval_config: EvaluationConfig,
    ):
        from benchflow.eval_sharding import ShardWorkerError

        try:
            if worker_concurrency is None:
                result = asyncio.run(
                    Evaluation(
                        tasks_dir=str(resolved_tasks_dir),
                        jobs_dir=output_jobs_dir,
                        config=eval_config,
                    ).run()
                )
            else:
                from benchflow.eval_sharding import run_sharded_evaluation

                result = asyncio.run(
                    run_sharded_evaluation(
                        tasks_dir=resolved_tasks_dir,
                        jobs_dir=Path(output_jobs_dir),
                        config=eval_config,
                        worker_concurrency=worker_concurrency,
                        worker_retries=worker_retries,
                        worker_start_stagger_sec=worker_start_stagger_sec,
                        environment_manifest_path=environment_manifest,
                    )
                )
        except EmptyTaskSelectionError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None
        except (ValueError, RuntimeError, ShardWorkerError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        _report_eval_result(result)
        _exit_if_evaluation_had_errors(result)
        return result

    if config_file:
        j = Evaluation.from_yaml(config_file)
        if agent is not None:
            j._config.agent = eval_agent
        else:
            j._config.agent = _normalize_eval_agent_or_exit(j._config.agent)
        if model is not None:
            j._config.model = effective_model(j._config.agent, model)
        else:
            j._config.model = effective_model(j._config.agent, j._config.model)
        if reasoning_effort is not None:
            j._config.reasoning_effort = eval_reasoning_effort
        if environment is not None:
            j._config.environment = eval_environment
        j._config.agent_env = {**j._config.agent_env, **parsed_env}
        j._config.sandbox_user = normalize_sandbox_user(j._config.sandbox_user)
        if jobs_dir is not None:
            j._jobs_dir = Path(jobs_dir)
        if concurrency is not None:
            j._config.concurrency = concurrency
        if build_concurrency is not None:
            j._config.build_concurrency = build_concurrency
        if prompt is not None:
            j._config.prompts = eval_prompts
        if agent_idle_timeout is not None:
            j._config.agent_idle_timeout = eval_agent_idle_timeout
        if usage_tracking_overridden:
            j._config.usage_tracking = j._config.usage_tracking.overlay(
                eval_usage_tracking
            )
        if include_tasks:
            j._config.include_tasks = include_tasks
        if exclude_tasks:
            j._config.exclude_tasks = exclude_tasks
        # CLI --environment-manifest wins over whatever the YAML carried
        # so an operator can override a baseline manifest without editing
        # the config file.
        if eval_env_manifest is not None:
            j._config.environment_manifest = eval_env_manifest
        try:
            result = asyncio.run(j.run())
        except (EmptyTaskSelectionError, ValueError) as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None
        _report_eval_result(result)
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
        if eval_environment != "docker":
            console.print(
                f"[yellow]--sandbox {eval_environment!r} is not used by source-env runs; "
                "the hosted Verifiers environment owns its harness/sandbox.[/yellow]"
            )
        if eval_agent != DEFAULT_AGENT:
            console.print(
                f"[dim]source-env records --agent {eval_agent!r}, but executes the model endpoint through Verifiers.[/dim]"
            )
        if eval_env_manifest is not None:
            console.print(
                "[yellow]--environment-manifest is for benchflow Environment-plane rollouts; "
                "the hosted Verifiers environment owns its own provisioning. Ignoring.[/yellow]"
            )
        if usage_tracking_overridden:
            console.print(
                "[yellow]--usage-tracking is for BenchFlow ACP rollouts; "
                "source-env runs own their provider calls. Ignoring.[/yellow]"
            )
        if prompt is not None:
            console.print(
                "[yellow]--prompt is for BenchFlow ACP rollouts; "
                "source-env runs own their prompts. Ignoring.[/yellow]"
            )

        try:
            ref = HostedEnvRef.parse(source_env, version=source_env_version)
            run_result = run_hosted_env(
                HostedEnvRunConfig(
                    source_env=ref,
                    model=model or "",
                    env_args=parse_source_env_args(source_env_arg),
                    agent=eval_agent,
                    jobs_dir=Path(output_jobs_dir),
                    concurrency=eval_concurrency,
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
        from benchflow._utils.benchmark_repos import resolve_source_with_metadata

        resolved = resolve_source_with_metadata(
            source_repo, path=source_path, ref=source_ref
        )
        resolved_tasks_dir = resolved.path
        _run_batch_eval(
            resolved_tasks_dir,
            _make_eval_config(source_provenance=resolved.provenance),
        )
    elif tasks_dir:
        resolved_tasks_dir = tasks_dir
        # Single-task and batch share one orchestration path. Evaluation
        # handles both layouts (Evaluation._get_task_dirs detects when
        # tasks_dir itself contains task.toml) and applies include/exclude
        # filters uniformly (#400, #401, #407).
        _run_batch_eval(
            resolved_tasks_dir,
            _make_eval_config(),
        )
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
    table.add_column("Memory")

    def memory_label(data: dict) -> str:
        score = data.get("memory_score")
        return f"{score:.1%}" if isinstance(score, int | float) else "—"

    root_summary = jobs_dir / "summary.json"
    if root_summary.exists():
        data = json.loads(root_summary.read_text())
        table.add_row(
            jobs_dir.name,
            str(data.get("total", "?")),
            f"{data.get('passed', '?')}/{data.get('total', '?')} ({data.get('score', '?')})",
            memory_label(data),
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
                memory_label(data),
            )
        else:
            sub_count = sum(1 for s in d.iterdir() if s.is_dir())
            table.add_row(d.name, str(sub_count), "[dim]no summary[/dim]", "—")

    console.print(table)


env_app = typer.Typer(help="Environment management commands.")
app.add_typer(env_app, name="environment")


@env_app.command("create")
def environment_create(
    task_dir: Annotated[
        Path,
        typer.Argument(help="Task directory with task.md or task.toml + Dockerfile"),
    ],
    sandbox: SandboxOption = "daytona",
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

    now = datetime.now(UTC)
    total = 0
    # daytona SDK >=0.18: ``list()`` yields an auto-paginating Iterator[Sandbox]
    # (was a paged ``list(page=, limit=)`` -> page object with ``.items``).
    for sb in d.list():
        total += 1
        age = ""
        if sb.created_at:
            created = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
            mins = (now - created).total_seconds() / 60
            age = f"{mins:.0f}m"
        target = getattr(sb, "target", "") or ""
        table.add_row(sb.id[:12] + "…", str(sb.state), age, str(target)[:40])

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


monitor_app = typer.Typer(
    help=(
        "Monitor mode — score a rollout in production (#386). "
        "API surface scaffold; runtime not yet implemented."
    ),
)
app.add_typer(monitor_app, name="monitor")


def _monitor_not_implemented() -> None:
    """Emit the canonical not-implemented message and exit non-zero.

    Centralised so every ``bench monitor`` subcommand fails closed with the
    same wording and exit code, matching the issue's "fail closed with a
    clear not-implemented status" requirement.
    """
    from benchflow.monitor import not_implemented_message

    console.print("[yellow]Monitor mode not implemented yet.[/yellow]")
    console.print(not_implemented_message())
    # Exit code 2 (not 1) — distinguishes "feature absent" from "feature
    # ran and failed", so CI dashboards do not conflate the two.
    raise typer.Exit(2)


@monitor_app.command("run")
def monitor_run(
    source: Annotated[
        str,
        typer.Argument(help="Source trajectory (persisted rollout dir, file, or URI)."),
    ],
    rubric: Annotated[
        Path | None,
        typer.Option("--rubric", help="Rubric/verifier definition to score against."),
    ] = None,
    jobs_dir: MonitorJobsDirOption = "jobs/monitor",
    run_name: Annotated[
        str | None,
        typer.Option("--run-name", help="Human-readable id for this monitor run."),
    ] = None,
) -> None:
    """Score one trajectory under monitor semantics. **Not yet implemented (#386).**"""
    del source, rubric, jobs_dir, run_name  # accepted for API stability
    _monitor_not_implemented()


@monitor_app.command("replay")
def monitor_replay(
    trajectory_path: Annotated[
        Path,
        typer.Argument(help="Path to a persisted rollout/trajectory to re-score."),
    ],
    jobs_dir: MonitorJobsDirOption = "jobs/monitor",
) -> None:
    """Re-score a persisted rollout under monitor semantics. **Not yet implemented (#386).**"""
    del trajectory_path, jobs_dir  # accepted for API stability
    _monitor_not_implemented()


@monitor_app.command("watch")
def monitor_watch(
    source: Annotated[
        str,
        typer.Argument(help="Live event source (webhook, polling endpoint, queue)."),
    ],
    jobs_dir: MonitorJobsDirOption = "jobs/monitor",
) -> None:
    """Stream-score live production events. **Not yet implemented (#386).**"""
    del source, jobs_dir  # accepted for API stability
    _monitor_not_implemented()


if __name__ == "__main__":
    app()
