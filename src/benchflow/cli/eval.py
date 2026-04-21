"""`bf eval` — the benchflow eval-runner command group.

The future-facing entry point for running evaluations. Anthropic-style shape:
resource creation, one command, return the result or a job-id.

    bf eval create <task-ref> [flags]
        One-shot eval — creates an Agent + Environment + Trajectory under
        the hood and runs the task. `task-ref` can be:
          - a path to a task directory (single task)
          - a path to a directory of task directories (batch)
          - a `harbor://<name>[@<version>]` ref (full Harbor dataset)
          - a `harbor://<name>/<task>` ref (single task from a Harbor dataset)
          - a `benchflow://<name>[@<version>]` ref (benchflow-owned dataset)

    bf eval list          Show recent eval runs (reads the jobs/ dir)
    bf eval retrieve ID   Look up a specific trajectory by trial name

Replaces `bf run` + `bf job` as the idiomatic way to run evals. `bf run`
stays around as a deprecated alias for one release.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from benchflow.job import DEFAULT_AGENT, effective_model as _effective_model

console = Console()

app = typer.Typer(
    name="eval",
    help="Run evaluations. `bf eval create <task>` is the main entry point.",
    no_args_is_help=True,
)


def _resolve_task_ref(task_ref: str) -> tuple[Path, bool]:
    """Resolve a positional task reference to a local directory.

    Returns `(path, is_batch)`. `is_batch` is True when the reference
    points at a directory containing multiple task dirs (each with its
    own `task.toml`), False when the reference is a single task dir.
    """

    # Registry prefix: fetch the full dataset and treat as batch.
    if task_ref.startswith("harbor://") or task_ref.startswith("benchflow://"):
        from benchflow.task_download import ensure_tasks

        # Allow `harbor://<name>/<task>` shorthand for a single task within
        # a dataset. Split off the trailing segment if it matches a task.
        prefix, _, tail = task_ref.partition("://")
        head = tail
        sub_task: str | None = None
        if "/" in tail and "@" not in tail.split("/", 1)[1]:
            dataset, sub_task = tail.split("/", 1)
            head = dataset
        dataset_ref = f"{prefix}://{head}"
        dataset_dir = ensure_tasks(dataset_ref)
        if sub_task is not None:
            candidate = dataset_dir / sub_task
            if not candidate.exists() or not (candidate / "task.toml").exists():
                console.print(
                    f"[red]Harbor dataset {head!r} has no task named {sub_task!r}.[/red]"
                )
                raise typer.Exit(1)
            return candidate, False
        return dataset_dir, True

    # Filesystem path: single task if task.toml is present, batch otherwise.
    path = Path(task_ref).expanduser().resolve()
    if not path.exists():
        console.print(f"[red]Task reference not found: {task_ref}[/red]")
        raise typer.Exit(1)
    if (path / "task.toml").exists():
        return path, False
    if any(
        child.is_dir() and (child / "task.toml").exists() for child in path.iterdir()
    ):
        return path, True
    console.print(
        f"[red]{path} is neither a single task (no task.toml) "
        f"nor a directory of tasks.[/red]"
    )
    raise typer.Exit(1)


@app.command("create")
def eval_create(
    task: Annotated[
        str,
        typer.Argument(
            help=(
                "Task reference. Path to a task dir, a dir of tasks, or a "
                "registry ref (harbor://name, benchflow://name)."
            )
        ),
    ],
    agent: Annotated[
        str, typer.Option("--agent", "-a", help="Agent name from the registry")
    ] = DEFAULT_AGENT,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model id; agent default if unset"),
    ] = None,
    environment: Annotated[
        str,
        typer.Option(
            "--environment",
            "-e",
            help="docker | daytona | ... (uses the agent's default if unset)",
        ),
    ] = "docker",
    prompt: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Prompt text; repeat for multi-turn. Default: instruction.md",
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            "-c",
            help="Max parallel trials when task is a batch",
        ),
    ] = 4,
    max_retries: Annotated[
        int,
        typer.Option(
            "--max-retries",
            help="Per-trial retry count on transient errors",
        ),
    ] = 1,
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", "-o", help="Where to write result files"),
    ] = "jobs",
    skills_dir: Annotated[
        Path | None,
        typer.Option(
            "--skills-dir",
            "-s",
            help="Skills directory to mount into the sandbox",
        ),
    ] = None,
    sandbox_user: Annotated[
        str | None,
        typer.Option(
            "--sandbox-user",
            help="Non-root sandbox user (default 'agent'; 'none' = root)",
        ),
    ] = "agent",
    agent_env: Annotated[
        list[str] | None,
        typer.Option("--agent-env", help="Extra agent env var (KEY=VALUE)"),
    ] = None,
) -> None:
    """Create and run an eval — one-shot.

    Under the hood:
      1. Resolves `task` to a local directory (fetching from Harbor if needed).
      2. If it's a single task: runs `SDK.run()` once and prints the reward.
      3. If it's a batch: runs a `Job` at `--concurrency` and prints pass rate.

    This is the idiomatic way to run evals going forward. `bf run` and
    `bf job` remain as one-release deprecated aliases that forward here.
    """

    resolved, is_batch = _resolve_task_ref(task)

    parsed_env: dict[str, str] = {}
    for entry in agent_env or []:
        if "=" not in entry:
            console.print(f"[red]Invalid --agent-env value: {entry!r}[/red]")
            raise typer.Exit(2)
        k, v = entry.split("=", 1)
        parsed_env[k] = v

    if is_batch:
        _run_batch(
            tasks_dir=resolved,
            agent=agent,
            model=model,
            environment=environment,
            concurrency=concurrency,
            max_retries=max_retries,
            jobs_dir=jobs_dir,
            skills_dir=skills_dir,
            sandbox_user=sandbox_user,
            agent_env=parsed_env,
        )
    else:
        _run_single(
            task_dir=resolved,
            agent=agent,
            model=model,
            environment=environment,
            prompt=prompt,
            jobs_dir=jobs_dir,
            skills_dir=skills_dir,
            sandbox_user=sandbox_user,
            agent_env=parsed_env,
        )


def _run_single(
    *,
    task_dir: Path,
    agent: str,
    model: str | None,
    environment: str,
    prompt: list[str] | None,
    jobs_dir: str,
    skills_dir: Path | None,
    sandbox_user: str | None,
    agent_env: dict[str, str],
) -> None:
    from typing import cast

    from benchflow.sdk import SDK

    sdk = SDK()
    eff_model = _effective_model(agent, model)
    result = asyncio.run(
        sdk.run(
            task_path=task_dir,
            agent=agent,
            model=eff_model,
            environment=environment,
            prompts=cast("list[str | None] | None", prompt),
            agent_env=agent_env,
            job_name="eval-create",
            jobs_dir=jobs_dir,
            skills_dir=skills_dir,
            sandbox_user=None if sandbox_user == "none" else sandbox_user,
        )
    )
    reward = getattr(result, "reward", None)
    err = getattr(result, "error", None) or getattr(result, "verifier_error", None)
    console.print()
    if err:
        console.print(f"[red]failed:[/red] {err}")
        raise typer.Exit(1)
    console.print(
        f"[bold]reward={reward}[/bold]  tools={getattr(result, 'n_tool_calls', 0)}"
    )


def _run_batch(
    *,
    tasks_dir: Path,
    agent: str,
    model: str | None,
    environment: str,
    concurrency: int,
    max_retries: int,
    jobs_dir: str,
    skills_dir: Path | None,
    sandbox_user: str | None,
    agent_env: dict[str, str],
) -> None:
    from benchflow.job import Job, JobConfig, RetryConfig

    eff_model = _effective_model(agent, model)
    config = JobConfig(
        agent=agent,
        model=eff_model,
        environment=environment,
        concurrency=concurrency,
        retry=RetryConfig(max_retries=max_retries),
        agent_env=agent_env,
        sandbox_user=None if sandbox_user == "none" else sandbox_user,
        skills_dir=str(skills_dir) if skills_dir else None,
    )
    job = Job(
        tasks_dir=tasks_dir,
        jobs_dir=Path(jobs_dir),
        config=config,
    )
    result = asyncio.run(job.run())
    console.print()
    console.print(
        f"[bold]{result.passed}/{result.total} "
        f"({result.score:.1%})[/bold] errors={result.errored}"
    )


@app.command("list")
def eval_list(
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", "-o", help="Results directory to scan"),
    ] = "jobs",
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max rows to show"),
    ] = 20,
) -> None:
    """List recent eval runs by scanning the jobs/ dir."""

    root = Path(jobs_dir)
    if not root.exists():
        console.print(f"[yellow]{root} does not exist yet.[/yellow]")
        return
    runs: list[tuple[str, int, int]] = []
    for job_root in root.iterdir():
        if not job_root.is_dir():
            continue
        for stamp in job_root.iterdir():
            if not stamp.is_dir():
                continue
            trials = list(stamp.iterdir())
            passed = 0
            total = 0
            for trial in trials:
                result = trial / "result.json"
                if not result.exists():
                    continue
                total += 1
                try:
                    import json

                    data = json.loads(result.read_text())
                    if (data.get("rewards") or {}).get("reward") == 1.0:
                        passed += 1
                except Exception:
                    continue
            if total:
                runs.append((f"{job_root.name}/{stamp.name}", passed, total))
    runs.sort(reverse=True)
    table = Table(title=f"Recent evals in {root}")
    table.add_column("run", style="cyan")
    table.add_column("passed", justify="right", style="green")
    table.add_column("total", justify="right")
    table.add_column("rate", justify="right", style="yellow")
    for run, passed, total in runs[:limit]:
        rate = f"{100 * passed / total:.0f}%" if total else "-"
        table.add_row(run, str(passed), str(total), rate)
    console.print(table)


@app.command("retrieve")
def eval_retrieve(
    trial_name: Annotated[
        str, typer.Argument(help="Trial dir name, e.g. my-task__abc")
    ],
    jobs_dir: Annotated[
        str,
        typer.Option("--jobs-dir", "-o"),
    ] = "jobs",
) -> None:
    """Print the result.json for a specific trial."""

    import json

    root = Path(jobs_dir)
    matches = list(root.rglob(f"{trial_name}/result.json"))
    if not matches:
        console.print(f"[red]no trial named {trial_name!r} under {root}[/red]")
        raise typer.Exit(1)
    console.print(f"[dim]{matches[0]}[/dim]")
    data = json.loads(matches[0].read_text())
    from rich import print_json

    print_json(data=data)
