"""benchflow CLI — agent benchmarking framework.

This module owns the top-level Typer ``app``, the global callback/version flag,
and the ``eval`` command group (``eval create`` / ``eval list``). ``eval create``
is defined here on purpose: tests pin its callback ``__module__`` to
``benchflow.cli.main`` and import it (plus the Daytona helpers) from here.

Every other command group lives in a sibling ``cli/<group>.py`` module and is
attached through a ``register_<group>(app)`` call below, mirroring the
pre-existing ``register_agent_router`` / ``register_continue`` /
``register_tasks_generate`` precedent. The shared console + display helpers live
in :mod:`benchflow.cli._shared` and are re-exported here for backwards
compatibility.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from benchflow import __version__
from benchflow._dotenv import load_dotenv_env
from benchflow._utils.config import normalize_sandbox_user
from benchflow.agents.registry import parse_agent_spec
from benchflow.cli._options import ModelOption, SkillModeOption
from benchflow.cli._shared import (
    _exit_if_evaluation_had_errors,
    _report_eval_result,
    console,
)
from benchflow.cli.agent import register_agent
from benchflow.cli.compat import register_compat
from benchflow.cli.continue_cmd import register_continue
from benchflow.cli.environment import register_environment
from benchflow.cli.legacy import register_legacy
from benchflow.cli.monitor import register_monitor
from benchflow.cli.skills import register_skills
from benchflow.cli.tasks import register_tasks
from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan
from benchflow.evaluation import DEFAULT_AGENT, effective_model
from benchflow.skill_policy import SKILL_MODE_NO_SKILL

if TYPE_CHECKING:
    from benchflow.eval_plan import EvalPlan
    from benchflow.evaluation import EvaluationConfig

# Public surface that tests and downstream callers import from
# ``benchflow.cli.main``. The Daytona helpers are defined here (not in a sibling
# module) so tests that monkeypatch ``_daytona_client_or_exit`` on ``cli.main``
# redirect the call ``_cleanup_daytona_sandboxes`` makes through this namespace.
__all__ = [
    "_cleanup_daytona_sandboxes",
    "_daytona_client_or_exit",
    "app",
    "eval_app",
    "eval_create",
]

# Show progress messages (logger.info) from benchflow internals by default.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

app = typer.Typer(
    name="benchflow",
    help="ACP-native agent benchmarking framework.",
    no_args_is_help=True,
)


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
        typer.Option("--sandbox", help="Sandbox: docker, daytona, modal, or cua"),
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
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit a machine-readable eval run report"),
    ] = False,
) -> None:
    """Run an evaluation — single task or batch.

    Sandbox: docker, daytona, modal, or cua.
    """
    _apply_dotenv_to_process_env()

    request = EvalCreateRequest(
        config_file=config_file,
        tasks_dir=tasks_dir,
        source_repo=source_repo,
        source_env=source_env,
        agent=agent,
        model=model,
        reasoning_effort=reasoning_effort,
        environment=environment,
        usage_tracking=usage_tracking,
        environment_manifest=environment_manifest,
        prompt=prompt,
        concurrency=concurrency,
        build_concurrency=build_concurrency,
        worker_concurrency=worker_concurrency,
        worker_retries=worker_retries,
        worker_start_stagger_sec=worker_start_stagger_sec,
        agent_idle_timeout=agent_idle_timeout,
        jobs_dir=jobs_dir,
        sandbox_user=sandbox_user,
        sandbox_setup_timeout=sandbox_setup_timeout,
        skills_dir=skills_dir,
        skill_mode=skill_mode,
        skill_creator_dir=skill_creator_dir,
        self_gen_no_internet=self_gen_no_internet,
        agent_env=_parse_agent_env(agent_env),
        include=include,
        exclude=exclude,
    )
    try:
        plan = build_eval_plan(request)
    except EvalPlanError as exc:
        if output_json:
            typer.echo(json.dumps(_eval_create_error_payload(str(exc))))
            raise typer.Exit(1) from None
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    if config_file:
        _run_config_file_eval(plan, output_json=output_json)
    elif source_env:
        _run_source_env_eval(
            plan,
            source_env_version=source_env_version,
            source_env_arg=source_env_arg,
            source_env_num_examples=source_env_num_examples,
            source_env_rollouts_per_example=source_env_rollouts_per_example,
            source_env_max_tokens=source_env_max_tokens,
            source_env_temperature=source_env_temperature,
            source_env_sampling_arg=source_env_sampling_arg,
            output_json=output_json,
        )
    elif source_repo:
        from benchflow._utils.benchmark_repos import resolve_source_with_metadata

        resolved = resolve_source_with_metadata(
            source_repo, path=source_path, ref=source_ref
        )
        run_batch_eval(
            plan,
            resolved.path,
            plan.make_eval_config(source_provenance=resolved.provenance),
            output_json=output_json,
        )
    elif tasks_dir:
        # Single-task and batch share one orchestration path. Evaluation
        # handles both layouts (Evaluation._get_task_dirs detects when
        # tasks_dir itself contains task.toml) and applies include/exclude
        # filters uniformly (#400, #401, #407).
        run_batch_eval(
            plan, tasks_dir, plan.make_eval_config(), output_json=output_json
        )
    else:
        if output_json:
            typer.echo(
                json.dumps(
                    _eval_create_error_payload(
                        "Provide --config, --tasks-dir, --source-repo, or --source-env"
                    )
                )
            )
            raise typer.Exit(1)
        console.print(
            "[red]Provide --config, --tasks-dir, --source-repo, or --source-env[/red]"
        )
        raise typer.Exit(1)


def run_batch_eval(
    plan: "EvalPlan",
    resolved_tasks_dir: Path,
    eval_config: "EvaluationConfig",
    *,
    output_json: bool = False,
):
    """Run the source-repo / tasks-dir batch path and report its result.

    Promoted from the ``eval_create`` ``_run_batch_eval`` closure: the worker /
    jobs-dir / manifest knobs it used to capture now ride in on ``plan``.
    """
    from benchflow.adapters.inbound import UnsupportedInboundTaskError
    from benchflow.cli._adapter_reporting import unsupported_adapter_task_or_exit
    from benchflow.eval_sharding import ShardWorkerError
    from benchflow.evaluation import EmptyTaskSelectionError, Evaluation

    try:
        if plan.request.worker_concurrency is None:
            result = asyncio.run(
                Evaluation(
                    tasks_dir=str(resolved_tasks_dir),
                    jobs_dir=plan.output_jobs_dir,
                    config=eval_config,
                ).run()
            )
        else:
            from benchflow.eval_sharding import run_sharded_evaluation

            result = asyncio.run(
                run_sharded_evaluation(
                    tasks_dir=resolved_tasks_dir,
                    jobs_dir=Path(plan.output_jobs_dir),
                    config=eval_config,
                    worker_concurrency=plan.request.worker_concurrency,
                    worker_retries=plan.request.worker_retries,
                    worker_start_stagger_sec=plan.request.worker_start_stagger_sec,
                    environment_manifest_path=plan.request.environment_manifest,
                )
            )
    except EmptyTaskSelectionError as e:
        if output_json:
            typer.echo(json.dumps(_eval_create_error_payload(str(e))))
            raise typer.Exit(1) from None
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    except UnsupportedInboundTaskError as e:
        unsupported_adapter_task_or_exit(resolved_tasks_dir, e, output_json=output_json)
    except (ValueError, RuntimeError, ShardWorkerError) as e:
        if output_json:
            typer.echo(json.dumps(_eval_create_error_payload(str(e))))
            raise typer.Exit(1) from None
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if output_json:
        typer.echo(
            json.dumps(_eval_create_result_payload(result, plan.output_jobs_dir))
        )
    else:
        _report_eval_result(result)
    _exit_if_evaluation_had_errors(result)
    return result


def _run_config_file_eval(plan: "EvalPlan", *, output_json: bool = False) -> None:
    """Apply CLI overrides onto a YAML-loaded Evaluation, then run and report it."""
    from benchflow.evaluation import EmptyTaskSelectionError, Evaluation

    req = plan.request
    config_file = req.config_file
    assert (
        config_file is not None
    )  # config-file source: build_eval_plan guarantees this
    j = Evaluation.from_yaml(config_file)
    if req.agent is not None:
        j._config.agent = plan.eval_agent
    else:
        j._config.agent = _normalize_eval_agent_or_exit(j._config.agent)
    if req.model is not None:
        j._config.model = effective_model(j._config.agent, req.model)
    else:
        j._config.model = effective_model(j._config.agent, j._config.model)
    if req.reasoning_effort is not None:
        j._config.reasoning_effort = plan.eval_reasoning_effort
    if req.environment is not None:
        j._config.environment = plan.eval_environment
    j._config.agent_env = {**j._config.agent_env, **plan.parsed_env}
    j._config.sandbox_user = normalize_sandbox_user(j._config.sandbox_user)
    if req.jobs_dir is not None:
        j._jobs_dir = Path(req.jobs_dir)
    if req.concurrency is not None:
        j._config.concurrency = req.concurrency
    if req.build_concurrency is not None:
        j._config.build_concurrency = req.build_concurrency
    if req.prompt is not None:
        j._config.prompts = plan.eval_prompts
    if req.agent_idle_timeout is not None:
        j._config.agent_idle_timeout = plan.eval_agent_idle_timeout
    if plan.usage_tracking_overridden:
        j._config.usage_tracking = j._config.usage_tracking.overlay(
            plan.eval_usage_tracking
        )
    if plan.include_tasks:
        j._config.include_tasks = plan.include_tasks
    if plan.exclude_tasks:
        j._config.exclude_tasks = plan.exclude_tasks
    # CLI --environment-manifest wins over whatever the YAML carried
    # so an operator can override a baseline manifest without editing
    # the config file.
    if plan.eval_env_manifest is not None:
        j._config.environment_manifest = plan.eval_env_manifest
    try:
        result = asyncio.run(j.run())
    except (EmptyTaskSelectionError, ValueError) as e:
        if output_json:
            typer.echo(json.dumps(_eval_create_error_payload(str(e))))
            raise typer.Exit(1) from None
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None
    if output_json:
        typer.echo(json.dumps(_eval_create_result_payload(result, j._jobs_dir)))
    else:
        _report_eval_result(result)
    _exit_if_evaluation_had_errors(result)


def _run_source_env_eval(
    plan: "EvalPlan",
    *,
    source_env_version: str | None,
    source_env_arg: list[str] | None,
    source_env_num_examples: int,
    source_env_rollouts_per_example: int,
    source_env_max_tokens: int,
    source_env_temperature: float,
    source_env_sampling_arg: list[str] | None,
    output_json: bool = False,
) -> None:
    """Warn about ignored ACP-only flags, then run a hosted Verifiers env."""
    from benchflow.hosted_env import (
        HostedEnvError,
        HostedEnvRef,
        HostedEnvRunConfig,
        parse_sampling_args,
        parse_source_env_args,
        run_hosted_env,
    )

    req = plan.request
    if plan.parsed_env and not output_json:
        console.print(
            "[yellow]--agent-env is for BenchFlow ACP agents; source-env runs inherit the process environment.[/yellow]"
        )
    if plan.eval_environment != "docker" and not output_json:
        console.print(
            f"[yellow]--sandbox {plan.eval_environment!r} is not used by source-env runs; "
            "the hosted Verifiers environment owns its harness/sandbox.[/yellow]"
        )
    if plan.eval_agent != DEFAULT_AGENT and not output_json:
        console.print(
            f"[dim]source-env records --agent {plan.eval_agent!r}, but executes the model endpoint through Verifiers.[/dim]"
        )
    if plan.eval_env_manifest is not None and not output_json:
        console.print(
            "[yellow]--environment-manifest is for benchflow Environment-plane rollouts; "
            "the hosted Verifiers environment owns its own provisioning. Ignoring.[/yellow]"
        )
    if plan.usage_tracking_overridden and not output_json:
        console.print(
            "[yellow]--usage-tracking is for BenchFlow ACP rollouts; "
            "source-env runs own their provider calls. Ignoring.[/yellow]"
        )
    if req.prompt is not None and not output_json:
        console.print(
            "[yellow]--prompt is for BenchFlow ACP rollouts; "
            "source-env runs own their prompts. Ignoring.[/yellow]"
        )

    source_env = req.source_env
    assert source_env is not None  # source-env source: build_eval_plan guarantees this
    try:
        ref = HostedEnvRef.parse(source_env, version=source_env_version)
        run_result = run_hosted_env(
            HostedEnvRunConfig(
                source_env=ref,
                model=req.model or "",
                env_args=parse_source_env_args(source_env_arg),
                agent=plan.eval_agent,
                jobs_dir=Path(plan.output_jobs_dir),
                concurrency=plan.eval_concurrency,
                num_examples=source_env_num_examples,
                rollouts_per_example=source_env_rollouts_per_example,
                max_tokens=source_env_max_tokens,
                temperature=source_env_temperature,
                sampling_args=parse_sampling_args(source_env_sampling_arg),
            )
        )
    except HostedEnvError as e:
        if output_json:
            typer.echo(json.dumps(_eval_create_error_payload(str(e))))
            raise typer.Exit(1) from None
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": "completed" if not run_result.error else "error",
                    "source": "source-env",
                    "source_env": run_result.source_env.env_uid,
                    "hub": run_result.source_env.hub_url,
                    "model": run_result.model,
                    "normalized_model": run_result.normalized_model,
                    "run_dir": str(run_result.run_dir),
                    "reward": run_result.reward,
                    "total_tool_calls": run_result.total_tool_calls,
                    "error": run_result.error,
                }
            )
        )
        if run_result.error:
            raise typer.Exit(1)
        return

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


def _eval_create_error_payload(reason: str) -> dict[str, object]:
    """Machine-readable failure payload for eval-adoption automation."""
    return {
        "status": "error",
        "ok": False,
        "reason": reason,
    }


def _eval_create_result_payload(
    result: object,
    jobs_dir: str | Path,
) -> dict[str, object]:
    """Return a JSON-safe eval run report backed by the persisted summary."""
    jobs_path = Path(jobs_dir)
    summary_path = jobs_path / "summary.json"
    summary: dict[str, object] | None = None
    summary_error: str | None = None
    if summary_path.exists():
        try:
            loaded_summary = json.loads(summary_path.read_text())
            if isinstance(loaded_summary, dict):
                summary = loaded_summary
            else:
                summary_error = "summary.json did not contain a JSON object"
        except (OSError, json.JSONDecodeError) as exc:
            summary_error = str(exc)

    errored = _as_int(getattr(result, "errored", 0))
    verifier_errored = _as_int(getattr(result, "verifier_errored", 0))
    status = "completed-with-errors" if errored or verifier_errored else "completed"
    payload: dict[str, object] = {
        "status": status,
        "ok": status == "completed",
        "jobs_dir": str(jobs_path),
        "summary_path": str(summary_path) if summary_path.exists() else None,
        "result": {
            "job_name": getattr(result, "job_name", None),
            "total": _as_int(getattr(result, "total", 0)),
            "passed": _as_int(getattr(result, "passed", 0)),
            "failed": _as_int(getattr(result, "failed", 0)),
            "errored": errored,
            "verifier_errored": verifier_errored,
            "score": _as_float(getattr(result, "score", None)),
            "score_excl_errors": _as_float(getattr(result, "score_excl_errors", None)),
            "elapsed_sec": _as_float(getattr(result, "elapsed_sec", None)),
        },
        "summary": summary,
    }
    if summary_error is not None:
        payload["summary_error"] = summary_error
    return payload


def _as_int(value: object) -> int:
    if value is None:
        return 0
    if not isinstance(value, int | float | str):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


# ── Command-group wiring ──────────────────────────────────────────────
#
# Each ``register_<group>(app)`` attaches one command group defined in a sibling
# ``cli/<group>.py`` module, mirroring the existing ``register_continue`` /
# ``register_tasks_generate`` / ``register_agent_router`` precedent. Order does
# not affect behavior; it follows the historical top-level help ordering.
register_continue(app)
register_legacy(app)
register_skills(app)
register_tasks(app)
register_compat(app)
register_agent(app)
register_environment(app)
register_monitor(app)


if __name__ == "__main__":
    app()
