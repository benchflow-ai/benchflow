"""benchflow CLI — agent benchmarking framework.

This module owns the top-level Typer ``app``, the global callback/version flag,
and the ``eval`` command group (``eval run`` / ``eval list``). ``eval run``
is defined here on purpose: tests pin its callback ``__module__`` to
``benchflow.cli.main`` and import it (plus the Daytona helpers) from here.
(``eval create`` remains as a deprecated alias of ``eval run``.)

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
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow import __version__
from benchflow._dotenv import load_dotenv_env
from benchflow._utils.config import normalize_sandbox_user
from benchflow.agents.registry import parse_agent_spec
from benchflow.cli._live_progress import (
    LiveEvalProgress,
    live_session,
    progress_enabled,
)
from benchflow.cli._options import AgentOption, ModelOption, SkillModeOption
from benchflow.cli._shared import (
    _exit_if_evaluation_had_errors,
    _report_eval_result,
    console,
    err_console,
    print_error,
)
from benchflow.cli.adopt import register_adopt_deprecated, register_eval_adopt
from benchflow.cli.agent import register_agent
from benchflow.cli.continue_cmd import register_continue
from benchflow.cli.environment import register_environment
from benchflow.cli.eval_artifacts import postprocess_eval_artifacts, run_matrix_eval
from benchflow.cli.eval_lift import register_eval_lift
from benchflow.cli.hub import register_hub
from benchflow.cli.monitor import register_monitor
from benchflow.cli.sandbox import register_sandbox
from benchflow.cli.skills import register_skills
from benchflow.cli.tasks import register_tasks
from benchflow.cli.train import register_train
from benchflow.eval_plan import EvalCreateRequest, EvalPlanError, build_eval_plan
from benchflow.evaluation import DEFAULT_AGENT, effective_model
from benchflow.sandbox.providers import providers_phrase
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
    "eval_run",
    "eval_create",  # deprecated import alias of eval_run
]

# Show progress messages (logger.info) from benchflow internals by default.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

_TAGLINE = "The universal environment framework — run, author, and adopt agent benchmarks across any environment."

app = typer.Typer(
    name="benchflow",
    help=_TAGLINE,
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
    """The universal environment framework — run, author, and adopt agent benchmarks."""


def _parse_agent_env(entries: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            print_error(f"Invalid env var: {entry}")
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
        print_error(f"Unsupported eval agent protocol: {protocol}")
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
            print_error(f"daytona SDK import failed: {exc}")
        raise typer.Exit(1) from None
    except Exception as exc:
        print_error(f"daytona SDK import failed: {exc}")
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
app.add_typer(eval_app, name="eval", rich_help_panel="Core")
# Canonical single `bench eval adopt` command (eval is the universal benchmark
# entry point; adopt makes a foreign benchmark runnable).
register_eval_adopt(eval_app)
register_eval_lift(eval_app)


@eval_app.command("run")
def eval_run(
    config_file: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "--run-config",
            help="YAML run-config file (the whole run spec)",
        ),
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
        typer.Option("--sandbox", help=f"Sandbox: {providers_phrase()}"),
    ] = None,
    usage_tracking: Annotated[
        str | None,
        typer.Option(
            "--usage-tracking",
            help=(
                "Telemetry-enforcement policy: auto, required, or off. The "
                "LiteLLM proxy is always used for routable agents (usage, cost, "
                "and llm_trajectory.jsonl are always captured); this flag only "
                "controls whether trusted telemetry is required."
            ),
        ),
    ] = None,
    environment_manifest: Annotated[
        Path | None,
        typer.Option(
            "--environment-manifest",
            help=(
                "Environment-plane manifest applied to every rollout: a path to "
                "an environment.toml, OR a 'name@version' registry spec resolved "
                "via $BENCHFLOW_ENV_REGISTRY (the S axis). The manifest-declared "
                "stateful environment is provisioned, gated on readiness, and "
                "torn down."
            ),
        ),
    ] = None,
    state: Annotated[
        str | None,
        typer.Option(
            "--state",
            help=(
                "S-axis environment binding, decoupled from the task. Inline JSON "
                'with an optional tool subset (e.g. {"name": "env0", "tools": '
                '["gmail", "gcal"]}), OR a name@version spec, OR a manifest path. '
                "Takes precedence over --environment-manifest."
            ),
        ),
    ] = None,
    prompt: Annotated[
        list[str] | None,
        typer.Option("--prompt", help="Prompt(s) to send (default: instruction.md)"),
    ] = None,
    config_override: Annotated[
        str | None,
        typer.Option(
            "--config-override",
            help=(
                "C-axis overlay: deep-merge a config patch into each task's "
                "resolved config for this run. Inline JSON/YAML/TOML or an @file "
                'ref, e.g. --config-override \'{"agent":{"timeout_sec":120}}\'. Varies one '
                "knob (budget/skills/stopping rules) while T/A/M/S/R stay fixed; "
                "recorded by content hash for replay."
            ),
        ),
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
    context_root: Annotated[
        Path | None,
        typer.Option(
            "--context-root",
            help=(
                "Repo/build-context root used to stage Dockerfile COPY sources "
                "for monorepo-authored local tasks."
            ),
        ),
    ] = None,
    base_image_override: Annotated[
        str | None,
        typer.Option(
            "--base-image-override",
            help=(
                "Rewrite task Dockerfile FROM images on the runtime task copy. "
                "Use only for reproducing runs whose base image moved namespaces."
            ),
        ),
    ] = None,
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
    loop_strategy: Annotated[
        str | None,
        typer.Option(
            "--loop-strategy",
            help=(
                "Harness loop strategy, e.g. 'verify-retry:k=3,feedback=names', "
                "'self-review:k=3', or 'single-shot' (default). verify-retry "
                "re-prompts with filtered soft-verifier feedback; self-review "
                "re-prompts the agent to critique its OWN work (no verifier "
                "signal) — both loop until the soft verifier passes or k retries "
                "are spent."
            ),
        ),
    ] = None,
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
    dataset: Annotated[
        str | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Registry dataset to run, as <name>@<version> (e.g. skillsbench@1.1). "
            "Resolves the pinned snapshot and verifies task content digests.",
        ),
    ] = None,
    registry: Annotated[
        str | None,
        typer.Option(
            "--registry",
            help="Dataset registry JSON URL or local file "
            "(default: the skillsbench registry). Only valid with --dataset.",
        ),
    ] = None,
    ignore_bench_version: Annotated[
        bool,
        typer.Option(
            "--ignore-bench-version",
            help="Run a dataset even when this bench version is outside the "
            "range it was validated against. Only valid with --dataset.",
        ),
    ] = False,
    task_manifest_out: Annotated[
        Path | None,
        typer.Option(
            "--task-manifest-out", help="Write selected task-set manifest JSON"
        ),
    ] = None,
    run_config_out: Annotated[
        Path | None,
        typer.Option(
            "--run-config-out", help="Write redacted normalized run config JSON"
        ),
    ] = None,
    health_summary_out: Annotated[
        Path | None,
        typer.Option(
            "--health-summary-out", help="Write trajectory health summary JSON"
        ),
    ] = None,
    expected_tasks: Annotated[
        int | None,
        typer.Option(
            "--expected-tasks", help="Fail unless the selected task count matches"
        ),
    ] = None,
    canonicalize: Annotated[
        str,
        typer.Option(
            "--canonicalize",
            help="Canonicalization policy: none or one-healthy-per-task",
        ),
    ] = "none",
    canonical_selection_out: Annotated[
        Path | None,
        typer.Option(
            "--canonical-selection-out",
            help="Write canonical rollout-selection JSON",
        ),
    ] = None,
    canonical_jobs_dir: Annotated[
        Path | None,
        typer.Option(
            "--canonical-jobs-dir",
            help="Materialize selected rollout dirs for trainer conversion",
        ),
    ] = None,
    retry_policy: Annotated[
        str,
        typer.Option("--retry-policy", help="Retry policy: default or unscored-only"),
    ] = "default",
    retry_attempts: Annotated[
        int | None,
        typer.Option("--retry-attempts", help="Reserved retry-attempt override"),
    ] = None,
    max_retries: Annotated[
        int | None,
        typer.Option(
            "--max-retries",
            help="Alias for --retry-attempts; maximum retries per task.",
        ),
    ] = None,
    retry_concurrency: Annotated[
        int | None,
        typer.Option("--retry-concurrency", help="Reserved retry concurrency"),
    ] = None,
    publish_hf: Annotated[
        str | None,
        typer.Option(
            "--publish-hf", help="Upload final eval artifacts to a HF dataset repo"
        ),
    ] = None,
    hf_prefix: Annotated[
        str | None,
        typer.Option("--hf-prefix", help="Path prefix in the HF repo"),
    ] = None,
    hf_public_read_check: Annotated[
        bool,
        typer.Option(
            "--hf-public-read-check", help="Verify public HF reads after upload"
        ),
    ] = False,
    matrix: Annotated[
        Path | None,
        typer.Option("--matrix", help="YAML model matrix for repeated evals"),
    ] = None,
    trials: Annotated[
        int,
        typer.Option("--trials", help="Number of trials for --matrix"),
    ] = 1,
) -> None:
    """Run an evaluation — single task or batch.

    Sandbox: docker, daytona, or modal.
    """
    _apply_dotenv_to_process_env()

    if retry_attempts is not None and max_retries is not None:
        print_error("--retry-attempts and --max-retries are aliases; pass only one")
        raise typer.Exit(1)
    retry_attempt_override = (
        retry_attempts if retry_attempts is not None else max_retries
    )

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
        state=state,
        config_override=config_override,
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
        context_root=context_root,
        base_image_override=base_image_override,
        skills_dir=skills_dir,
        skill_mode=skill_mode,
        skill_creator_dir=skill_creator_dir,
        self_gen_no_internet=self_gen_no_internet,
        loop_strategy=loop_strategy,
        agent_env=_parse_agent_env(agent_env),
        include=include,
        exclude=exclude,
        dataset=dataset,
        registry=registry,
        ignore_bench_version=ignore_bench_version,
        task_manifest_out=task_manifest_out,
        run_config_out=run_config_out,
        health_summary_out=health_summary_out,
        expected_tasks=expected_tasks,
        canonicalize=canonicalize,
        canonical_selection_out=canonical_selection_out,
        canonical_jobs_dir=canonical_jobs_dir,
        retry_policy=retry_policy,
        retry_attempts=retry_attempt_override,
        retry_concurrency=retry_concurrency,
        publish_hf=publish_hf,
        hf_prefix=hf_prefix,
        hf_public_read_check=hf_public_read_check,
        matrix=matrix,
        trials=trials,
    )
    # --source-path/--source-ref only apply to --source-repo; otherwise they're
    # silently ignored (e.g. `--dataset X --source-ref abc` drops the ref).
    if (source_path or source_ref) and not source_repo:
        print_error("--source-path/--source-ref require --source-repo")
        raise typer.Exit(1)
    try:
        plan = build_eval_plan(request)
    except EvalPlanError as exc:
        print_error(f"{exc}")
        raise typer.Exit(1) from None

    if config_file:
        _run_config_file_eval(plan)
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
        )
    elif source_repo:
        import subprocess

        from benchflow._utils.benchmark_repos import resolve_source_with_metadata
        from benchflow.adapters.source import adapt_resolved_source_if_needed

        try:
            resolved = resolve_source_with_metadata(
                source_repo, path=source_path, ref=source_ref
            )
            resolved = adapt_resolved_source_if_needed(resolved)
        except (subprocess.CalledProcessError, OSError, ValueError, RuntimeError) as e:
            # A clone failure (missing/private repo, bad --source-ref, auth,
            # network) otherwise escapes as a raw traceback — unlike the sibling
            # config/source-env/dataset branches which all map to a clean error.
            print_error(f"Could not resolve --source-repo {source_repo!r}: {e}")
            raise typer.Exit(1) from None
        run_batch_eval(
            plan,
            resolved.path,
            plan.make_eval_config(source_provenance=resolved.provenance),
        )
    elif tasks_dir:
        # Single-task and batch share one orchestration path. Evaluation
        # handles both layouts (Evaluation._get_task_dirs detects when
        # tasks_dir itself contains task.toml) and applies include/exclude
        # filters uniformly (#400, #401, #407).
        if not Path(tasks_dir).is_dir():
            # Without this guard a file (or missing path) reaches iterdir() and
            # dumps a raw NotADirectoryError, unlike sandbox create / eval metrics.
            print_error(f"Not a directory: {tasks_dir}")
            raise typer.Exit(1)
        if matrix is not None:
            run_matrix_eval(plan, tasks_dir, run_batch_eval)
        else:
            run_batch_eval(plan, tasks_dir, plan.make_eval_config())
    elif dataset:
        from benchflow._utils.dataset_registry import (
            DEFAULT_REGISTRY_SOURCE,
            DatasetResolutionError,
            bench_version_issue,
            resolve_dataset,
        )

        registry_source = registry or DEFAULT_REGISTRY_SOURCE
        try:
            with console.status(f"Resolving dataset {dataset}…"):
                resolved_dataset = resolve_dataset(dataset, registry=registry_source)
        except DatasetResolutionError as e:
            print_error(f"{e}")
            raise typer.Exit(1) from None
        version_issue = bench_version_issue(resolved_dataset.bench_version)
        if version_issue and not ignore_bench_version:
            # Hard gate: published results must come from a harness the
            # dataset version was validated against. The escape hatch keeps
            # local experimentation possible without weakening the default.
            print_error(f"{version_issue}")
            # The remediation hint is part of the error — keep it on stderr too so
            # a `--json` consumer's stdout stays clean.
            err_console.print(
                "Pick a dataset version validated for this harness, or re-run "
                "with --ignore-bench-version to proceed anyway."
            )
            raise typer.Exit(1)
        if version_issue:
            console.print(
                f"[yellow]Warning:[/yellow] {escape(str(version_issue))} (--ignore-bench-version)"
            )
        console.print(
            f"[green]✓[/green] {escape(str(resolved_dataset.spec))}: "
            f"{len(resolved_dataset.task_names)} tasks, digests verified "
            f"({str(resolved_dataset.provenance.get('resolved_sha', ''))[:12]})"
        )
        # tasks_dir is the resolved source checkout (a superset); restrict to
        # the dataset's pinned task set, further narrowed by any --include.
        dataset_include = (
            resolved_dataset.task_names & plan.include_tasks
            if plan.include_tasks
            else resolved_dataset.task_names
        )
        run_batch_eval(
            plan,
            resolved_dataset.tasks_dir,
            plan.make_eval_config(
                source_provenance=resolved_dataset.provenance,
                dataset_name=resolved_dataset.name,
                dataset_version=resolved_dataset.version,
                dataset_task_digests=resolved_dataset.task_digests,
                include_tasks=dataset_include,
            ),
        )
    else:
        console.print(
            "[red]Provide --config, --tasks-dir, --source-repo, --source-env, "
            "or --dataset[/red]"
        )
        raise typer.Exit(1)


def _eval_label(plan: "EvalPlan", tasks_dir: Path) -> str:
    """A short source descriptor for the live dashboard header."""
    req = plan.request
    for attr in ("source_repo", "dataset", "source_env"):
        val = getattr(req, attr, None)
        if val:
            return str(val)
    return tasks_dir.name


def run_batch_eval(
    plan: "EvalPlan",
    resolved_tasks_dir: Path,
    eval_config: "EvaluationConfig",
):
    """Run the source-repo / tasks-dir batch path and report its result.

    Promoted from the ``eval_run`` ``_run_batch_eval`` closure: the worker /
    jobs-dir / manifest knobs it used to capture now ride in on ``plan``.
    """
    from benchflow.eval_sharding import ShardWorkerError
    from benchflow.evaluation import EmptyTaskSelectionError, Evaluation
    from benchflow.task.discovery import resolve_task_collection_root

    task_collection_dir = resolve_task_collection_root(resolved_tasks_dir)
    if eval_config.source_provenance is None:
        from benchflow._utils.hf_datasets import load_source_sidecar

        eval_config.source_provenance = load_source_sidecar(task_collection_dir)

    try:
        if plan.request.retry_attempts is not None:
            eval_config.retry.max_retries = plan.request.retry_attempts
        if plan.request.worker_concurrency is None:
            # One Evaluation construction; the live dashboard contributes hooks +
            # a context manager on a TTY, and a null context + no hooks otherwise
            # (CI/pipes keep the plain logger stream).
            hooks: dict = {}
            run_ctx: AbstractContextManager = nullcontext()
            if progress_enabled(console):
                live = LiveEvalProgress(
                    console,
                    label=_eval_label(plan, task_collection_dir),
                    agent=eval_config.agent,
                    model=eval_config.model,
                    sandbox=eval_config.environment,
                )
                hooks = {
                    "on_plan": live.on_plan,
                    "on_task_start": live.on_task_start,
                    "on_result": live.on_result,
                }
                run_ctx = live_session(live)
            with run_ctx:
                result = asyncio.run(
                    Evaluation(
                        tasks_dir=str(task_collection_dir),
                        jobs_dir=plan.output_jobs_dir,
                        config=eval_config,
                        **hooks,
                    ).run()
                )
        else:
            from benchflow.eval_sharding import run_sharded_evaluation

            result = asyncio.run(
                run_sharded_evaluation(
                    tasks_dir=task_collection_dir,
                    jobs_dir=Path(plan.output_jobs_dir),
                    config=eval_config,
                    worker_concurrency=plan.request.worker_concurrency,
                    worker_retries=plan.request.worker_retries,
                    worker_start_stagger_sec=plan.request.worker_start_stagger_sec,
                )
            )
    except EmptyTaskSelectionError as e:
        print_error(f"{e}")
        raise typer.Exit(1) from None
    except (ValueError, RuntimeError, ShardWorkerError) as e:
        print_error(f"{e}")
        raise typer.Exit(1) from None

    job_name = getattr(result, "job_name", None)
    job_dir = Path(plan.output_jobs_dir) / job_name if job_name else None
    postprocess_eval_artifacts(plan, task_collection_dir, eval_config, job_dir)
    _report_eval_result(result, job_dir)
    _exit_if_evaluation_had_errors(result)
    return result


def _run_config_file_eval(plan: "EvalPlan") -> None:
    """Apply CLI overrides onto a YAML-loaded Evaluation, then run and report it."""
    import subprocess

    import yaml

    from benchflow.evaluation import EmptyTaskSelectionError, Evaluation

    req = plan.request
    config_file = req.config_file
    assert (
        config_file is not None
    )  # config-file source: build_eval_plan guarantees this
    # from_yaml + the override overlay run BEFORE j.run(); a malformed config
    # surfaces here (bad YAML, missing source.repo, a list where a mapping is
    # expected, a non-existent/invalid environment_manifest, …). Guard the
    # whole block so the CLI prints one clean error instead of ~10 distinct raw
    # tracebacks. The exception type is kept in the message for diagnosability.
    try:
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
        if req.retry_attempts is not None:
            j._config.retry.max_retries = req.retry_attempts
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
        # CLI --config / --config-override (the C axis) likewise wins over the
        # YAML's own config_override. Parsed + allowlist-validated at plan time
        # and applied per task at the rollout layer. Without this the
        # file-config path silently dropped the overlay (it threaded every
        # other override but never this one), so a --config-override on a
        # run-config file was a no-op.
        if plan.eval_config_override is not None:
            j._config.config_override = plan.eval_config_override
    except subprocess.CalledProcessError as e:
        # A source.repo clone/fetch failure (git exits non-zero) otherwise escapes
        # as a raw traceback — it is not a config-parse error, so give it its own
        # clean message. Mirrors the --source-repo guard's CalledProcessError catch.
        console.print(
            f"[red]Failed to fetch source repo for {escape(str(config_file))}:[/red] "
            f"{escape(str(e))}"
        )
        raise typer.Exit(1) from None
    except (yaml.YAMLError, ValueError, TypeError, LookupError, OSError) as e:
        # LookupError covers missing source.repo (KeyError) and empty legacy
        # agents:/datasets: lists (IndexError). Type-mismatch cases (e.g. a
        # non-string source.repo) are converted to ValueError at the parse
        # source rather than catching AttributeError here, which would mask
        # genuine bugs.
        console.print(
            f"[red]Invalid eval config {escape(str(config_file))}:[/red] "
            f"{type(e).__name__}: {escape(str(e))}"
        )
        raise typer.Exit(1) from None
    try:
        result = asyncio.run(j.run())
    except (EmptyTaskSelectionError, ValueError) as e:
        print_error(f"{e}")
        raise typer.Exit(1) from None
    job_name = getattr(result, "job_name", None)
    job_dir = Path(j._jobs_dir) / job_name if job_name else None
    _report_eval_result(result, job_dir)
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
    if plan.parsed_env:
        console.print(
            "[yellow]--agent-env is for BenchFlow ACP agents; source-env runs inherit the process environment.[/yellow]"
        )
    if plan.eval_environment != "docker":
        console.print(
            f"[yellow]--sandbox {escape(repr(plan.eval_environment))} is not used by source-env runs; "
            "the hosted Verifiers environment owns its harness/sandbox.[/yellow]"
        )
    if plan.eval_agent != DEFAULT_AGENT:
        console.print(
            f"[dim]source-env records --agent {escape(repr(plan.eval_agent))}, but executes the model endpoint through Verifiers.[/dim]"
        )
    if plan.eval_env_manifest is not None:
        console.print(
            "[yellow]--environment-manifest is for benchflow Environment-plane rollouts; "
            "the hosted Verifiers environment owns its own provisioning. Ignoring.[/yellow]"
        )
    if plan.usage_tracking_overridden:
        console.print(
            "[yellow]--usage-tracking is for BenchFlow ACP rollouts; "
            "source-env runs own their provider calls. Ignoring.[/yellow]"
        )
    if req.prompt is not None:
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
        print_error(f"{e}")
        raise typer.Exit(1) from None

    console.print(f"\n[bold]Environment:[/bold] {run_result.source_env.env_uid}")
    console.print(f"[bold]Hub:[/bold] {escape(str(run_result.source_env.hub_url))}")
    console.print(
        f"[bold]Model:[/bold] {escape(str(run_result.normalized_model))}"
        + (
            f" [dim](from {run_result.model})[/dim]"
            if run_result.normalized_model != run_result.model
            else ""
        )
    )
    console.print(f"[bold]Run dir:[/bold] {escape(str(run_result.run_dir))}")
    console.print(f"[bold]Reward:[/bold] {run_result.reward}")
    if run_result.total_tool_calls is not None:
        console.print(f"[bold]Tool calls:[/bold] {run_result.total_tool_calls}")
    if run_result.error:
        console.print(f"[red]Error:[/red] {escape(str(run_result.error))}")
        raise typer.Exit(1)


# `bench eval create` was renamed to `bench eval run`. Keep the old name as a
# visible, deprecated alias so existing scripts, configs, and downstream repos
# (e.g. benchflow-ai/skillsbench) keep working; Click prints a deprecation
# notice when the alias is invoked.
eval_app.command("create", deprecated=True)(eval_run)

# Back-compat for the Python symbol: `eval_create` was part of this module's
# public surface (``__all__``). Keep it as an import alias of ``eval_run`` so any
# `from benchflow.cli.main import eval_create` keeps resolving.
eval_create = eval_run


@eval_app.command("list")
def eval_list(
    jobs_dir: Annotated[
        Path | None,
        typer.Argument(help="Jobs directory to list (default: ./jobs)"),
    ] = None,
) -> None:
    """List completed evaluations."""
    # None means the argument was omitted: the default ./jobs simply not existing
    # yet is a benign first-run state (exit 0). An *explicit* path that doesn't
    # exist is a typo and should fail like `eval metrics` does, so scripts don't
    # read it as success. (A literal `eval list jobs` is then treated as explicit,
    # which is correct — the user named a path.)
    explicit = jobs_dir is not None
    jobs_dir = jobs_dir or Path("jobs")
    if not jobs_dir.exists():
        if not explicit:
            console.print("[yellow]No jobs yet.[/yellow]")
            return
        print_error(f"No such jobs directory: {jobs_dir}")
        raise typer.Exit(1)
    if not jobs_dir.is_dir():
        # exists() is True for a file; iterdir() below would NotADirectoryError.
        console.print(f"[red]Not a directory: {escape(str(jobs_dir))}[/red]")
        raise typer.Exit(1)

    table = Table(title="Evaluations")
    table.add_column("Evaluation", style="cyan")
    table.add_column("Tasks", justify="right")
    table.add_column("Summary")
    table.add_column("Memory")

    def memory_label(data: dict) -> str:
        score = data.get("memory_score")
        return f"{score:.1%}" if isinstance(score, int | float) else "—"

    def add_summary_row(label: str, summary_path: Path) -> None:
        # A corrupt / truncated / non-object summary.json must not abort the
        # whole listing — degrade that one row, mirroring the "no summary"
        # fallback (and matching collect_metrics / _get_completed_tasks, which
        # both skip-guard).
        try:
            data = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            data = None
        if not isinstance(data, dict):
            table.add_row(escape(label), "?", "[dim]corrupt summary[/dim]", "—")
            return
        table.add_row(
            escape(label),
            str(data.get("total", "?")),
            f"{data.get('passed', '?')}/{data.get('total', '?')} ({data.get('score', '?')})",
            memory_label(data),
        )

    root_summary = jobs_dir / "summary.json"
    if root_summary.exists():
        add_summary_row(jobs_dir.name, root_summary)
        console.print(table)
        return

    for d in sorted(jobs_dir.iterdir()):
        if not d.is_dir():
            continue
        summary = d / "summary.json"
        if summary.exists():
            add_summary_row(d.name, summary)
        else:
            sub_count = sum(1 for s in d.iterdir() if s.is_dir())
            table.add_row(escape(d.name), str(sub_count), "[dim]no summary[/dim]", "—")

    console.print(table)


@eval_app.command("metrics")
def eval_metrics(
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

    if not Path(jobs_dir).is_dir():
        # Without this, collect_metrics rglobs nothing and reports a green
        # all-zeros table with exit 0 — a silent trap for scripted collectors.
        print_error(f"Not a directory: {jobs_dir}")
        raise typer.Exit(1)
    m = collect_metrics(str(jobs_dir), benchmark=benchmark, agent=agent, model=model)
    summary = m.summary()

    if output_json:
        console.print(json.dumps(summary, indent=2))
        return

    table = Table(title=f"Results: {escape(str(jobs_dir))}")
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
            f"\n[green]Passed:[/green] {escape(', '.join(summary['passed_tasks']))}"
        )
    if summary["errored_tasks"]:
        console.print(
            f"[yellow]Errors:[/yellow] {escape(', '.join(summary['errored_tasks']))}"
        )
    if summary["error_breakdown"]:
        console.print(
            f"[yellow]Error breakdown:[/yellow] {escape(str(summary['error_breakdown']))}"
        )


@eval_app.command("view")
def eval_view(
    rollout_dir: Annotated[
        Path,
        typer.Argument(help="Rollout or job directory with trajectories"),
    ],
    port: Annotated[int, typer.Option(help="Server port")] = 8888,
) -> None:
    """View a trial trajectory in the browser."""
    from benchflow.trajectories.viewer import serve

    serve(str(rollout_dir), port)


# ── Command-group wiring ──────────────────────────────────────────────
#
# Each ``register_<group>(app)`` attaches one command group defined in a sibling
# ``cli/<group>.py`` module, mirroring the existing ``register_continue`` /
# ``register_tasks_generate`` / ``register_agent_router`` precedent. Order does
# not affect behavior; it follows the historical top-level help ordering.
register_continue(eval_app, alias_app=app)
register_skills(app)
register_tasks(app)
register_train(app)
register_hub(app)
register_agent(app)
register_adopt_deprecated(app)
register_sandbox(app)
register_environment(app)
register_monitor(app)


if __name__ == "__main__":
    app()
