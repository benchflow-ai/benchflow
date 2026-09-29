"""``bench hillclimb``: automated eval hill-climbing with a held-out test split.

Each flag is a field of :class:`benchflow.HillclimbConfig` (the proposer's
flags are fields of :class:`benchflow.hillclimbing.ProposerSettings`), so
``bf.hillclimb(...)`` takes the same settings. See docs/hillclimb.md.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._shared import (
    _apply_dotenv_to_process_env,
    _parse_agent_env,
    console,
    print_error,
)
from benchflow.sandbox.providers import providers_phrase

# The CLI flag of every HillclimbConfig / ProposerSettings field it sets
# (tests/test_hillclimb_cli.py checks both directions).
FLAG_FIELDS: dict[str, str] = {
    "--tasks-dir": "tasks",
    "--task": "tasks",
    "--surface": "surface",
    "--out": "out",
    "--agent": "agent",
    "--model": "model",
    "--reasoning-effort": "reasoning_effort",
    "--sandbox": "environment",
    "--concurrency": "concurrency",
    "--agent-env": "agent_env",
    "--sandbox-user": "sandbox_user",
    "--agent-idle-timeout": "agent_idle_timeout",
    "--retry-attempts": "retry_attempts",
    "--config-override": "config_override",
    "--include": "include",
    "--exclude": "exclude",
    "--test-frac": "test_frac",
    "--seed": "seed",
    "--split-file": "split_file",
    "--stratify-by": "stratify_by",
    "--objective": "objective",
    "--rounds": "rounds",
    "--trials": "trials",
    "--min-gain": "min_gain",
    "--max-cost-usd": "max_cost_usd",
    "--stall-rounds": "stall_rounds",
    "--candidates": "candidates",
    "--max-infra-error-rate": "max_infra_error_rate",
    "--skip-controls": "controls",
    "--exclude-broken-tasks": "exclude_broken_tasks",
    "--force": "force",
    "--leak-check": "leak_check",
    "--bootstrap-samples": "bootstrap_samples",
    "--analyze-at-end": "analyze_at_end",
    "--proposer-agent": "proposer.agent",
    "--proposer-model": "proposer.model",
    "--proposer-reasoning-effort": "proposer.reasoning_effort",
    "--proposer-sandbox": "proposer.environment",
    "--proposer-env": "proposer.agent_env",
    "--proposer-timeout": "proposer.timeout_sec",
    "--proposer-image": "proposer.image",
    "--proposer-open-network": "proposer.open_network",
    "--proposer-max-failures": "proposer.max_failures",
}


def register_hillclimb(app: typer.Typer) -> None:
    """Attach ``bench hillclimb`` to the top-level app."""

    @app.command("hillclimb", rich_help_panel="Core")
    def hillclimb_cmd(
        surface: Annotated[
            list[str],
            typer.Option(
                "--surface",
                help=(
                    "What the optimizer may edit; repeatable. A directory is a skills "
                    "folder (deployed like --skills-dir), a file a prompt prepended to "
                    "every task prompt. Prefix skills= or prompt= to be explicit."
                ),
            ),
        ],
        tasks_dir: Annotated[
            Path | None,
            typer.Option("--tasks-dir", help="Folder of tasks to split and climb on"),
        ] = None,
        task: Annotated[
            list[Path] | None,
            typer.Option(
                "--task", help="A task folder; repeatable (instead of --tasks-dir)"
            ),
        ] = None,
        out: Annotated[
            Path | None,
            typer.Option(
                "--out",
                help="Run folder (default: jobs/hillclimb/<timestamp>); must be new",
            ),
        ] = None,
        agent: Annotated[
            str, typer.Option("--agent", help="Agent under test")
        ] = "claude-agent-acp",
        model: Annotated[
            str | None, typer.Option("--model", help="Model of the agent under test")
        ] = None,
        reasoning_effort: Annotated[
            str | None,
            typer.Option(
                "--reasoning-effort", help="Reasoning effort of the agent under test"
            ),
        ] = None,
        sandbox: Annotated[
            str, typer.Option("--sandbox", help=f"Sandbox: {providers_phrase()}")
        ] = "docker",
        concurrency: Annotated[
            int,
            typer.Option(
                "--concurrency",
                min=1,
                help="Rollouts at once, shared by an evaluation's trials and splits",
            ),
        ] = 4,
        agent_env: Annotated[
            list[str] | None,
            typer.Option("--agent-env", help="Agent under test env var (KEY=VALUE)"),
        ] = None,
        sandbox_user: Annotated[
            str | None,
            typer.Option("--sandbox-user", help="Sandbox user (none for root)"),
        ] = "agent",
        agent_idle_timeout: Annotated[
            str | None,
            typer.Option(
                "--agent-idle-timeout",
                help="Idle seconds before a prompt is aborted (default 600; 0 or none disables)",
            ),
        ] = "600",
        retry_attempts: Annotated[
            int | None,
            typer.Option(
                "--retry-attempts",
                min=0,
                help="Retries per trial after an infrastructure error (default 2)",
            ),
        ] = None,
        config_override: Annotated[
            str | None,
            typer.Option(
                "--config-override",
                help="Config overlay for every trial (JSON/YAML/TOML or @file), as in bench eval run",
            ),
        ] = None,
        include: Annotated[
            list[str] | None,
            typer.Option("--include", help="Only these task names; repeatable"),
        ] = None,
        exclude: Annotated[
            list[str] | None,
            typer.Option("--exclude", help="Skip these task names; repeatable"),
        ] = None,
        test_frac: Annotated[
            float,
            typer.Option(
                "--test-frac", help="Share of tasks held out as the test split"
            ),
        ] = 0.3,
        seed: Annotated[
            int, typer.Option("--seed", help="Seed of the split and the bootstrap")
        ] = 0,
        split_file: Annotated[
            Path | None,
            typer.Option(
                "--split-file",
                help='JSON {"train": [...], "test": [...]} instead of a random split',
            ),
        ] = None,
        stratify_by: Annotated[
            str,
            typer.Option(
                "--stratify-by",
                help="Task metadata key the random split is stratified by (none: off)",
            ),
        ] = "category",
        objective: Annotated[
            str,
            typer.Option(
                "--objective",
                help="score (raise it), or cost (cut it while the score holds within noise)",
            ),
        ] = "score",
        rounds: Annotated[
            int, typer.Option("--rounds", min=0, help="Rounds of propose and evaluate")
        ] = 5,
        trials: Annotated[
            int, typer.Option("--trials", min=1, help="Trials per task per evaluation")
        ] = 3,
        min_gain: Annotated[
            float,
            typer.Option(
                "--min-gain",
                help=(
                    "Smallest train gain worth keeping: reward points for score "
                    "(0.05 = 5 points), a fraction of cost for cost (0.1 = 10%)"
                ),
            ),
        ] = 0.05,
        max_cost_usd: Annotated[
            float | None,
            typer.Option(
                "--max-cost-usd",
                help="Stop before spending more than this (agent under test plus proposer)",
            ),
        ] = None,
        stall_rounds: Annotated[
            int,
            typer.Option(
                "--stall-rounds",
                min=1,
                help="Stop and analyze the failures after this many rounds with nothing kept",
            ),
        ] = 3,
        candidates: Annotated[
            int,
            typer.Option(
                "--candidates",
                min=1,
                help="Patches proposed per round from the same base",
            ),
        ] = 1,
        max_infra_error_rate: Annotated[
            float,
            typer.Option(
                "--max-infra-error-rate",
                help="Stop when more than this share of an evaluation's trials has no score",
            ),
        ] = 0.25,
        skip_controls: Annotated[
            bool,
            typer.Option(
                "--skip-controls",
                help="Do not run the oracle and a do-nothing agent to check the graders",
            ),
        ] = False,
        exclude_broken_tasks: Annotated[
            bool,
            typer.Option(
                "--exclude-broken-tasks",
                help="Drop tasks whose oracle fails or where doing nothing passes",
            ),
        ] = False,
        force: Annotated[
            bool,
            typer.Option(
                "--force",
                help="Climb even when the noise gate refuses (the run is marked ungated)",
            ),
        ] = False,
        leak_check: Annotated[
            str,
            typer.Option(
                "--leak-check",
                help="Patches that paste train instructions or grader output: reject, warn or off",
            ),
        ] = "reject",
        bootstrap_samples: Annotated[
            int,
            typer.Option("--bootstrap-samples", min=100, help="Bootstrap replicates"),
        ] = 2000,
        analyze_at_end: Annotated[
            bool,
            typer.Option(
                "--analyze-at-end",
                help="Also sort the remaining train failures by cause when the rounds run out",
            ),
        ] = False,
        proposer_agent: Annotated[
            str, typer.Option("--proposer-agent", help="The optimizer agent")
        ] = "claude-agent-acp",
        proposer_model: Annotated[
            str | None, typer.Option("--proposer-model", help="The optimizer's model")
        ] = None,
        proposer_reasoning_effort: Annotated[
            str | None,
            typer.Option(
                "--proposer-reasoning-effort", help="The optimizer's reasoning effort"
            ),
        ] = None,
        proposer_sandbox: Annotated[
            str | None,
            typer.Option(
                "--proposer-sandbox",
                help="The optimizer's sandbox (default: --sandbox)",
            ),
        ] = None,
        proposer_env: Annotated[
            list[str] | None,
            typer.Option("--proposer-env", help="Optimizer env var (KEY=VALUE)"),
        ] = None,
        proposer_timeout: Annotated[
            int,
            typer.Option(
                "--proposer-timeout", min=60, help="Optimizer time limit (seconds)"
            ),
        ] = 1800,
        proposer_image: Annotated[
            str | None,
            typer.Option(
                "--proposer-image",
                help="Optimizer sandbox image (default: pinned python)",
            ),
        ] = None,
        proposer_open_network: Annotated[
            bool,
            typer.Option(
                "--proposer-open-network",
                help="Give the optimizer network access (it could then fetch public test tasks)",
            ),
        ] = False,
        proposer_max_failures: Annotated[
            int,
            typer.Option(
                "--proposer-max-failures",
                min=1,
                help="Failed train trials shown to the optimizer per round",
            ),
        ] = 24,
        quiet: Annotated[
            bool,
            typer.Option("--quiet", help="Suppress per-rollout progress output"),
        ] = False,
    ) -> None:
        """Hill-climb a skills folder or prompt against a held-out test split.

        Splits the tasks into train and test, checks the graders and the noise,
        then each round has an optimizer agent edit the surface once from the
        train failures, and keeps the edit only if train gains at least
        --min-gain and test improves. The optimizer's sandbox never holds the
        test split. Writes hillclimb.json, report.html, surface-history/ and
        every evaluation as normal BenchFlow jobs under --out.
        """
        import os

        from benchflow._utils.config import normalize_agent_idle_timeout
        from benchflow.cli.doctor import eval_preflight
        from benchflow.evaluation import effective_model
        from benchflow.hillclimbing import (
            HillclimbConfig,
            HillclimbError,
            ProposerSettings,
            hillclimb,
            summarize,
        )
        from benchflow.hillclimbing.evaluate import TaskSetError
        from benchflow.hillclimbing.proposer import default_image
        from benchflow.hillclimbing.split import SplitError
        from benchflow.hillclimbing.surface import SurfaceError

        _apply_dotenv_to_process_env()
        if quiet:
            os.environ["BENCHFLOW_PROGRESS"] = "off"
            os.environ["BENCHFLOW_NO_PROGRESS"] = "1"
        if (tasks_dir is None) == (not task):
            print_error("give either --tasks-dir or one or more --task")
            raise typer.Exit(1)
        if objective not in ("score", "cost"):
            print_error("--objective is score or cost")
            raise typer.Exit(1)
        if leak_check not in ("reject", "warn", "off"):
            print_error("--leak-check is reject, warn or off")
            raise typer.Exit(1)
        overlay = None
        if config_override is not None:
            from benchflow._utils.config_override import (
                load_config_override,
                validate_overlay,
            )

            try:
                overlay = load_config_override(config_override)
                if overlay:
                    validate_overlay(overlay)
            except (OSError, ValueError) as exc:
                print_error(f"--config-override: {exc}")
                raise typer.Exit(1) from None
        try:
            resolved_model = effective_model(agent, model)
            idle = normalize_agent_idle_timeout(agent_idle_timeout)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        agent_env_map = _parse_agent_env(agent_env)
        eval_preflight(
            sandbox=sandbox, agent=agent, model=resolved_model, agent_env=agent_env_map
        )
        run_dir = out or Path("jobs") / "hillclimb" / datetime.now().strftime(
            "%Y-%m-%d__%H-%M-%S"
        )
        try:
            config = HillclimbConfig(
                tasks=str(tasks_dir)
                if tasks_dir is not None
                else [str(t) for t in task or []],
                surface=list(surface),
                out=run_dir,
                agent=agent,
                model=resolved_model,
                reasoning_effort=reasoning_effort,
                environment=sandbox,
                concurrency=concurrency,
                agent_env=agent_env_map,
                sandbox_user=sandbox_user,
                agent_idle_timeout=idle,
                retry_attempts=retry_attempts,
                config_override=overlay,
                include=list(include or []),
                exclude=list(exclude or []),
                test_frac=test_frac,
                seed=seed,
                split_file=split_file,
                stratify_by=None
                if stratify_by.lower() in ("", "none")
                else stratify_by,
                objective="cost" if objective == "cost" else "score",
                rounds=rounds,
                trials=trials,
                min_gain=min_gain,
                max_cost_usd=max_cost_usd,
                stall_rounds=stall_rounds,
                candidates=candidates,
                max_infra_error_rate=max_infra_error_rate,
                controls=not skip_controls,
                exclude_broken_tasks=exclude_broken_tasks,
                force=force,
                leak_check="warn"
                if leak_check == "warn"
                else "off"
                if leak_check == "off"
                else "reject",
                bootstrap_samples=bootstrap_samples,
                analyze_at_end=analyze_at_end,
                proposer=ProposerSettings(
                    agent=proposer_agent,
                    model=proposer_model,
                    reasoning_effort=proposer_reasoning_effort,
                    environment=proposer_sandbox or sandbox,
                    agent_env=_parse_agent_env(proposer_env),
                    timeout_sec=proposer_timeout,
                    image=proposer_image or default_image(),
                    open_network=proposer_open_network,
                    max_failures=proposer_max_failures,
                ),
            )
            result = hillclimb(config)
        except (HillclimbError, TaskSetError, SplitError, SurfaceError) as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None
        console.print(escape(summarize(result)))
        # 0: finished (or stopped by the budget with a best version);
        # 2: refused by the noise gate; 1: stopped by infrastructure errors.
        if result.status == "refused":
            raise typer.Exit(2)
        stop = result.record.stop
        if result.status == "stopped" and stop is not None and stop.reason == "infra":
            raise typer.Exit(1)
