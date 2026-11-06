"""CLI-side artifact handling for ``bench eval run``."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import typer
from rich.markup import escape

from benchflow.cli._shared import command_arg, console, print_error

if TYPE_CHECKING:
    from benchflow.eval_plan import EvalPlan
    from benchflow.evaluation import EvaluationConfig, EvaluationResult


def _count_next(manifest: Path) -> str:
    """Keep the selected-task manifest path literal in the recovery step."""
    return (
        f"{command_arg(manifest, '<manifest file>')} lists the selected tasks; "
        "fix --include/--exclude or --expected-tasks"
    )


def _finished_next(job_dir: Path) -> str:
    """The next step when a step after the trials fails: they are not lost."""
    return "fix the cause above; the trials finished and are kept in " + command_arg(
        job_dir, "<job dir>"
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _redacted_eval_config(eval_config: EvaluationConfig) -> dict:
    return {
        "agent": eval_config.agent,
        "model": eval_config.model,
        "reasoning_effort": eval_config.reasoning_effort,
        "harness": eval_config.harness,
        "codex_apps_policy": eval_config.codex_apps_policy,
        "environment": eval_config.environment,
        "concurrency": eval_config.concurrency,
        "build_concurrency": eval_config.build_concurrency,
        "warm_pool": eval_config.warm_pool,
        "skill_mode": eval_config.skill_mode,
        "skills_dir": eval_config.skills_dir,
        "usage_tracking": eval_config.usage_tracking.to_config_artifact(),
        "agent_env_keys": sorted(eval_config.agent_env),
        "reviewer": eval_config.reviewer.to_config_artifact(),
        "include_tasks": sorted(eval_config.include_tasks),
        "exclude_tasks": sorted(eval_config.exclude_tasks),
        "source_provenance": eval_config.source_provenance,
        "dataset_name": eval_config.dataset_name,
        "dataset_version": eval_config.dataset_version,
    }


def _apply_source_sidecar(
    eval_config: EvaluationConfig, resolved_tasks_dir: Path
) -> EvaluationConfig:
    if eval_config.source_provenance is not None:
        return eval_config
    from benchflow._utils.hf_datasets import load_source_sidecar

    eval_config.source_provenance = load_source_sidecar(resolved_tasks_dir)
    return eval_config


def postprocess_eval_artifacts(
    plan: EvalPlan,
    resolved_tasks_dir: Path,
    eval_config: EvaluationConfig,
    job_dir: Path | None,
) -> None:
    req = plan.request
    if job_dir is None:
        return
    eval_config = _apply_source_sidecar(eval_config, resolved_tasks_dir)
    from benchflow.eval_artifacts import (
        TaskManifestOptions,
        canonical_task_rows,
        materialize_canonical_job,
        render_run_summary_markdown,
        write_canonical_selection,
        write_health_summary,
        write_task_manifest,
    )

    if req.task_manifest_out is not None:
        manifest = write_task_manifest(
            req.task_manifest_out,
            TaskManifestOptions(
                tasks_dir=resolved_tasks_dir,
                include_tasks=eval_config.include_tasks,
                exclude_tasks=eval_config.exclude_tasks,
                source_provenance=eval_config.source_provenance,
                dataset_name=eval_config.dataset_name,
                dataset_version=eval_config.dataset_version,
                dataset_task_digests=eval_config.dataset_task_digests,
            ),
        )
        if req.expected_tasks is not None and manifest["total"] != req.expected_tasks:
            print_error(
                f"selected task count {manifest['total']} != expected {req.expected_tasks}",
                next_step=_count_next(req.task_manifest_out),
            )
            raise typer.Exit(1)
        console.print(
            f"[green]Task manifest:[/green] {escape(str(req.task_manifest_out))}"
        )
    if req.run_config_out is not None:
        _write_json(
            req.run_config_out,
            {
                "schema_version": 1,
                "jobs_dir": str(job_dir),
                "eval": _redacted_eval_config(eval_config),
                "retry_policy": req.retry_policy,
                "retry_attempts": req.retry_attempts,
                "retry_concurrency": req.retry_concurrency,
            },
        )
        console.print(f"[green]Run config:[/green] {escape(str(req.run_config_out))}")
    if req.health_summary_out is not None:
        write_health_summary(req.health_summary_out, job_dir)
        console.print(
            f"[green]Health summary:[/green] {escape(str(req.health_summary_out))}"
        )
    if req.canonicalize != "none" and req.canonical_selection_out is not None:
        try:
            policy = cast(Literal["one-healthy-per-task"], req.canonicalize)
            write_canonical_selection(
                req.canonical_selection_out,
                job_dir,
                policy=policy,
                expected_tasks=req.expected_tasks,
            )
        except ValueError as exc:
            print_error(str(exc), next_step=_finished_next(job_dir))
            raise typer.Exit(1) from None
        console.print(
            "[green]Canonical selection:[/green] "
            f"{escape(str(req.canonical_selection_out))}"
        )
        if req.canonical_jobs_dir is not None:
            try:
                materialize_canonical_job(
                    req.canonical_selection_out, req.canonical_jobs_dir
                )
            except ValueError as exc:
                print_error(str(exc), next_step=_finished_next(job_dir))
                raise typer.Exit(1) from None
            console.print(
                f"[green]Canonical jobs:[/green] {escape(str(req.canonical_jobs_dir))}"
            )
    if req.publish_hf is not None or req.publish_bucket is not None:
        (job_dir / "README.md").write_text(
            render_run_summary_markdown(
                job_dir, agent=eval_config.agent, model=eval_config.model
            )
        )
    if req.publish_hf is not None:
        from benchflow.publish.huggingface import publish_folder_to_hf

        prefix = req.hf_prefix or job_dir.name
        try:
            published = publish_folder_to_hf(
                job_dir,
                repo_id=req.publish_hf,
                path_in_repo=prefix,
                repo_type="dataset",
                public_read_check=req.hf_public_read_check,
            )
        except ValueError as exc:
            print_error(str(exc), next_step=_finished_next(job_dir))
            raise typer.Exit(1) from None
        console.print(f"[green]Published HF artifacts:[/green] {escape(published.url)}")
    if req.publish_bucket is not None:
        from benchflow.publish.huggingface import publish_folder_to_bucket

        prefix = req.hf_prefix or job_dir.name
        try:
            published = publish_folder_to_bucket(
                job_dir, bucket_id=req.publish_bucket, path_in_repo=prefix
            )
        except ValueError as exc:
            print_error(str(exc), next_step=_finished_next(job_dir))
            raise typer.Exit(1) from None
        console.print(f"[green]Published HF bucket:[/green] {escape(published.url)}")
    if req.eval_results_model is not None:
        from benchflow.publish.huggingface import open_eval_results_pr

        rows = canonical_task_rows(job_dir)
        rewards = [row["reward"] for row in rows if row["scored"]]
        if rows:
            value = round(100.0 * sum(rewards) / len(rows), 2)
            try:
                pr_url = open_eval_results_pr(
                    model_repo=req.eval_results_model,
                    dataset_id=cast(str, req.eval_results_dataset),
                    task_id=cast(str, req.eval_results_task),
                    value=value,
                    notes=f"Mean reward over canonical tasks (n = {len(rows)}, "
                    f"unscored as 0; scored n = {len(rewards)}).",
                )
            except ValueError as exc:
                print_error(str(exc), next_step=_finished_next(job_dir))
                raise typer.Exit(1) from None
            console.print(f"[green]Eval-results PR:[/green] {escape(pr_url or '')}")
        else:
            console.print(
                "[yellow]--eval-results-model set but no rollouts were "
                "found; skipping eval-results PR[/yellow]"
            )


# What a --matrix file looks like; the loader's errors and docs/reference/
# pass-at-k.md show it (DF-p3-eval-runner-3).
MATRIX_FORMAT = (
    "models: maps each alias (its folder name) to a model name, or to a "
    "mapping with model, and optionally agent and agent_env; an agent that "
    "takes no model (oracle, nop) needs only agent, e.g. "
    "models: {haiku: claude-haiku-4-5, oracle: {agent: oracle}}"
)
_NOT_MODELS = {"", "none", "null", "~"}


def _load_eval_matrix(path: Path) -> dict[str, dict]:
    import yaml

    from benchflow.agents.registry import is_scripted_agent

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError(f"--matrix must be a YAML mapping whose {MATRIX_FORMAT}")
    models = {}
    for alias, raw in data["models"].items():
        if isinstance(raw, str) and raw.strip().lower() not in _NOT_MODELS:
            models[str(alias)] = {"model": raw}
        elif isinstance(raw, dict) and raw.get("model"):
            if str(raw["model"]).strip().lower() in _NOT_MODELS:
                raise ValueError(
                    f"matrix model {alias!r}: {raw['model']!r} is not a model name; "
                    f"{MATRIX_FORMAT}"
                )
            models[str(alias)] = dict(raw)
        elif (
            isinstance(raw, dict)
            and isinstance(raw.get("agent"), str)
            and is_scripted_agent(raw["agent"])
            and raw.get("model") is None
        ):
            models[str(alias)] = {**raw, "model": None}
        elif raw is None or isinstance(raw, str):
            raise ValueError(
                f"matrix model {alias!r}: {raw!r} is not a model name; {MATRIX_FORMAT}"
            )
        else:
            raise ValueError(f"matrix model {alias!r}: {MATRIX_FORMAT}")
    return models


def _matrix_agent_env(raw: object) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    raise ValueError("matrix agent_env must be a mapping")


def _matrix_file_path(path: Path | None, alias: str, trial: int) -> Path | None:
    if path is None:
        return None
    return path.parent / alias / f"trial-{trial:02d}" / path.name


def _matrix_dir_path(path: Path | None, alias: str, trial: int) -> Path | None:
    if path is None:
        return None
    return path / alias / f"trial-{trial:02d}"


def _write_matrix_task_manifest(
    plan: EvalPlan, resolved_tasks_dir: Path, eval_config: EvaluationConfig
) -> None:
    req = plan.request
    if req.task_manifest_out is None:
        return
    eval_config = _apply_source_sidecar(eval_config, resolved_tasks_dir)
    from benchflow.eval_artifacts import TaskManifestOptions, write_task_manifest

    manifest = write_task_manifest(
        req.task_manifest_out,
        TaskManifestOptions(
            tasks_dir=resolved_tasks_dir,
            include_tasks=eval_config.include_tasks,
            exclude_tasks=eval_config.exclude_tasks,
            source_provenance=eval_config.source_provenance,
            dataset_name=eval_config.dataset_name,
            dataset_version=eval_config.dataset_version,
            dataset_task_digests=eval_config.dataset_task_digests,
        ),
    )
    if req.expected_tasks is not None and manifest["total"] != req.expected_tasks:
        print_error(
            f"selected task count {manifest['total']} != expected {req.expected_tasks}",
            next_step=_count_next(req.task_manifest_out),
        )
        raise typer.Exit(1)
    console.print(f"[green]Task manifest:[/green] {escape(str(req.task_manifest_out))}")


def matrix_solve_rates(
    root: Path,
    aliases: list[str],
    *,
    include_controls: bool = False,
    ks: list[int] | None = None,
) -> dict[str, Any]:
    """pass@k / pass^k per matrix entry, pooling its ``trial-NN`` folders.

    None for an entry with no readable trial. A job matrix scores every
    cell, controls (oracle, nop) included, at the default ks
    and at its n_attempts (review B-M1, B-L1).
    """
    import benchflow as bf

    out: dict[str, Any] = {}
    for alias in aliases:
        try:
            out[alias] = (
                bf.load_job(root / alias)
                .solve_rates(include_controls=include_controls, ks=ks)
                .to_dict()
            )
        except FileNotFoundError:
            out[alias] = None
    return out


def run_matrix_eval(
    plan: EvalPlan,
    resolved_tasks_dir: Path,
    run_batch_eval: Callable[..., EvaluationResult],
    *,
    config_overrides: dict[str, Any] | None = None,
) -> None:
    """Run each matrix entry ``--trials`` times, one job per entry and trial,
    in ``<jobs dir>/<alias>/trial-NN/``, then write matrix-summary.json.

    Without ``--matrix`` (``--trials N`` alone) the one entry is the run's
    own agent and model, under the agent's name. ``config_overrides`` are
    the source's ``make_eval_config`` keywords (a ``--dataset``'s
    provenance, digests and task selection).
    """
    import re

    from benchflow.task.discovery import resolve_task_collection_root

    req = plan.request
    overrides = config_overrides or {}
    if req.matrix is not None:
        try:
            models = _load_eval_matrix(req.matrix)
        except (OSError, ValueError) as exc:
            print_error(
                str(exc),
                next_step="fix the --matrix file; docs/reference/pass-at-k.md "
                "shows one",
            )
            raise typer.Exit(1) from None
    else:
        own = plan.make_eval_config(**overrides)
        alias = re.sub(r"[^A-Za-z0-9._-]+", "-", own.agent).strip("-") or "agent"
        models = {alias: {"model": own.model}}
    root = Path(plan.output_jobs_dir)
    task_collection_dir = resolve_task_collection_root(resolved_tasks_dir)
    _write_matrix_task_manifest(
        plan, task_collection_dir, plan.make_eval_config(**overrides)
    )
    # Every trial runs: a trial job applies no gate and never exits on its
    # own errors (its request's matrix_trial); the matrix gates once, below,
    # over every trial, and matrix-summary.json is written however the loop
    # ends (DX-F-MATRIX-STOP).
    matrix_runs: list[dict[str, Any]] = []
    results: list[_TrialJob] = []
    complete = False
    restore_reuse = _plan_layer_reuse(req.trials)
    try:
        for alias, raw in models.items():
            for trial in range(1, req.trials + 1):
                trial_dir = root / alias / f"trial-{trial:02d}"
                trial_plan = replace(
                    plan,
                    request=_trial_request(req, alias, trial),
                    output_jobs_dir=str(trial_dir),
                    parsed_env={
                        **plan.parsed_env,
                        **_matrix_agent_env(raw.get("agent_env")),
                    },
                )
                eval_config = trial_plan.make_eval_config(**overrides)
                # None: an agent that takes no model (oracle, nop).
                eval_config.model = None if raw["model"] is None else str(raw["model"])
                if raw.get("agent"):
                    eval_config.agent = str(raw["agent"])
                    eval_config.agent_defaulted = False
                try:
                    result = run_batch_eval(
                        trial_plan, task_collection_dir, eval_config
                    )
                except typer.Exit as stop:
                    if stop.exit_code != 1:
                        raise  # an interrupt or a usage error ends the matrix
                    # The trial job itself failed (its error is printed):
                    # recorded, and the next trial runs.
                    matrix_runs.append(_failed_run(alias, trial, trial_dir))
                    continue
                job = _trial_job(alias, trial, result, eval_config)
                results.append(job)
                matrix_runs.append(_trial_run(job, trial_dir))
        complete = True
    finally:
        restore_reuse()
        whole = _write_matrix_summary(
            root,
            matrix=str(req.matrix) if req.matrix is not None else None,
            trials=req.trials,
            complete=complete,
            aliases=list(models),
            results=results,
            matrix_runs=matrix_runs,
        )
    _report_matrix(req, whole, root, matrix_runs)


def _plan_layer_reuse(trials: int) -> Callable[[], None]:
    """With more than one trial, later trials start each task's image again:
    Daytona's cached install layers are then worth one cold build
    (``daytona_layers.claim_layer_build``). Returns what restores the
    environment."""
    from benchflow.sandbox.daytona_layers import LAYER_REUSE_ENV

    before = os.environ.get(LAYER_REUSE_ENV)
    if trials > 1:
        os.environ[LAYER_REUSE_ENV] = "1"

    def put_back() -> None:
        if before is None:
            os.environ.pop(LAYER_REUSE_ENV, None)
        else:
            os.environ[LAYER_REUSE_ENV] = before

    return put_back


def _trial_request(req: Any, alias: str, trial: int) -> Any:
    """One trial job's request: its artifact paths under ``<alias>/trial-NN``,
    and no gate or run summary of its own (the matrix's, once, at the end)."""
    return replace(
        req,
        hf_prefix=(
            f"{req.hf_prefix.rstrip('/')}/{alias}/trial-{trial:02d}"
            if req.hf_prefix
            else None
        ),
        task_manifest_out=None,
        run_config_out=_matrix_file_path(req.run_config_out, alias, trial),
        health_summary_out=_matrix_file_path(req.health_summary_out, alias, trial),
        canonical_selection_out=_matrix_file_path(
            req.canonical_selection_out, alias, trial
        ),
        canonical_jobs_dir=_matrix_dir_path(req.canonical_jobs_dir, alias, trial),
        matrix=None,
        fail_under=None,
        fail_on=None,
        summary_out=None,
        matrix_trial=True,
    )


def _failed_run(alias: str, trial: int, trial_dir: Path) -> dict[str, Any]:
    return {
        "alias": alias,
        "trial": trial,
        "jobs_dir": str(trial_dir),
        "job_failed": True,
    }


def _trial_job(alias: str, trial: int, result: Any, config: Any) -> _TrialJob:
    control = _control_kind(config.agent, config.model)
    return _TrialJob(alias, trial, result, config.agent, control)


def _trial_run(job: _TrialJob, trial_dir: Path) -> dict[str, Any]:
    result = job.result
    return {
        "alias": job.alias,
        "trial": job.trial,
        "jobs_dir": str(trial_dir),
        "passed": result.passed,
        "failed": result.failed,
        "errored": result.errored,
        "verifier_errored": result.verifier_errored,
        "total": result.total,
        "not_run": getattr(result, "not_run", 0),
        "control": job.control,
    }


def _write_matrix_summary(
    root: Path,
    *,
    matrix: str | None,
    trials: int,
    complete: bool,
    aliases: list[str],
    results: list[_TrialJob],
    matrix_runs: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    include_controls: bool = False,
    ks: list[int] | None = None,
) -> Any:
    """Write ``<root>/matrix-summary.json``; returns the whole matrix's result."""
    whole = _matrix_result(root, results, matrix_runs)
    summary_path = root / "matrix-summary.json"
    _write_json(
        summary_path,
        {
            # 2: solve_rates count every trial (an unscored one as a fail),
            # with the scored-only figures under *_scored_only.
            "schema_version": 2,
            "matrix": matrix,
            "trials": trials,
            # False when an interrupt stopped the matrix before its end.
            "complete": complete,
            **(extra or {}),
            "runs": matrix_runs,
            "solve_rates": matrix_solve_rates(
                root, aliases, include_controls=include_controls, ks=ks
            ),
            # What --fail-under counts: control runs are
            # left out unless every trial job is one.
            "gate": {
                "trials": whole.total,
                "agents": whole.gated_agents,
                "controls_left_out": whole.controls_left_out,
                "control_kinds": whole.control_kinds,
                "controls_counted": whole.controls_counted,
            },
        },
    )
    console.print(f"[green]Matrix summary:[/green] {escape(str(summary_path))}")
    return whole


def _report_matrix(
    req: Any, whole: Any, root: Path, matrix_runs: list[dict[str, Any]]
) -> None:
    """The matrix's closing line, its gate, and the run's end (exit code)."""
    from benchflow.cli.main import _finish_run

    console.print(
        f"Matrix: {len(matrix_runs)} trial job(s), {whole.all_passed}/{whole.all_total} "
        f"trials passed, errors={whole.errored} verifier-errors="
        f"{whole.verifier_errored}"
        + (
            f"; {whole.all_not_run} of {whole.all_planned} planned trials not run, so "
            f"{whole.all_passed}/{whole.all_planned} "
            f"({whole.all_passed / whole.all_planned:.1%}) of the plan"
            if whole.all_not_run
            else ""
        )
        + (
            f", {whole.jobs_failed} trial job(s) failed to run"
            if whole.jobs_failed
            else ""
        ),
        highlight=False,
        soft_wrap=True,
    )
    if req.fail_under is not None:
        console.print(gate_line(whole), highlight=False, soft_wrap=True)
    _finish_run(req, whole, root)


def run_job_matrix(
    plan: EvalPlan,
    job_plan: Any,
    root: Path,
    load: Callable[[int, Path, EvalPlan], Any],
    run_job: Callable[[EvalPlan, Any, str], Any],
    rerun: str,
) -> None:
    """Run a YAML job config of several jobs: each
    cell (an agent with a dataset) ``job_plan.attempts`` times, one job per
    cell and trial in ``<root>/<cell>/trial-NN/``, then matrix-summary.json
    with pass@k per cell, controls included, at the default ks and at k =
    n_attempts. ``load(cell, trial_dir, trial_plan)`` builds a trial's
    Evaluation with the command line's overrides; ``run_job(trial_plan,
    evaluation, resume)`` runs and reports it (``resume`` is the line a stop
    prints), and exits 1 when the job fails to run (recorded; the next trial
    runs). ``rerun`` is the command that resumes the whole matrix: finished
    trial jobs are kept and the rest run."""
    from benchflow.pass_at_k import default_ks

    req = plan.request
    total = len(job_plan.cells) * job_plan.attempts
    matrix_runs: list[dict[str, Any]] = []
    results: list[_TrialJob] = []
    # What each cell ran, after the command line's overrides (review B-M4).
    ran: dict[int, tuple[str | None, str | None]] = {}
    complete = False
    restore_reuse = _plan_layer_reuse(job_plan.attempts)
    try:
        for cell in job_plan.cells:
            for trial in range(1, job_plan.attempts + 1):
                trial_dir = root / cell.alias / f"trial-{trial:02d}"
                trial_plan = replace(
                    plan,
                    request=_trial_request(req, cell.alias, trial),
                    output_jobs_dir=str(trial_dir),
                )
                console.print(
                    f"[bold]{escape(cell.alias)}[/bold] trial {trial}/{job_plan.attempts}",
                    highlight=False,
                )
                evaluation = load(cell.index, trial_dir, trial_plan)
                ran.setdefault(
                    cell.index, (evaluation._config.agent, evaluation._config.model)
                )
                left = total - len(matrix_runs)
                resume = (
                    f"run the same command again to finish the whole matrix "
                    f"({left} of {total} trial jobs not finished; finished ones "
                    f"are kept): {rerun}"
                )
                try:
                    result = run_job(trial_plan, evaluation, resume)
                except typer.Exit as stop:
                    if stop.exit_code != 1:
                        raise
                    matrix_runs.append(_failed_run(cell.alias, trial, trial_dir))
                    continue
                job = _trial_job(cell.alias, trial, result, evaluation._config)
                results.append(job)
                matrix_runs.append(_trial_run(job, trial_dir))
        complete = True
    finally:
        restore_reuse()
        ks = sorted({*default_ks(job_plan.attempts), job_plan.attempts})
        whole = _write_matrix_summary(
            root,
            matrix=str(req.config_file),
            trials=job_plan.attempts,
            complete=complete,
            aliases=[cell.alias for cell in job_plan.cells],
            results=results,
            matrix_runs=matrix_runs,
            extra={
                "job_cells": [
                    {
                        "alias": cell.alias,
                        "agent": ran.get(cell.index, (cell.agent, cell.model))[0],
                        "model": ran.get(cell.index, (cell.agent, cell.model))[1],
                        "dataset": cell.dataset,
                    }
                    for cell in job_plan.cells
                ],
                "resume": None if complete else rerun,
            },
            # Every configured agent is scored (B-M1).
            include_controls=True,
            ks=ks,
        )
    _report_matrix(req, whole, root, matrix_runs)


@dataclass(frozen=True)
class _TrialJob:
    """One trial job of a matrix and what ran it."""

    alias: str
    trial: int
    result: Any  # EvaluationResult, or a caller's stand-in
    agent: str | None
    # "oracle" or "nop" for a control run (the rule bf.load_job uses), else None.
    control: str | None


def _control_kind(agent: str | None, model: str | None) -> str | None:
    """``oracle`` or ``nop`` when a trial job runs a control, else None
    (``Trial.control``'s rule: the oracle, or an empty agent with no model)."""
    name = (agent or "").strip().lower()
    if name == "oracle":
        return "oracle"
    if name in ("nop", "noop", "empty") and not model:
        return "nop"
    return None


def gate_line(whole: Any) -> str:
    """What --fail-under counted over a matrix, e.g. ``Gate:
    0.75 over 8 trials of claude-code; 4 control trials (nop) left out``."""
    line = (
        f"Gate: {whole.score_of_plan:.2f} over {whole.planned} trials of "
        f"{', '.join(whole.gated_agents) or 'no agent'}"
    )
    if whole.controls_left_out:
        line += (
            f"; {whole.controls_left_out} control trials "
            f"({', '.join(whole.control_kinds)}) left out"
        )
    elif whole.controls_counted:
        line += " (every trial is a control run, so they are counted)"
    return line


def _matrix_result(
    root: Path,
    results: list[_TrialJob],
    runs: list[dict[str, Any]],
) -> Any:
    """The whole matrix as one result for the gate and --summary-out.

    Counting is P4's every-trial rule: an errored or unscored trial is not
    passed, and a trial job that failed to run is an error. The gate's
    figures (``total``, ``passed``, ``score``, ``score_of_plan``) follow the
    controls rule of every read command: control trial jobs
    (oracle, nop) are left out, unless every trial job is one. Errors count
    over every trial job, so an errored control still fails the run.
    ``all_*`` are the figures over every trial.
    """
    from types import SimpleNamespace

    from benchflow._utils.scoring import pass_rate

    def count(jobs: list[_TrialJob], name: str) -> int:
        # getattr: a sharded run's result, or a caller's, may lack a field.
        return sum(int(getattr(j.result, name, 0) or 0) for j in jobs)

    agent_jobs = [j for j in results if j.control is None]
    control_jobs = [j for j in results if j.control is not None]
    gated = agent_jobs or control_jobs
    left_out = control_jobs if agent_jobs else []
    total, passed, failed = (
        count(gated, "total"),
        count(gated, "passed"),
        count(gated, "failed"),
    )
    not_run = count(gated, "not_run")
    jobs_failed = sum(1 for run in runs if run.get("job_failed"))
    rewards: list[tuple[float, int]] = [
        (float(mean), int(j.result.total))
        for j in gated
        if (mean := getattr(j.result, "mean_reward", None)) is not None
        and getattr(j.result, "total", 0)
    ]
    weight = sum(n for _, n in rewards)
    merged = {
        f"{j.alias}/trial-{j.trial:02d}/{name}": rollout
        for j in results
        for name, rollout in (getattr(j.result, "results", None) or {}).items()
    }
    planned = total + not_run
    all_total, all_passed = count(results, "total"), count(results, "passed")
    all_not_run = count(results, "not_run")
    return SimpleNamespace(
        job_name=root.name,
        total=total,
        passed=passed,
        failed=failed,
        # Over every trial job: a trial job that failed to run is an error.
        errored=count(results, "errored") + jobs_failed,
        verifier_errored=count(results, "verifier_errored"),
        jobs_failed=jobs_failed,
        not_run=not_run,
        planned=planned,
        score=pass_rate(passed=passed, total=total),
        score_of_plan=pass_rate(passed=passed, total=planned),
        mean_reward=(sum(m * n for m, n in rewards) / weight) if weight else None,
        mean_reward_scored_only=None,
        results=merged,
        reused=count(gated, "reused"),
        ran=count(gated, "ran"),
        budget=None,
        gated_agents=sorted({j.agent or "?" for j in gated}),
        controls_left_out=count(left_out, "total"),
        control_kinds=sorted({j.control or "" for j in left_out}),
        controls_counted=not agent_jobs and bool(control_jobs),
        all_total=all_total,
        all_passed=all_passed,
        all_not_run=all_not_run,
        all_planned=all_total + all_not_run,
    )
