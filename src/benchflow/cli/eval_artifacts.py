"""CLI-side artifact handling for ``bench eval run``."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error

if TYPE_CHECKING:
    from benchflow.eval_plan import EvalPlan
    from benchflow.evaluation import EvaluationConfig, EvaluationResult


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _redacted_eval_config(eval_config: EvaluationConfig) -> dict:
    return {
        "agent": eval_config.agent,
        "model": eval_config.model,
        "reasoning_effort": eval_config.reasoning_effort,
        "environment": eval_config.environment,
        "concurrency": eval_config.concurrency,
        "build_concurrency": eval_config.build_concurrency,
        "skill_mode": eval_config.skill_mode,
        "skills_dir": eval_config.skills_dir,
        "usage_tracking": eval_config.usage_tracking.to_config_artifact(),
        "agent_env_keys": sorted(eval_config.agent_env),
        "include_tasks": sorted(eval_config.include_tasks),
        "exclude_tasks": sorted(eval_config.exclude_tasks),
        "source_provenance": eval_config.source_provenance,
        "dataset_name": eval_config.dataset_name,
        "dataset_version": eval_config.dataset_version,
    }


def postprocess_eval_artifacts(
    plan: EvalPlan,
    resolved_tasks_dir: Path,
    eval_config: EvaluationConfig,
    job_dir: Path | None,
) -> None:
    req = plan.request
    if job_dir is None:
        return
    from benchflow.eval_artifacts import (
        TaskManifestOptions,
        materialize_canonical_job,
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
                f"selected task count {manifest['total']} != expected {req.expected_tasks}"
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
            print_error(str(exc))
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
                print_error(str(exc))
                raise typer.Exit(1) from None
            console.print(
                f"[green]Canonical jobs:[/green] {escape(str(req.canonical_jobs_dir))}"
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
            print_error(str(exc))
            raise typer.Exit(1) from None
        console.print(f"[green]Published HF artifacts:[/green] {escape(published.url)}")


def _load_eval_matrix(path: Path) -> dict[str, dict]:
    import yaml

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ValueError("--matrix must be a YAML mapping with a models: mapping")
    models = {}
    for alias, raw in data["models"].items():
        if isinstance(raw, str):
            models[str(alias)] = {"model": raw}
        elif isinstance(raw, dict) and raw.get("model"):
            models[str(alias)] = dict(raw)
        else:
            raise ValueError(
                f"matrix model {alias!r} must be a string or model mapping"
            )
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
            f"selected task count {manifest['total']} != expected {req.expected_tasks}"
        )
        raise typer.Exit(1)
    console.print(f"[green]Task manifest:[/green] {escape(str(req.task_manifest_out))}")


def run_matrix_eval(
    plan: EvalPlan,
    resolved_tasks_dir: Path,
    run_batch_eval: Callable[[EvalPlan, Path, EvaluationConfig], EvaluationResult],
) -> None:
    req = plan.request
    assert req.matrix is not None
    try:
        models = _load_eval_matrix(req.matrix)
    except (OSError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(1) from None
    root = Path(plan.output_jobs_dir)
    _write_matrix_task_manifest(plan, resolved_tasks_dir, plan.make_eval_config())
    matrix_runs = []
    for alias, raw in models.items():
        for trial in range(1, req.trials + 1):
            trial_dir = root / alias / f"trial-{trial:02d}"
            trial_prefix = (
                f"{req.hf_prefix.rstrip('/')}/{alias}/trial-{trial:02d}"
                if req.hf_prefix
                else None
            )
            trial_request = replace(
                req,
                hf_prefix=trial_prefix,
                task_manifest_out=None,
                run_config_out=_matrix_file_path(req.run_config_out, alias, trial),
                health_summary_out=_matrix_file_path(
                    req.health_summary_out, alias, trial
                ),
                canonical_selection_out=_matrix_file_path(
                    req.canonical_selection_out, alias, trial
                ),
                canonical_jobs_dir=_matrix_dir_path(
                    req.canonical_jobs_dir, alias, trial
                ),
                matrix=None,
            )
            trial_plan = replace(
                plan,
                request=trial_request,
                output_jobs_dir=str(trial_dir),
                parsed_env={
                    **plan.parsed_env,
                    **_matrix_agent_env(raw.get("agent_env")),
                },
            )
            eval_config = trial_plan.make_eval_config()
            eval_config.model = str(raw["model"])
            if raw.get("agent"):
                eval_config.agent = str(raw["agent"])
            result = run_batch_eval(trial_plan, resolved_tasks_dir, eval_config)
            matrix_runs.append(
                {
                    "alias": alias,
                    "trial": trial,
                    "jobs_dir": str(trial_dir),
                    "passed": result.passed,
                    "failed": result.failed,
                    "errored": result.errored,
                    "verifier_errored": result.verifier_errored,
                    "total": result.total,
                }
            )
    summary_path = root / "matrix-summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": 1,
            "matrix": str(req.matrix),
            "trials": req.trials,
            "runs": matrix_runs,
        },
    )
    console.print(f"[green]Matrix summary:[/green] {escape(str(summary_path))}")
