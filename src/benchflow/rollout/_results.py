"""Result, config, reward-log and trainer-artifact writers for the rollout.

This module owns the *output* side of a rollout: building the
:class:`~benchflow.models.RolloutResult` and writing ``result.json``,
``config.json`` (with secret-bearing env entries redacted), ``rewards.jsonl``,
``timing.json``, ``prompts.json`` and the trainer-format artifacts. It also
carries the small user-prompt composition helpers used by the user-driven loop.

Split out of ``rollout.py`` for cohesion; every name is re-exported from
:mod:`benchflow.rollout` so existing imports (``_build_rollout_result``,
``_write_config``, ``_write_rewards_jsonl`` from ``sdk.py`` and the test suite)
keep resolving unchanged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from benchflow._types import Role, Scene
from benchflow._utils import timing_spans
from benchflow._utils.config_redaction import (
    _SECRET_ENV_SUBSTRINGS as _SECRET_ENV_SUBSTRINGS,
)
from benchflow._utils.config_redaction import (
    _SECRET_URL_PATH_MARKERS as _SECRET_URL_PATH_MARKERS,
)
from benchflow._utils.config_redaction import (
    _is_secret_env_key as _is_secret_env_key,
)
from benchflow._utils.config_redaction import (
    _is_secret_env_value as _is_secret_env_value,
)
from benchflow._utils.config_redaction import (
    _should_record_env_entry as _should_record_env_entry,
)
from benchflow._utils.result_metadata import (
    final_metrics_from_agent_result,
    incomplete_usage_fields,
    trajectory_summary_from_events,
)
from benchflow._utils.reward_events import build_rewards_jsonl_events
from benchflow._utils.scoring import (
    API_ERROR,
    PROVIDER_STALL,
    classify_error,
    classify_verifier_error,
)
from benchflow._utils.source_provenance import artifact_source_provenance
from benchflow.contracts import (
    BaseUser,
    DocumentNudgeUser,
    ModelDocumentNudgeUser,
)
from benchflow.diagnostics import RolloutDiagnostics
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.loop_strategies import LoopStrategySpec, loop_block
from benchflow.models import RolloutResult, TrajectorySource
from benchflow.review.outcome import ScoringResult
from benchflow.review.persistence import write_json_atomic
from benchflow.rewards.validation import reward_lenient_from_env
from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    TaskSkillPolicy,
    resolve_task_skill_policy,
)
from benchflow.trajectories._capture import TrajectoryWriter
from benchflow.trajectories.metrics import count_skill_invocations
from benchflow.trajectories.types import stamp_observation_ids
from benchflow.usage_tracking import UsageTrackingConfig, normalize_usage_source

logger = logging.getLogger(__name__)

_DIAG_TRUNCATE = 2000


def _write_rewards_jsonl(
    rollout_dir: Path,
    rewards: dict | None,
    finished_at: datetime,
) -> None:
    """Write rewards.jsonl — one JSON line per reward event.

    Architecture (``docs/architecture.md``, "Evaluation — the five spaces"):
    every reward record is tagged ``(space, granularity, value)``. The shared
    :func:`build_rewards_jsonl_events` helper promotes those tags from any
    verifier-supplied per-item dict to first-class fields, falling back to the
    ``RewardEvent`` defaults (``space="output"``; ``granularity="step"`` for
    rubric items, ``"terminal"`` for the scalar reward) — the hosted-env writer
    uses the same helper so the two paths can't drift.
    """
    events = build_rewards_jsonl_events(rewards, finished_at)
    if events:
        path = rollout_dir / "rewards.jsonl"
        path.write_text("\n".join(json.dumps(e, default=str) for e in events) + "\n")


def _environment_manifest_metadata(
    manifest: EnvironmentManifest | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "name": manifest.name,
        "image": manifest.image,
        "base_image": manifest.base_image,
        "owns_lifecycle": manifest.owns_lifecycle,
        "isolation": manifest.isolation,
        "services": [service.name for service in manifest.services],
    }


def _write_config(
    rollout_dir: Path,
    *,
    task_path: Path,
    agent: str,
    model: str | None,
    environment: str,
    skill_policy: TaskSkillPolicy,
    sandbox_user: str | None,
    context_root: str | Path | None,
    sandbox_locked_paths: list[str] | None = None,
    sandbox_setup_timeout: int = 120,
    sandbox_setup_timeout_given: bool | None = None,
    sandbox_user_setup_timeout: int | None = None,
    skip_agent_install: bool = False,
    timeout: int,
    started_at: datetime,
    agent_env: dict[str, str],
    base_image_override: str | None = None,
    verifier_sandbox_mode: str = "shared",
    verifier_include: Sequence[str] = (),
    reasoning_effort: str | None = None,
    codex_apps_policy: str | None = None,
    usage_tracking: UsageTrackingConfig | None = None,
    concurrency: int | None = None,
    agent_idle_timeout: int | None = None,
    scenes: list[Scene] | None = None,
    source_provenance: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    task_digest: str | None = None,
    environment_manifest: EnvironmentManifest | None = None,
    config_override: dict | None = None,
    loop_strategy: LoopStrategySpec | None = None,
    review: dict | None = None,
    verifier_recovery: dict | None = None,
    purpose: Literal["task", "reviewer"] = "task",
    parent_rollout: str | None = None,
    freeze_workspace: bool = False,
    integrity: str = "off",
    verifier_sandbox: str = "task",
    harness: str = "acp",
    litellm_proxy: dict[str, Any] | None = None,
) -> None:
    """Write config.json to rollout_dir with secrets filtered out.

    ``harness_mode`` records how the agent ran (``acp`` or ``native``). It is
    not called ``harness``: published metadata and ``bf.compare`` already use
    that name for the agent itself.
    """
    from benchflow.acp.selection import selected_acp_transport
    from benchflow.agents.install import effective_install_timeout

    artifact_source = artifact_source_provenance(source_provenance)
    recorded_task_path = (
        str(artifact_source.get("path"))
        if artifact_source and artifact_source.get("path")
        else task_path.name
    )
    recorded_env = {
        k: v for k, v in agent_env.items() if _should_record_env_entry(k, v)
    }
    from benchflow import __version__

    config_data = {
        # Which BenchFlow ran this trial: a job resumed across versions holds
        # trials of both (DF-p4-upgrader-3).
        "benchflow_version": __version__,
        "task_path": recorded_task_path,
        "agent": agent,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "codex_apps_policy": codex_apps_policy,
        "environment": environment,
        "acp_transport": selected_acp_transport(
            agent=agent,
            environment=environment,
        ),
        "harness_mode": harness,
        "environment_manifest": _environment_manifest_metadata(environment_manifest),
        **skill_policy.config_metadata(),
        "sandbox_user": sandbox_user,
        "sandbox_locked_paths": sandbox_locked_paths,
        "sandbox_setup_timeout": sandbox_setup_timeout,
        # Whether the run named it, and what handing the workspace to the
        # sandbox user had; both absent before.
        **(
            {"sandbox_setup_timeout_given": sandbox_setup_timeout_given}
            if sandbox_setup_timeout_given is not None
            else {}
        ),
        **(
            {"sandbox_user_setup_timeout": sandbox_user_setup_timeout}
            if sandbox_user_setup_timeout is not None
            else {}
        ),
        "skip_install": skip_agent_install,
        "agent_install_timeout": None
        if skip_agent_install
        else effective_install_timeout(agent, sandbox_setup_timeout),
        "context_root": str(context_root) if context_root else None,
        "base_image_override": base_image_override,
        "verifier_sandbox_mode": verifier_sandbox_mode,
        "verifier_include": list(verifier_include),
        "timeout_sec": timeout,
        "concurrency": concurrency,
        "agent_idle_timeout_sec": agent_idle_timeout,
        "started_at": str(started_at),
        "agent_env": recorded_env,
        "scenes": _scene_metadata(scenes or []),
        "loop": loop_block(loop_strategy),
        # Whether reward files are read leniently (BENCHFLOW_REWARD_LENIENT,
        # P6): it changes which reward maps score, so the trial records it.
        "reward_lenient": reward_lenient_from_env(),
    }
    if usage_tracking is not None:
        config_data["usage_tracking"] = usage_tracking.to_config_artifact()
    if artifact_source is not None:
        config_data["source"] = artifact_source
    if dataset is not None:
        config_data["dataset_name"] = dataset.get("name")
        config_data["dataset_version"] = dataset.get("version")
    if task_digest is not None:
        config_data["task_digest"] = task_digest
    # C-axis provenance: persist the bound config overlay + its content hash so
    # the run records exactly which configuration it ran with (symmetric to
    # environment_manifest above).
    if config_override:
        from benchflow._utils.config_override import overlay_hash

        config_data["config_override"] = {
            "keys": sorted(config_override),
            "sha256": overlay_hash(config_override),
            "patch": config_override,
        }
    if verifier_recovery is not None:
        config_data["verifier_recovery"] = verifier_recovery
    if review is not None:
        config_data["review"] = review
    if freeze_workspace:
        config_data["freeze_workspace"] = True
    if integrity != "off":
        config_data["integrity"] = integrity
    # The run's --verifier-sandbox setting (task or separate); result.json's
    # verifier_isolation says where the verifier actually ran, and why.
    config_data["verifier_sandbox"] = verifier_sandbox
    if harness == "native":
        from benchflow.native_harness.harnesses import native_harness_for

        native = native_harness_for(agent)
        config_data["native_harness"] = {"cli": native.cli, "package": native.package}
    if purpose != "task":
        config_data.update(purpose=purpose, parent_rollout=parent_rollout)
    if litellm_proxy is not None:
        # Where the model proxy ran, the locked requirement file it installed
        # (by name and sha256) and the extras it added on top of it.
        config_data["litellm_proxy"] = litellm_proxy
    write_json_atomic(rollout_dir / "config.json", config_data)


def _record_reported_models(
    rollout_dir: Path | None, models: list[str], source: str
) -> None:
    """Add the models that answered to the trial's config.json (P6, G-28).

    ``model`` stays what the run asked for (an alias such as ``sonnet`` on a
    subscription, or a gateway name); ``reported_models`` lists, once each in
    the order first seen, the models the answers said they came from, and
    ``reported_models_source`` says whose word that is: ``provider_response``
    (the model field of the provider's responses, as BenchFlow's model proxy
    recorded them) or ``agent_session_log`` (the agent's own session file, an
    agent-written report). Nothing is written when no answer named a model,
    so a trial never shows a made-up one. Never fails the trial.
    """
    if rollout_dir is None or not models:
        return
    path = rollout_dir / "config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(config, dict):
        return
    config["reported_models"] = list(models)
    config["reported_models_source"] = source
    try:
        write_json_atomic(path, config)
    except OSError as e:
        logger.warning(f"Cannot record reported models in {path}: {e}")


def _role_metadata(role: Role) -> dict[str, Any]:
    return {
        "name": role.name,
        "agent": role.agent,
        "model": role.model,
        "reasoning_effort": role.reasoning_effort,
        "timeout_sec": role.timeout_sec,
        "idle_timeout_sec": role.idle_timeout_sec,
        "skills_dir": str(role.skills_dir) if role.skills_dir else None,
        "capabilities": role.capabilities,
        "env_keys": sorted(role.env),
    }


def _scene_metadata(scenes: list[Scene]) -> list[dict[str, Any]]:
    return [
        {
            "name": scene.name,
            "skills_dir": str(scene.skills_dir) if scene.skills_dir else None,
            "roles": [_role_metadata(role) for role in scene.roles],
            "turns": [
                {"role": turn.role, "has_prompt": turn.prompt is not None}
                for turn in scene.turns
            ],
        }
        for scene in scenes
    ]


def _build_rollout_result(
    rollout_dir: Path,
    *,
    task_name: str,
    rollout_name: str,
    agent: str,
    agent_name: str,
    model: str | None,
    n_tool_calls: int,
    prompts: list[str],
    error: str | None,
    verifier_error: str | None,
    trajectory: list[dict],
    partial_trajectory: bool,
    export_error: str | None = None,
    trajectory_source: TrajectorySource | None = None,
    rewards: dict | None,
    started_at: datetime,
    timing: dict[str, Any],
    scenes: list[Scene] | None = None,
    n_input_tokens: int | None = None,
    n_output_tokens: int | None = None,
    n_cache_read_tokens: int | None = None,
    n_cache_creation_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    usage_source: str = "unavailable",
    price_source: str | None = None,
    usage_source_legacy: str | None = None,
    cost_source: str | None = None,
    price_table_date: str | None = None,
    n_cache_write_5m_tokens: int | None = None,
    n_cache_write_1h_tokens: int | None = None,
    cost_usd_provider: float | None = None,
    n_calls_cost_estimated: int | None = None,
    usage_incomplete: bool | None = None,
    usage_incomplete_reason: str | None = None,
    usage_details: dict[str, Any] | None = None,
    usage_tracking: dict[str, Any] | None = None,
    evolved_skills: dict[str, str] | None = None,
    source_provenance: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    task_digest: str | None = None,
    diagnostics: RolloutDiagnostics | None = None,
    skill_policy: TaskSkillPolicy | None = None,
    sandbox_id: str | None = None,
    loop: dict[str, Any] | None = None,
    scoring: ScoringResult | None = None,
    purpose: Literal["task", "reviewer"] = "task",
    parent_rollout: str | None = None,
    result_filename: str | None = "result.json",
    branches: dict[str, Any] | None = None,
    verifier_isolation: tuple[str, str] | None = None,
    cloud_metadata: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    effort: dict[str, Any] | None = None,
    test_sh_preflight: dict[str, Any] | None = None,
) -> RolloutResult:
    """Build RolloutResult and write result.json, timing.json, prompts.json, trajectory.

    Diagnostics flow through the :class:`RolloutDiagnostics` collector
    (issue #503). Callers that previously passed per-field ``*_info``
    dicts should construct a collector and ``set()`` typed diagnostics.
    ``result_filename=None`` builds the result without writing any result
    file: a reviewed trial publishes nothing until its scoring commits.
    """
    if diagnostics is None:
        diagnostics = RolloutDiagnostics()
    if skill_policy is None:
        skill_policy = resolve_task_skill_policy(
            task_path=Path(task_name),
            skill_mode=SKILL_MODE_NO_SKILL,
            runtime_skills_dir=None,
            declared_sandbox_skills_dir=None,
        )
    finished_at = datetime.now()
    n_skill_invocations = count_skill_invocations(trajectory)
    error_category = (
        diagnostics.category_for_channel("error") if error is not None else None
    ) or classify_error(error)
    if classify_error(error) == "invalid_provider_response":
        error_category = "invalid_provider_response"
    if error is not None and error.startswith(PROVIDER_STALL):
        # Over the timeout diagnostic it ended with: the provider's error.
        error_category = API_ERROR
    verifier_error_category = (
        diagnostics.category_for_channel("verifier_error")
        if verifier_error is not None
        else None
    ) or classify_verifier_error(verifier_error)
    # Before the result and the final write share the list: each tool call's
    # observation identity is fixed once, before any redaction.
    stamp_observation_ids(trajectory)
    result = RolloutResult(
        task_name=task_name,
        rollout_name=rollout_name,
        rewards=rewards,
        scoring=scoring,
        purpose=purpose,
        parent_rollout=parent_rollout,
        trajectory=trajectory,
        agent=agent,
        agent_name=agent_name,
        model=model,
        n_tool_calls=n_tool_calls,
        n_skill_invocations=n_skill_invocations,
        n_prompts=len(prompts),
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
        n_cache_read_tokens=n_cache_read_tokens,
        n_cache_creation_tokens=n_cache_creation_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        usage_source=normalize_usage_source(usage_source),
        price_source=price_source,
        usage_source_legacy=usage_source_legacy,
        cost_source=cost_source,
        price_table_date=price_table_date,
        n_cache_write_5m_tokens=n_cache_write_5m_tokens,
        n_cache_write_1h_tokens=n_cache_write_1h_tokens,
        usage_details=usage_details,
        error=error,
        error_category=error_category,
        verifier_error=verifier_error,
        verifier_error_category=verifier_error_category,
        export_error=export_error,
        partial_trajectory=partial_trajectory,
        trajectory_source=trajectory_source,
        evolved_skills=evolved_skills,
        usage_incomplete=usage_incomplete,
        usage_incomplete_reason=usage_incomplete_reason,
        source_provenance=source_provenance,
        started_at=started_at,
        finished_at=finished_at,
        rollout_dir=rollout_dir,
    )
    timing["total"] = (finished_at - started_at).total_seconds()
    timing = timing_spans.rounded(timing)
    from benchflow.agents.registry import is_scripted_agent

    no_usage_reason = "no_model_call" if is_scripted_agent(agent) else None
    if no_usage_reason is not None:
        # No model was called: every count and the cost are null, never 0.
        for field in (
            "n_input_tokens",
            "n_output_tokens",
            "n_cache_read_tokens",
            "n_cache_creation_tokens",
            "n_cache_write_5m_tokens",
            "n_cache_write_1h_tokens",
            "total_tokens",
            "cost_usd",
            "cost_source",
            "price_table_date",
        ):
            setattr(result, field, None)
    agent_result = {
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": result.n_skill_invocations,
        "n_prompts": result.n_prompts,
        "n_input_tokens": result.n_input_tokens,
        "n_output_tokens": result.n_output_tokens,
        "n_cache_read_tokens": result.n_cache_read_tokens,
        "n_cache_creation_tokens": result.n_cache_creation_tokens,
        # its token split: n_input_tokens is uncached; the cache writes
        # (n_cache_creation_tokens, kept as an alias) with their 5-minute and
        # 1-hour shares where the provider reports them; output with reasoning.
        "n_cache_write_tokens": result.n_cache_write_tokens,
        "n_cache_write_5m_tokens": result.n_cache_write_5m_tokens,
        "n_cache_write_1h_tokens": result.n_cache_write_1h_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "usage_source": result.usage_source,
        "cost_source": result.cost_source,
        "price_table_date": result.price_table_date,
        "price_source": result.price_source,
        # The earlier name, for readers that tell a metered run from a
        # login's (trajectories/results.py, job_export.py's cost_status).
        "usage_source_legacy": result.usage_source_legacy,
        # Why a trial has no usage when it never could: "no_model_call" for a
        # control or an operator script (oracle, nop, script:...), whose
        # null tokens and cost are no missing measurement.
        "no_usage_reason": no_usage_reason,
    }
    if cost_usd_provider is not None:
        # Some calls carried their provider's bill and some did not: the
        # billed part and how many calls were estimated.
        agent_result["cost_usd_provider"] = cost_usd_provider
        agent_result["n_calls_cost_estimated"] = n_calls_cost_estimated
    # A count that misses calls says so, and so does its cost.
    agent_result.update(
        incomplete_usage_fields(
            {
                "usage_incomplete": usage_incomplete,
                "usage_incomplete_reason": usage_incomplete_reason,
                "cost_usd": agent_result["cost_usd"],
                # An unpriced warm-up, or unreported writes on a route
                # that prices them, makes the cost a floor.
                "usage_details": result.usage_details,
            }
        )
    )
    if result.usage_details is not None:
        agent_result["usage_details"] = result.usage_details
    final_metrics = final_metrics_from_agent_result(agent_result)
    trajectory_summary = trajectory_summary_from_events(
        trajectory,
        partial_trajectory=partial_trajectory,
        trajectory_source=trajectory_source,
    )
    traj_dir = rollout_dir / "trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    # Final write — overwrites whatever the live streaming writer left
    # in place. Identical content in the normal ACP path, but this is
    # the only writer for oracle / scraped-fallback / no-session paths.
    # Redaction is applied inside TrajectoryWriter so every write path
    # (streaming + final) is scrubbed (#537/#585).
    TrajectoryWriter(traj_dir / "acp_trajectory.jsonl").write_final(trajectory)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    result_data = {
        "task_name": result.task_name,
        "rollout_name": result.rollout_name,
        "rewards": result.rewards,
        "agent": result.agent,
        "agent_name": result.agent_name,
        "model": result.model,
        **skill_policy.config_metadata(),
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": result.n_skill_invocations,
        "n_prompts": result.n_prompts,
        "agent_result": agent_result,
        "final_metrics": final_metrics,
        "trajectory_summary": trajectory_summary,
        "usage_tracking": usage_tracking,
        "error": result.error,
        "error_category": result.error_category,
        "verifier_error": result.verifier_error,
        "verifier_error_category": result.verifier_error_category,
        "export_error": result.export_error,
        **diagnostics.to_result_fields(),
        "partial_trajectory": result.partial_trajectory,
        "trajectory_source": result.trajectory_source,
        "started_at": str(result.started_at),
        "finished_at": str(result.finished_at),
        "timing": timing,
        "scenes": _scene_metadata(scenes or []),
        "loop": loop or loop_block(None),
        **(
            {"source": artifact_source_provenance(source_provenance)}
            if source_provenance is not None
            else {}
        ),
        **(
            {
                "dataset_name": dataset.get("name"),
                "dataset_version": dataset.get("version"),
            }
            if dataset is not None
            else {}
        ),
        **({"task_digest": task_digest} if task_digest is not None else {}),
        "sandbox_id": sandbox_id,
        # Where the verifier ran (shared or separate) and what decided it
        # (task, run-flag or integrity-strict); None before setup resolved it.
        "verifier_isolation": verifier_isolation[0] if verifier_isolation else None,
        "verifier_isolation_source": (
            verifier_isolation[1] if verifier_isolation else None
        ),
        # What test.sh needs, resolved before the agent's turn as the
        # verifier runs it (task/test_sh_preflight.py); None when unchecked.
        "test_sh_preflight": test_sh_preflight,
    }
    if scoring is not None:
        result_data["scoring"] = scoring.to_dict()
    if branches is not None:
        result_data["branches"] = branches
    if cloud_metadata is not None:
        # The cloud metadata block's state, and the run's opt-out when it
        # made one; cloud-metadata.json has the whole record.
        result_data["cloud_metadata"] = cloud_metadata
    if effort is not None:
        # The effort asked for beside the one that ran; a mismatch
        # is a trial warning too.
        from benchflow.effort_record import effort_warning

        result_data["effort"] = effort
        if (note := effort_warning(effort)) is not None:
            warnings = [*(warnings or []), note]
    if warnings:
        # What the trial ran without or around, by name, never a value
        # (benchflow._utils.trial_warnings); absent when there is none.
        result_data["warnings"] = list(warnings)
    if purpose != "task":
        result_data.update(purpose=purpose, parent_rollout=parent_rollout)
    if result_filename is None:
        return result
    if result_filename != "result.json":
        write_json_atomic(rollout_dir / result_filename, result_data)
        write_json_atomic(rollout_dir / "prompts.json", prompts)
        return result
    (rollout_dir / "timing.json").write_text(json.dumps(timing, indent=2))
    (rollout_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    _write_rewards_jsonl(rollout_dir, rewards, finished_at)
    _write_trainer_artifact(
        rollout_dir,
        task_name=task_name,
        rollout_name=rollout_name,
        agent_name=agent_name or agent,
        prompts=prompts,
        effort=effort,
        trajectory=trajectory,
        rewards=rewards,
        model=model,
        error=error,
        verifier_error=verifier_error,
        partial_trajectory=partial_trajectory,
        scoring=scoring,
        total_prompt_tokens=final_metrics["total_prompt_tokens"],
        total_completion_tokens=final_metrics["total_completion_tokens"],
        total_cached_tokens=final_metrics["total_cached_tokens"],
        total_cost_usd=final_metrics["total_cost_usd"],
        metrics_extra=final_metrics["extra"],
    )
    _write_results_jsonl(
        rollout_dir,
        task_name=task_name,
        rollout_name=rollout_name,
        agent=agent,
        agent_name=agent_name,
        model=model,
        n_tool_calls=n_tool_calls,
        prompts=prompts,
        trajectory=trajectory,
        partial_trajectory=partial_trajectory,
        rewards=rewards,
        error=error,
        verifier_error=verifier_error,
        error_category=result.error_category,
        verifier_error_category=result.verifier_error_category,
        diagnostics=diagnostics,
        export_error=export_error,
        timing=timing,
        agent_result=agent_result,
        scoring=scoring,
        purpose=purpose,
        parent_rollout=parent_rollout,
        branches=branches,
    )
    write_json_atomic(rollout_dir / "result.json", result_data)
    return result


def _write_results_jsonl(
    rollout_dir: Path,
    **kwargs: Any,
) -> None:
    """Emit the Verifiers/Prime-RL-shaped rollout result row."""
    from benchflow.trajectories.results import write_rollout_results_jsonl

    try:
        write_rollout_results_jsonl(rollout_dir, **kwargs)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("results.jsonl write failed: %s", e)


def rewrite_exports_as_read(trial_dir: Path) -> bool:
    """Rewrite one trial's export records as every reader scores the trial: ``results.jsonl``, ``trainer/verifiers.jsonl`` and the
    other trainer formats, and ``rewards.jsonl``.

    The rollout writes them when the trial ends, from the reward the verifier
    gave. A trial the judge (:func:`benchflow.integration_health.scored_as_read`)
    withholds on read, a final idle timeout whose agent did nothing, say, is
    an unscored integration failure to every reader, so its records are
    rewritten unscored, as for a trial the run withheld itself: the same
    writers, the judged error and category, no reward. result.json is left as
    the run wrote it. Returns whether the trial's records were rewritten.
    """
    from collections.abc import Mapping

    from benchflow._utils.result_metadata import (
        atif_metrics_extra,
        atif_prompt_tokens,
    )
    from benchflow.integration_health import scored_as_read
    from benchflow.review.outcome import scoring_from_result

    try:
        result = json.loads((trial_dir / "result.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(result, dict) or isinstance(
        result.get("integration_failure_info"), Mapping
    ):
        return False  # unreadable, or withheld by the run, which wrote them so
    judged = scored_as_read(result, trial_dir)
    if not isinstance(judged.get("integration_failure_info"), Mapping):
        return False
    try:
        trajectory = [
            json.loads(line)
            for line in (trial_dir / "trajectory" / "acp_trajectory.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
    except (OSError, ValueError):
        trajectory = []
    try:
        prompts = json.loads((trial_dir / "prompts.json").read_text())
    except (OSError, ValueError):
        prompts = []
    agent_result = result.get("agent_result") or {}
    try:
        scoring = scoring_from_result(result)
    except ValueError:
        scoring = None
    (trial_dir / "rewards.jsonl").unlink(missing_ok=True)
    if (trial_dir / "trainer").is_dir():
        _write_trainer_artifact(
            trial_dir,
            task_name=result.get("task_name") or "",
            rollout_name=result.get("rollout_name") or trial_dir.name,
            agent_name=result.get("agent_name") or result.get("agent") or "",
            prompts=prompts,
            trajectory=trajectory,
            rewards=None,
            model=result.get("model"),
            verifier_error=result.get("verifier_error"),
            error=judged.get("error"),
            partial_trajectory=bool(result.get("partial_trajectory", False)),
            scoring=scoring,
            total_prompt_tokens=atif_prompt_tokens(agent_result),
            total_completion_tokens=agent_result.get("n_output_tokens"),
            total_cached_tokens=agent_result.get("n_cache_read_tokens"),
            total_cost_usd=agent_result.get("cost_usd"),
            metrics_extra=atif_metrics_extra(agent_result),
        )
    if (trial_dir / "results.jsonl").is_file():
        _write_results_jsonl(
            trial_dir,
            task_name=result.get("task_name") or "",
            rollout_name=result.get("rollout_name") or trial_dir.name,
            agent=result.get("agent") or "",
            agent_name=result.get("agent_name") or "",
            model=result.get("model"),
            n_tool_calls=result.get("n_tool_calls") or 0,
            prompts=prompts,
            trajectory=trajectory,
            partial_trajectory=bool(result.get("partial_trajectory", False)),
            rewards=None,
            error=judged.get("error"),
            verifier_error=result.get("verifier_error"),
            error_category=judged.get("error_category"),
            verifier_error_category=result.get("verifier_error_category"),
            diagnostics=RolloutDiagnostics.from_result_fields(judged),
            export_error=result.get("export_error"),
            timing=result.get("timing") or {},
            agent_result=agent_result,
            scoring=scoring,
            purpose=result.get("purpose", "task"),
            parent_rollout=result.get("parent_rollout"),
            branches=result.get("branches"),
        )
    return True


def rewrite_job_exports_as_read(job_dir: Path) -> int:
    """:func:`rewrite_exports_as_read` for every trial of the job, before its
    job-level ``results.jsonl``, ``verifiers.jsonl`` and ``adp.jsonl`` are
    built from them. Returns how many trials' records were rewritten."""
    from benchflow._utils.result_paths import iter_task_result_paths

    rewritten = 0
    for path in iter_task_result_paths(job_dir):
        try:
            rewritten += rewrite_exports_as_read(path.parent)
        except Exception:  # an export never fails the job or the rescore
            logger.warning("Rewriting %s's exports failed", path.parent, exc_info=True)
    return rewritten


def _write_trainer_artifact(
    rollout_dir: Path,
    *,
    task_name: str,
    rollout_name: str,
    agent_name: str,
    prompts: list[str],
    effort: dict[str, Any] | None = None,
    trajectory: list[dict],
    rewards: dict | None,
    model: str | None,
    verifier_error: str | None,
    error: str | None = None,
    partial_trajectory: bool = False,
    scoring: ScoringResult | None = None,
    total_prompt_tokens: int | None = None,
    total_completion_tokens: int | None = None,
    total_cached_tokens: int | None = None,
    total_cost_usd: float | None = None,
    metrics_extra: dict[str, Any] | None = None,
    strict: bool = False,
) -> None:
    """Emit the trainer-format artifacts for this scored rollout.

    The architecture's train-mode seam (issue #385): every scored rollout
    that reaches result-building produces a trainer-ready Verifiers / ORS
    record (``trainer/verifiers.jsonl``) plus the ecosystem trajectory
    formats — ATIF (``trainer/atif.json``; omitted for empty trajectories,
    which the schema forbids) and ADP (``trainer/adp.jsonl``). Each format
    is written independently; failures are logged by default. Terminal scoring
    commits use ``strict=True`` so a failed export cannot publish a healthy
    final result before its required artifacts exist.
    """
    from benchflow.trajectories.export import write_rollout_verifiers_jsonl
    from benchflow.trajectories.export_adp import write_rollout_adp_jsonl
    from benchflow.trajectories.export_atif import write_rollout_atif_json

    # Real rollout names already embed the task (``<task>__<hash>``); only
    # prefix when they don't, so ids stay unique without doubling the task.
    if rollout_name.startswith(f"{task_name}__"):
        trajectory_id = rollout_name
    else:
        trajectory_id = f"{task_name}__{rollout_name}"
    # The same completion and verdict results.jsonl writes: an errored,
    # truncated or unscored trial is not completed, and a failed scoring has
    # no reward (audit RECORDS-4).
    unscored = scoring is not None and scoring.status == "error"
    try:
        write_rollout_verifiers_jsonl(
            rollout_dir,
            task_id=task_name,
            prompts=prompts,
            trajectory=trajectory,
            rewards=None if unscored else rewards,
            model=model,
            environment=task_name,
            is_completed=not (error or verifier_error or partial_trajectory),
            is_truncated=bool(partial_trajectory),
            error=verifier_error or (scoring.error if unscored and scoring else None),
        )
    except Exception as e:  # pragma: no cover - defensive
        if strict:
            raise
        logger.warning("Trainer artifact write failed: %s", e)
    try:
        write_rollout_atif_json(
            rollout_dir,
            session_id=trajectory_id,
            agent_name=agent_name,
            prompts=prompts,
            trajectory=trajectory,
            model=model,
            effort=effort,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cached_tokens=total_cached_tokens,
            total_cost_usd=total_cost_usd,
            metrics_extra=metrics_extra,
        )
    except Exception as e:  # pragma: no cover - defensive
        if strict:
            raise
        logger.warning("ATIF artifact write failed: %s", e)
    try:
        write_rollout_adp_jsonl(
            rollout_dir,
            trajectory_id=trajectory_id,
            task_id=task_name,
            prompts=prompts,
            trajectory=trajectory,
            model=model,
            environment=task_name,
            reward=None if unscored else (rewards or {}).get("reward"),
        )
    except Exception as e:  # pragma: no cover - defensive
        if strict:
            raise
        logger.warning("ADP artifact write failed: %s", e)
    # Embodied episodes (benchflow.embodied): per-step state/action/reward rows and
    # the video index, next to the other trainer formats. Only trials whose verifier
    # downloaded an episode record (verifier/episode/steps.jsonl) produce them.
    try:
        from benchflow.embodied.export import write_rollout_embodied

        write_rollout_embodied(
            rollout_dir,
            trajectory_id=trajectory_id,
            task_name=task_name,
            rewards=rewards,
        )
    except Exception as e:  # an optional format never fails the rollout
        logger.warning("Embodied trainer artifact write failed: %s", e)


def _is_document_user(user: BaseUser) -> bool:
    return isinstance(user, (DocumentNudgeUser, ModelDocumentNudgeUser))


def _compose_scene_user_prompt(scene_prompt: str, user_prompt: str | None) -> str:
    """Attach an optional simulated-user nudge to a required scene prompt."""

    scene_prompt = scene_prompt.strip()
    if user_prompt is None:
        return scene_prompt
    user_prompt = user_prompt.strip()
    if not user_prompt or user_prompt == scene_prompt:
        return scene_prompt
    return f"{scene_prompt}\n\nUser follow-up:\n{user_prompt}"


def _user_confirmation_policy(user: BaseUser) -> str | None:
    value = getattr(user, "confirmation_policy", None)
    return value if isinstance(value, str) else None


def _user_handoff_kind(user: BaseUser) -> str | None:
    value = getattr(user, "handoff_kind", None)
    return value if isinstance(value, str) else None


def _least_permissive_option_id(
    options: list[str],
    option_kinds: dict[str, str] | None = None,
) -> str:
    """Select a deny/reject option for non-interactive human confirmation."""

    if not options:
        return "deny"
    option_kinds = option_kinds or {}
    reject_kinds = ("reject", "deny", "cancel", "disallow", "block")
    for option in options:
        kind = option_kinds.get(option, "").replace("-", "_").lower()
        if any(token in kind for token in reject_kinds):
            return option
    deny_tokens = ("deny", "reject", "cancel", "disallow", "block", "no")
    for option in options:
        normalized = option.replace("-", "_").lower()
        if any(token in normalized for token in deny_tokens):
            return option
    allow_tokens = ("allow", "approve", "bypass", "yes")
    for option in options:
        normalized = option.replace("-", "_").lower()
        if not any(token in normalized for token in allow_tokens):
            return option
    return options[0]
