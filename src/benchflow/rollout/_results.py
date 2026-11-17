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
from collections.abc import Callable, Sequence
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
    AGENT_INTEGRATION,
    API_ERROR,
    IDLE_TIMEOUT,
    PROVIDER_STALL,
    classify_error,
    classify_verifier_error,
)
from benchflow._utils.source_provenance import artifact_source_provenance
from benchflow.build_info import benchflow_build
from benchflow.contracts import (
    BaseUser,
    DocumentNudgeUser,
    ModelDocumentNudgeUser,
)
from benchflow.diagnostics import AgentPromptTimeoutDiagnostic, RolloutDiagnostics
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.integration_health import (
    IntegrationFailure,
    diagnose_kept_idle_timeout,
)
from benchflow.loop_strategies import LoopStrategySpec, loop_block
from benchflow.models import RolloutResult, TrajectorySource
from benchflow.review.outcome import ScoringResult
from benchflow.review.persistence import write_json_atomic
from benchflow.rewards.validation import reward_lenient_from_env
from benchflow.run_checks import (
    harness_ending,
    read_calls,
    settings_sent,
    turn_endings,
)
from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    TaskSkillPolicy,
    resolve_task_skill_policy,
)
from benchflow.trajectories._capture import TrajectoryWriter
from benchflow.trajectories.metrics import count_skill_invocations
from benchflow.trajectories.model_calls import (
    SUMMARY_FILE,
    agent_outcome,
    effort_outcome,
    effort_warning,
    refusal_ending,
    subagent_usage,
)
from benchflow.trajectories.otel_spans import emit_trial_spans
from benchflow.trajectories.types import (
    redact_trajectory_obj,
    redact_trajectory_text,
    stamp_observation_ids,
)
from benchflow.usage_tracking import (
    NO_MODEL_CALL,
    UsageTrackingConfig,
    normalize_usage_source,
    token_split,
)

logger = logging.getLogger(__name__)

_DIAG_TRUNCATE = 2000


def _write_rewards_jsonl(
    rollout_dir: Path,
    rewards: dict | None,
    finished_at: datetime,
) -> None:
    """Write rewards.jsonl — one JSON line per reward event.

    Architecture (``docs/start/how-it-works.md``, "Evaluation — the five spaces"):
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
    verifier_setup_timeout: int | None = None,
    verifier_setup_timeout_run: int | None = None,
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
    integrity: str = "audit",
    verifier_sandbox: str = "task",
    harness: str = "acp",
    litellm_proxy: dict[str, Any] | None = None,
    harness_versions: dict[str, Any] | None = None,
) -> None:
    """Write config.json to rollout_dir with secrets filtered out.

    ``harness_mode`` records how the agent ran (``acp`` or ``native``). It is
    not called ``harness``: published metadata and ``bf.compare`` already use
    that name for the agent itself. ``model_route`` is recorded only for an
    agent that routes its own model calls (``"agent"``): its
    ``model`` is then a label, and nothing metered its usage.
    """
    from benchflow.acp.selection import selected_acp_transport
    from benchflow.agents.install import effective_install_timeout
    from benchflow.agents.registry import MODEL_ROUTE_AGENT, agent_routes_own_model

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
        **({"model_route": MODEL_ROUTE_AGENT} if agent_routes_own_model(agent) else {}),
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
        # sandbox user had; absent before.
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
        # Seconds each verifier-setup command had; absent before.
        # The budget used, and the run's own value (null: the run named
        # none), which a regrade or a recovery passes on as the run's.
        **(
            {
                "verifier_setup_timeout": verifier_setup_timeout,
                "verifier_setup_timeout_run": verifier_setup_timeout_run,
            }
            if verifier_setup_timeout is not None
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
        "started_at": started_at.astimezone().isoformat(),
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
    # C-axis provenance: which configuration overlay the run bound
    # (symmetric to environment_manifest above). The values that look secret
    # are not persisted, nor hashed; a command that runs part of the trial
    # again is given them again
    # (benchflow._utils.config_override.recorded_overlay).
    if config_override:
        from benchflow._utils.config_override import overlay_record

        config_data["config_override"] = overlay_record(config_override)
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
    # The installed versions, once the agent's install has been probed
    # (_record_harness_versions_in_config adds them then).
    config_data["harness_versions"] = harness_versions
    write_json_atomic(rollout_dir / "config.json", config_data)


def _record_harness_versions_in_config(
    rollout_dir: Path, receipt: dict[str, Any]
) -> None:
    """Add the installed harness versions the rollout probed to config.json.

    From the rollout's own copy, never from ``agent/runtime.json``: the agent
    can write that folder. Never fails the trial.
    """
    path = rollout_dir / "config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(config, dict):
        return
    config["harness_versions"] = receipt
    try:
        write_json_atomic(path, config)
    except OSError as e:
        logger.warning(f"Cannot record the harness versions in {path}: {e}")


def _masked(text: str | None) -> str | None:
    """Error text as result.json keeps it: secret shapes masked."""
    return redact_trajectory_text(text) if isinstance(text, str) else text


def _record_effort_in_config(
    rollout_dir: Path, applied: str | None, route: str | None
) -> None:
    """Add the reasoning effort the agent session applied to config.json.

    ``reasoning_effort`` stays what the run asked for; ``reasoning_effort_applied``
    is the session's word for what it runs at, and ``reasoning_effort_route``
    the effort the model proxy's route sets on each request, when it sets one.
    result.json's ``effort`` sets both beside the model calls'. Never fails
    the trial.
    """
    path = rollout_dir / "config.json"
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(config, dict):
        return
    config["reasoning_effort_applied"] = applied
    if route is not None:
        config["reasoning_effort_route"] = route
    try:
        write_json_atomic(path, config)
    except OSError as e:
        logger.warning(f"Cannot record the applied effort in {path}: {e}")


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
    error_category: str | None = None,
    export_error: str | None = None,
    trajectory_source: TrajectorySource | None = None,
    stop_reason: str | None = None,
    stop_reasons: list[str | None] | None = None,
    compactions: dict[str, Any] | None = None,
    harness_versions: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    effort_applied: str | None = None,
    effort_route: str | None = None,
    effort_requested_from: str | None = None,
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
    n_cache_write_tokens: int | None = None,
    n_cache_write_5m_tokens: int | None = None,
    n_cache_write_1h_tokens: int | None = None,
    no_usage_reason: str | None = None,
    cost_usd_provider: float | None = None,
    n_calls_cost_estimated: int | None = None,
    usage_incomplete: bool | None = None,
    usage_incomplete_reason: str | None = None,
    usage_details: dict[str, Any] | None = None,
    judge_usage: dict[str, Any] | None = None,
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
    send_spans: bool = True,
    branches: dict[str, Any] | None = None,
    verifier_isolation: tuple[str, str] | None = None,
    cloud_metadata: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    trial_budget: dict[str, Any] | None = None,
    host_ptrace_scope: int | str | None = None,
    agent_user: dict[str, Any] | None = None,
    test_sh_preflight: dict[str, Any] | None = None,
) -> RolloutResult:
    """Build RolloutResult and write result.json, timing.json, prompts.json, trajectory.

    Diagnostics flow through the :class:`RolloutDiagnostics` collector
    (issue #503). Callers that previously passed per-field ``*_info``
    dicts should construct a collector and ``set()`` typed diagnostics.
    ``result_filename=None`` builds the result without writing any result
    file: a reviewed trial publishes nothing until its scoring commits.

    ``stop_reason`` is how the agent's last prompt ended (the ACP stop
    reason), None when it did not end; with the model-call totals teardown
    wrote, it decides result.json's ``agent_outcome``, which also lists
    ``stop_reasons``, every prompt turn's (benchflow.run_checks). ``harness_versions``
    is the rollout's probe of the installed harness; without one (a result
    built again in a later process), config.json's record of it.
    ``reasoning_effort`` is the effort the run asked for, ``effort_applied``
    the agent session's word for the one it runs at and ``effort_route`` one
    the model proxy's route sets: result.json's ``effort`` puts them beside
    the efforts the model calls carried, and a difference is a warning.
    ``trial_budget`` is the trial's budget and what stopped it, when it had
    one (the rollout's ``_trial_budget_block``): result.json's
    ``trial_budget`` adds what the trial used, by the model calls the proxy
    recorded, and what its subagents used, by the agent's own session log.
    ``host_ptrace_scope`` is the Yama ``kernel.yama.ptrace_scope`` the
    sandbox read when it started (0 to 3, ``"absent"`` on a kernel without
    Yama), None when it was not read; every result.json holds it.
    ``agent_user`` is the user the task's own harness runs the agent as and
    the one BenchFlow ran it as, with the ``--sandbox-user`` that matches
    (sandbox/agent_user.py); every result.json holds it.
    ``error_category`` is given when the code that recorded the error also
    decided its category (the host hard deadline's ``infra_failure``); it
    wins over the diagnostics, which may still hold an earlier agent timeout.
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
    # A script agent stopped at its time limit has no error text (its
    # workspace is graded as it stands), and is still a timeout.
    timed_out = diagnostics.get(AgentPromptTimeoutDiagnostic.field) is not None
    if error_category is None:
        error_category = (
            diagnostics.category_for_channel("error")
            if error is not None or timed_out
            else None
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
    if verifier_error_category is None:
        from benchflow._utils.scoring import (
            PROVENANCE_INCOMPLETE,
            provenance_incomplete,
        )

        if provenance_incomplete(
            {"verifier_error": verifier_error, "warnings": warnings}
        ):
            verifier_error_category = PROVENANCE_INCOMPLETE
    # Before the result and the final write share the list: each tool call's
    # observation identity is fixed once, before any redaction.
    stamp_observation_ids(trajectory)
    if harness_versions is None:
        harness_versions = _recorded_harness_versions(rollout_dir)
    # One token split: cache writes under their own name (the
    # old name stays as an alias), and a total only of known parts.
    if n_cache_write_tokens is None:
        n_cache_write_tokens = n_cache_creation_tokens
    n_cache_creation_tokens = n_cache_write_tokens
    from benchflow.agents.registry import is_scripted_agent

    if no_usage_reason is None and is_scripted_agent(agent):
        # An oracle, a nop or a script made no model call: nothing to count.
        no_usage_reason = NO_MODEL_CALL
    if no_usage_reason == NO_MODEL_CALL:
        n_input_tokens = n_output_tokens = n_cache_read_tokens = None
        n_cache_write_tokens = n_cache_creation_tokens = None
        n_cache_write_5m_tokens = n_cache_write_1h_tokens = None
        cost_usd = None
        usage_incomplete = usage_incomplete_reason = None
    total_tokens = token_split(
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
        n_cache_read_tokens=n_cache_read_tokens,
        n_cache_write_tokens=n_cache_write_tokens,
    )
    result = RolloutResult(
        harness_versions=harness_versions,
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
        n_cache_write_tokens=n_cache_write_tokens,
        n_cache_write_5m_tokens=n_cache_write_5m_tokens,
        n_cache_write_1h_tokens=n_cache_write_1h_tokens,
        no_usage_reason=no_usage_reason,
        cost_usd_provider=cost_usd_provider,
        n_calls_cost_estimated=n_calls_cost_estimated,
        usage_incomplete=usage_incomplete,
        usage_incomplete_reason=usage_incomplete_reason,
        usage_details=usage_details,
        judge_usage=judge_usage,
        error=error,
        error_category=error_category,
        verifier_error=verifier_error,
        verifier_error_category=verifier_error_category,
        export_error=export_error,
        partial_trajectory=partial_trajectory,
        trajectory_source=trajectory_source,
        evolved_skills=evolved_skills,
        source_provenance=source_provenance,
        started_at=started_at,
        finished_at=finished_at,
        rollout_dir=rollout_dir,
    )
    timing["total"] = (finished_at - started_at).total_seconds()
    timing = timing_spans.rounded(timing)
    agent_result = {
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": result.n_skill_invocations,
        "n_prompts": result.n_prompts,
        "n_input_tokens": result.n_input_tokens,
        "n_output_tokens": result.n_output_tokens,
        "n_cache_read_tokens": result.n_cache_read_tokens,
        "n_cache_write_tokens": result.n_cache_write_tokens,
        "n_cache_write_5m_tokens": result.n_cache_write_5m_tokens,
        "n_cache_write_1h_tokens": result.n_cache_write_1h_tokens,
        # Alias of n_cache_write_tokens for one release.
        "n_cache_creation_tokens": result.n_cache_creation_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "usage_source": result.usage_source,
        "cost_source": result.cost_source,
        "price_table_date": result.price_table_date,
        # The part of cost_usd the provider billed, and how many calls were
        # filled at list price because their responses carried no bill.
        "cost_usd_provider": result.cost_usd_provider,
        "n_calls_cost_estimated": result.n_calls_cost_estimated,
        "price_source": result.price_source,
        # The earlier name, for readers that tell a metered run from a
        # login's or an agent-routed one (trajectories/results.py,
        # job_export.py's cost_status, summaries' trust rule).
        "usage_source_legacy": result.usage_source_legacy,
        # Why a result has no tokens when nothing was there to count (an
        # oracle made no model call); None otherwise.
        "no_usage_reason": result.no_usage_reason,
        # A count that misses calls says so, and so does its cost.
        **incomplete_usage_fields(result),
    }
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
    timed_out = bool(trajectory) and trajectory[-1].get("type") == "agent_timeout"
    calls_summary = _model_calls_summary(rollout_dir)
    effort = effort_outcome(
        calls_summary,
        requested=reasoning_effort,
        applied=effort_applied,
        route=effort_route,
        requested_from=effort_requested_from,
    )
    warnings = list(warnings or [])
    if (effort_note := effort_warning(effort)) is not None:
        warnings.append(effort_note)
    inputs = (calls_summary or {}).get("input_files")
    inputs = inputs if isinstance(inputs, dict) else None
    warnings += _input_stub_warnings(inputs)
    calls = read_calls(rollout_dir) if calls_summary is not None else []
    outcome = {
        **agent_outcome(
            calls_summary,
            stop_reason=stop_reason,
            agent_error=error is not None or timed_out,
        ),
        **turn_endings(calls_summary, stop_reasons=stop_reasons, error=error),
        **harness_ending(error_category, stop_reasons),
    }
    if trial_budget is not None:
        trial_budget = {
            **trial_budget,
            "used": {
                "tokens": result.total_tokens,
                "turns": (calls_summary or {}).get("calls"),
            },
            "subagents": trial_budget["subagents"]
            if "subagents" in trial_budget
            else subagent_usage(rollout_dir),
        }
    result_data = {
        "task_name": result.task_name,
        "rollout_name": result.rollout_name,
        "rewards": result.rewards,
        "agent": result.agent,
        "agent_name": result.agent_name,
        # Which build of BenchFlow wrote it: version, source commit and
        # whether that tree had uncommitted changes (benchflow.build_info).
        "benchflow_build": benchflow_build(),
        "harness_versions": result.harness_versions,
        "model": result.model,
        **skill_policy.config_metadata(),
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": result.n_skill_invocations,
        "n_prompts": result.n_prompts,
        "agent_result": agent_result,
        # The verifier's LLM judge, in agent_result's fields and never in
        # them: its tokens and cost (rewards.judge_usage); null without one.
        "judge_usage": result.judge_usage,
        "agent_outcome": outcome,
        # The trial ended on a model refusal or a content filter: which call
        # and why (agent_outcome's status refusal), else None.
        "model_refusal": refusal_ending(outcome, calls_summary, calls),
        # Its token and turn budget, what it used and what stopped it; None
        # when it had none (benchflow.trial_budget).
        "trial_budget": trial_budget,
        "effort": effort,
        "tool_surface": _tool_surface(calls_summary, agent),
        "input_delivery": inputs,
        # What the agent's main calls carried (benchflow.run_checks): the
        # job compares it with what the run card asked for.
        "settings_sent": settings_sent(calls, model, agent)
        if calls_summary is not None
        else None,
        # Claude Code's compactions in the trial (rollout._session_log); None
        # when they were not counted.
        "compactions": compactions,
        "final_metrics": final_metrics,
        "trajectory_summary": trajectory_summary,
        "usage_tracking": usage_tracking,
        # Error text can quote a provider's answer or a command line: masked
        # here by the rules results.jsonl and the trajectory are written with.
        "error": _masked(result.error),
        "error_category": result.error_category,
        "verifier_error": _masked(result.verifier_error),
        "verifier_error_category": result.verifier_error_category,
        "export_error": _masked(result.export_error),
        **diagnostics.to_result_fields(),
        "partial_trajectory": result.partial_trajectory,
        "trajectory_source": result.trajectory_source,
        "started_at": started_at.astimezone().isoformat(),
        "finished_at": finished_at.astimezone().isoformat(),
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
        # The host's Yama ptrace_scope as the sandbox read it at start, in
        # every record: under 0 or "absent" a PID 1 of the sandbox user's is
        # refused (sandbox/pid1_ptrace.py); None when it was not read.
        "host_ptrace_scope": host_ptrace_scope,
        # The task's own harness user beside the one BenchFlow ran the agent
        # as, in every record; nothing here changed who ran it.
        "agent_user": agent_user,
        # What test.sh needs, resolved before the agent's turn as the
        # verifier runs it (task/test_sh_preflight.py); None when unchecked.
        "test_sh_preflight": test_sh_preflight,
    }
    result.model_refusal = result_data["model_refusal"]
    # The finding of a trial the writer withheld as an integration failure,
    # as result.json records it (benchflow.integration_health).
    result.integration_failure_info = result_data.get("integration_failure_info")
    result.trial_budget = trial_budget
    if scoring is not None:
        result_data["scoring"] = scoring.to_dict()
    if branches is not None:
        result_data["branches"] = branches
    if cloud_metadata is not None:
        # The cloud metadata block's state, and the run's opt-out when it
        # made one; cloud-metadata.json has the whole record.
        result_data["cloud_metadata"] = cloud_metadata
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
        trajectory=trajectory,
        rewards=rewards,
        model=model,
        effort=effort,
        error=error,
        verifier_error=verifier_error,
        partial_trajectory=partial_trajectory,
        scoring=scoring,
        # ATIF's own meaning: the prompt counts the cache.
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
    # The trial and its model calls as OpenTelemetry spans (and sent where
    # OTEL_EXPORTER_OTLP_*ENDPOINT says, unless the run sends them when it
    # decides the attempt: send_spans); never fails the trial.
    emit_trial_spans(rollout_dir, send=send_spans)
    return result


def _recorded_harness_versions(rollout_dir: Path) -> dict[str, Any] | None:
    """The harness versions config.json records (trial root, out of the
    sandbox's reach), or None."""
    try:
        config = json.loads((rollout_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    versions = config.get("harness_versions") if isinstance(config, dict) else None
    return versions if isinstance(versions, dict) else None


def _tool_surface(summary: dict[str, Any] | None, agent: str) -> dict[str, Any] | None:
    """result.json's ``tool_surface``: the tools the agent's first request
    offered (``model-calls.json`` ``first_agent_tools``: count, names and
    their hash), with ``fallback`` True when they are Codex's fallback surface
    (False for Codex's own, None for other agents); None when no record saw a
    request (no proxy)."""
    from benchflow.agents.codex_config import codex_fallback_tools

    tools = (summary or {}).get("first_agent_tools")
    if not isinstance(tools, dict):
        return None
    names = tools.get("names")
    names = [str(name) for name in names] if isinstance(names, list) else []
    return {
        "count": tools.get("count"),
        "names": names,
        "sha256": tools.get("sha256"),
        "fallback": codex_fallback_tools(names) if agent == "codex-acp" else None,
    }


def _input_stub_warnings(inputs: dict[str, Any] | None) -> list[str]:
    """A trial warning for each file the agent read whose content did not
    reach the model, as far as the proxy's requests show (``input_delivery``
    in result.json, from ``model-calls.json`` ``input_files``)."""
    warnings = []
    for entry in (inputs or {}).get("files") or []:
        if not isinstance(entry, dict) or entry.get("stub") is not True:
            continue
        warnings.append(
            f"input file {entry.get('path')} may not have reached the model: the "
            f"agent's read of it reached the model as {entry.get('text_chars')} "
            "characters of text and no document or image"
            + (f" (route {entry['route']})" if entry.get("route") else "")
            + (
                "; a later call that named the file brought back a long text"
                if entry.get("recovered")
                else ""
            )
        )
    return warnings


def _model_calls_summary(rollout_dir: Path) -> dict[str, Any] | None:
    """The model-call totals teardown wrote (``model-calls.json``), or None
    before then: building a result never reads the proxy's log again."""
    try:
        summary = json.loads((rollout_dir / SUMMARY_FILE).read_text())
    except (OSError, ValueError):
        return None
    return summary if isinstance(summary, dict) else None


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


def _write_trainer_artifact(
    rollout_dir: Path,
    *,
    task_name: str,
    rollout_name: str,
    agent_name: str,
    prompts: list[str],
    trajectory: list[dict],
    rewards: dict | None,
    model: str | None,
    effort: dict[str, Any] | None = None,
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
    atif: bool = True,
) -> None:
    """Emit the trainer-format artifacts for this scored rollout.

    The architecture's train-mode seam (issue #385): every scored rollout
    that reaches result-building produces a trainer-ready Verifiers / ORS
    record (``trainer/verifiers.jsonl``) plus the ecosystem trajectory
    formats — ATIF (``trainer/atif.json``; omitted for empty trajectories,
    which the schema forbids) and ADP (``trainer/adp.jsonl``). Each format
    is written independently; failures are logged by default. Terminal scoring
    commits use ``strict=True`` so a failed export cannot publish a healthy
    final result before its required artifacts exist. ``atif=False`` leaves
    ``trainer/atif.json`` as it is (it carries no reward and no ending).
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
        if atif:
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


def judge_kept_idle_timeout(result: RolloutResult) -> bool:
    """Judge an idle timeout the run keeps as its trial's result.

    A rollout the run's retry loop owns leaves an idle timeout unjudged so
    the loop can retry it: an integration failure is never retried. Once the
    run keeps the attempt (its ``final`` record,
    :mod:`benchflow._utils.attempt_decision`), it is judged
    as a wall-clock timeout is when it is written
    (:func:`benchflow.integration_health.diagnose_kept_idle_timeout`): an agent
    that did nothing before it went idle is an ``agent_integration`` failure,
    unscored, its verifier's reward kept as ``reward_withheld``. result.json
    is rewritten first and is the commit point: if it cannot be written,
    every file and *result* stay as they were. Once it is, *result* says the
    same, and the files derived from it follow (rewards.jsonl removed, the
    results.jsonl row, the reward-bearing trainer files, the trial span
    file; ``atif.json`` carries no reward and stays as written). A derived
    file that cannot be rewritten is removed rather than left saying
    otherwise. The spans are sent when the run records its decision. True
    when it was judged; never fails the trial.
    """
    folder = result.rollout_dir
    if result.error_category != IDLE_TIMEOUT or folder is None:
        return False
    folder = Path(folder)
    try:
        recorded = json.loads((folder / "result.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(recorded, dict):
        return False
    failure = diagnose_kept_idle_timeout(folder, recorded)
    if failure is None:
        return False
    try:
        _withhold_kept_idle_timeout(folder, recorded, failure)
    except Exception as e:  # the trial keeps the ending it was written with
        logger.warning(
            "Could not record the idle timeout of %s as an integration failure: %s",
            folder.name,
            e,
        )
        return False
    result.rewards = None
    result.error = failure.error_text()
    result.error_category = AGENT_INTEGRATION
    info = recorded.get("integration_failure_info")
    result.integration_failure_info = info if isinstance(info, dict) else None
    logger.warning(
        "Agent integration failure [%s]: %s — trial unscored",
        failure.cause,
        failure.evidence,
    )
    return True


def _withhold_kept_idle_timeout(
    folder: Path, recorded: dict[str, Any], failure: IntegrationFailure
) -> None:
    """Rewrite a kept idle timeout's records as the rollout writes a judged
    wall-clock timeout's: the reward withheld and the finding recorded.
    Raises only up to the commit (result.json written)."""
    trajectory: list[dict[str, Any]] = []
    for line in (
        (folder / "trajectory" / "acp_trajectory.jsonl").read_text().splitlines()
    ):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            trajectory.append(event)
    prompts = json.loads((folder / "prompts.json").read_text())
    info = failure.diagnostic(reward_withheld=recorded.get("rewards"))
    recorded.update(
        {
            "rewards": None,
            "error": _masked(failure.error_text()),
            "error_category": AGENT_INTEGRATION,
            info.field: redact_trajectory_obj(info.to_dict()),
        }
    )
    outcome = recorded.get("agent_outcome")
    if isinstance(outcome, dict):
        # The ending a judged trial has (run_checks.harness_ending): its
        # category is no longer the harness's idle timeout.
        outcome.update(harness_ending(AGENT_INTEGRATION, outcome.get("stop_reasons")))
    task_name = recorded["task_name"]
    rollout_name = recorded.get("rollout_name") or folder.name
    agent_result = recorded.get("agent_result") or {}
    scoring = recorded.get("scoring")
    scoring = (
        ScoringResult.model_validate(scoring, strict=False)
        if isinstance(scoring, dict)
        else None
    )
    partial = bool(recorded.get("partial_trajectory"))
    # The commit point: once result.json is written the trial is judged, and
    # every file derived from it follows it or is removed (_derive).
    write_json_atomic(folder / "result.json", recorded)
    _derive(folder, "rewards.jsonl", lambda: _drop_rewards(folder), [])
    # What _build_rollout_result gives the trainer writer, but the reward and
    # the error: atif.json, which alone takes the token totals, stays as it is.
    _derive(
        folder,
        "the trainer files",
        lambda: _write_trainer_artifact(
            folder,
            task_name=task_name,
            rollout_name=rollout_name,
            agent_name=recorded.get("agent_name") or recorded.get("agent") or "",
            prompts=prompts,
            trajectory=trajectory,
            rewards=None,
            model=recorded.get("model"),
            error=recorded["error"],
            verifier_error=recorded.get("verifier_error"),
            partial_trajectory=partial,
            scoring=scoring,
            strict=True,
            atif=False,
        ),
        [folder / "trainer" / name for name in ("verifiers.jsonl", "adp.jsonl")],
    )

    def row() -> None:
        from benchflow.trajectories.results import write_rollout_results_jsonl

        write_rollout_results_jsonl(
            folder,
            task_name=task_name,
            rollout_name=rollout_name,
            agent=recorded.get("agent") or "",
            agent_name=recorded.get("agent_name") or "",
            model=recorded.get("model"),
            n_tool_calls=recorded.get("n_tool_calls") or 0,
            prompts=prompts,
            trajectory=trajectory,
            partial_trajectory=partial,
            rewards=None,
            error=recorded["error"],
            verifier_error=recorded.get("verifier_error"),
            error_category=AGENT_INTEGRATION,
            verifier_error_category=recorded.get("verifier_error_category"),
            diagnostics=RolloutDiagnostics.from_result_fields(recorded),
            export_error=recorded.get("export_error"),
            timing=recorded.get("timing"),
            agent_result=agent_result,
            scoring=scoring,
            purpose=recorded.get("purpose") or "task",
            parent_rollout=recorded.get("parent_rollout"),
            branches=recorded.get("branches"),
        )

    _derive(folder, "results.jsonl", row, [folder / "results.jsonl"])

    def spans() -> None:
        # The judged ending; the run sends the file with its decision.
        if emit_trial_spans(folder, send=False) is None:
            raise OSError("the trial spans were not written")

    _derive(
        folder, "the trial spans", spans, [folder / "trajectory" / "otel-spans.json"]
    )


def _derive(
    folder: Path, what: str, step: Callable[[], object], stale: list[Path]
) -> None:
    """Rewrite one file derived from a judged result.json; one that cannot
    be rewritten (a full disk, a read-only folder) is removed, so no record
    says otherwise than result.json. Never raises."""
    try:
        step()
    except Exception as e:
        logger.warning(
            "Could not rewrite %s of %s after judging it (%s); removing them",
            what,
            folder.name,
            e,
        )
        for path in stale:
            try:
                path.unlink(missing_ok=True)
            except OSError as gone:
                logger.warning("Could not remove %s either: %s", path, gone)


def _drop_rewards(folder: Path) -> None:
    """Remove rewards.jsonl; if the folder refuses that, empty the file (a
    withheld trial has no reward events)."""
    path = folder / "rewards.jsonl"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        path.write_text("")


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
