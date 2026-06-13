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
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._types import Role, Scene
from benchflow._utils.result_metadata import (
    final_metrics_from_agent_result,
    trajectory_summary_from_events,
)
from benchflow._utils.reward_events import build_rewards_jsonl_events
from benchflow._utils.scoring import classify_error, classify_verifier_error
from benchflow.contracts import (
    BaseUser,
    DocumentNudgeUser,
    ModelDocumentNudgeUser,
)
from benchflow.diagnostics import RolloutDiagnostics
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.models import RolloutResult, TrajectorySource
from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    TaskSkillPolicy,
    resolve_task_skill_policy,
)
from benchflow.trajectories._capture import TrajectoryWriter
from benchflow.trajectories.metrics import count_skill_invocations
from benchflow.usage_tracking import UsageTrackingConfig

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


# Substrings that flag an env var name as secret-bearing for ``config.json``
# redaction. Matching is case-insensitive (callers ``.upper()`` the key first)
# and uses substring containment so derived names like ``MY_AUTH_HEADER``,
# ``SESSION_COOKIE``, or ``GH_TOKEN`` are caught. This stays a denylist (rather
# than an allowlist of safe keys) because agent env varies per agent — the
# union of safe keys is not knowable here — but the list now covers the common
# auth-bearing names that issue #410 called out (COOKIE, AUTHORIZATION, AUTH,
# BEARER, SESSION) on top of the original KEY/TOKEN/SECRET/PASSWORD/CREDENTIALS.
_SECRET_ENV_SUBSTRINGS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIALS",
    "COOKIE",
    "AUTHORIZATION",
    "AUTH",
    "BEARER",
    "SESSION",
)
_SECRET_URL_PATH_MARKERS: tuple[str, ...] = ("/__benchflow/",)


def _is_secret_env_key(name: str) -> bool:
    """Return True if *name* looks like it carries a secret value.

    Case-insensitive substring match against :data:`_SECRET_ENV_SUBSTRINGS`.
    Used by :func:`_write_config` to drop secret-bearing entries before
    persisting ``agent_env`` to the rollout's ``config.json``.
    """
    upper = name.upper()
    return any(s in upper for s in _SECRET_ENV_SUBSTRINGS)


def _is_secret_env_value(name: str, value: str) -> bool:
    """Return True if a normally public env value embeds a runtime secret."""
    upper = name.upper()
    if not upper.endswith("BASE_URL"):
        return False
    return any(marker in value for marker in _SECRET_URL_PATH_MARKERS)


def _should_record_env_entry(name: str, value: str) -> bool:
    return not _is_secret_env_key(name) and not _is_secret_env_value(name, value)


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
    timeout: int,
    started_at: datetime,
    agent_env: dict[str, str],
    reasoning_effort: str | None = None,
    usage_tracking: UsageTrackingConfig | None = None,
    concurrency: int | None = None,
    agent_idle_timeout: int | None = None,
    scenes: list[Scene] | None = None,
    source_provenance: dict[str, Any] | None = None,
    environment_manifest: EnvironmentManifest | None = None,
) -> None:
    """Write config.json to rollout_dir with secrets filtered out."""
    from benchflow.agents.install import effective_install_timeout

    recorded_env = {
        k: v for k, v in agent_env.items() if _should_record_env_entry(k, v)
    }
    config_data = {
        "task_path": str(task_path),
        "agent": agent,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "environment": environment,
        "environment_manifest": _environment_manifest_metadata(environment_manifest),
        **skill_policy.config_metadata(),
        "sandbox_user": sandbox_user,
        "sandbox_locked_paths": sandbox_locked_paths,
        "sandbox_setup_timeout": sandbox_setup_timeout,
        "agent_install_timeout": effective_install_timeout(
            agent, sandbox_setup_timeout
        ),
        "context_root": str(context_root) if context_root else None,
        "timeout_sec": timeout,
        "concurrency": concurrency,
        "agent_idle_timeout_sec": agent_idle_timeout,
        "started_at": str(started_at),
        "agent_env": recorded_env,
        "scenes": _scene_metadata(scenes or []),
    }
    if usage_tracking is not None:
        config_data["usage_tracking"] = usage_tracking.to_config_artifact()
    if source_provenance is not None:
        config_data["source"] = source_provenance
    (rollout_dir / "config.json").write_text(json.dumps(config_data, indent=2))


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
    timing: dict[str, float],
    scenes: list[Scene] | None = None,
    n_input_tokens: int | None = None,
    n_output_tokens: int | None = None,
    n_cache_read_tokens: int | None = None,
    n_cache_creation_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    usage_source: str = "unavailable",
    price_source: str | None = None,
    usage_details: dict[str, Any] | None = None,
    usage_tracking: dict[str, Any] | None = None,
    evolved_skills: dict[str, str] | None = None,
    source_provenance: dict[str, Any] | None = None,
    diagnostics: RolloutDiagnostics | None = None,
    skill_policy: TaskSkillPolicy | None = None,
    sandbox_id: str | None = None,
) -> RolloutResult:
    """Build RolloutResult and write result.json, timing.json, prompts.json, trajectory.

    Diagnostics flow through the :class:`RolloutDiagnostics` collector
    (issue #503). Callers that previously passed per-field ``*_info``
    dicts should construct a collector and ``set()`` typed diagnostics.
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
    verifier_error_category = (
        diagnostics.category_for_channel("verifier_error")
        if verifier_error is not None
        else None
    ) or classify_verifier_error(verifier_error)
    result = RolloutResult(
        task_name=task_name,
        rollout_name=rollout_name,
        rewards=rewards,
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
        usage_source=usage_source,
        price_source=price_source,
        usage_details=usage_details,
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
    )
    timing["total"] = (finished_at - started_at).total_seconds()
    timing = {k: round(v, 1) for k, v in timing.items()}
    agent_result = {
        "n_tool_calls": result.n_tool_calls,
        "n_skill_invocations": result.n_skill_invocations,
        "n_prompts": result.n_prompts,
        "n_input_tokens": result.n_input_tokens,
        "n_output_tokens": result.n_output_tokens,
        "n_cache_read_tokens": result.n_cache_read_tokens,
        "n_cache_creation_tokens": result.n_cache_creation_tokens,
        "total_tokens": result.total_tokens,
        "cost_usd": result.cost_usd,
        "usage_source": result.usage_source,
        "price_source": result.price_source,
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
    (rollout_dir / "result.json").write_text(
        json.dumps(
            {
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
                **(
                    {"source": source_provenance}
                    if source_provenance is not None
                    else {}
                ),
                "sandbox_id": sandbox_id,
            },
            indent=2,
        )
    )
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
        verifier_error=verifier_error,
        total_prompt_tokens=n_input_tokens,
        total_completion_tokens=n_output_tokens,
        total_cached_tokens=n_cache_read_tokens,
        total_cost_usd=cost_usd,
    )
    return result


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
    verifier_error: str | None,
    total_prompt_tokens: int | None = None,
    total_completion_tokens: int | None = None,
    total_cached_tokens: int | None = None,
    total_cost_usd: float | None = None,
) -> None:
    """Emit the trainer-format artifacts for this scored rollout.

    The architecture's train-mode seam (issue #385): every scored rollout
    that reaches result-building produces a trainer-ready Verifiers / ORS
    record (``trainer/verifiers.jsonl``) plus the ecosystem trajectory
    formats — ATIF (``trainer/atif.json``; omitted for empty trajectories,
    which the schema forbids) and ADP (``trainer/adp.jsonl``). Each format
    is written independently; failures are logged but never block result
    writing or each other.
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
    try:
        write_rollout_verifiers_jsonl(
            rollout_dir,
            task_id=task_name,
            prompts=prompts,
            trajectory=trajectory,
            rewards=rewards,
            model=model,
            environment=task_name,
            error=verifier_error,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Trainer artifact write failed: %s", e)
    try:
        write_rollout_atif_json(
            rollout_dir,
            session_id=trajectory_id,
            agent_name=agent_name,
            prompts=prompts,
            trajectory=trajectory,
            model=model,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cached_tokens=total_cached_tokens,
            total_cost_usd=total_cost_usd,
        )
    except Exception as e:  # pragma: no cover - defensive
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
            reward=(rewards or {}).get("reward"),
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("ADP artifact write failed: %s", e)


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
