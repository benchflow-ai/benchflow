"""Rollout — the single execution path for one agent-on-task evaluation.

The ``Rollout`` class owns the 5-phase lifecycle::

    rollout = await Rollout.create(RolloutConfig(task_path=..., agent=..., model=...))
    await rollout.setup()
    await rollout.start()
    await rollout.install_agent()
    await rollout.connect()
    await rollout.execute()
    result = await rollout.verify()
    await rollout.cleanup()

Or use ``rollout.run()`` for the full lifecycle.

Phases can be composed for multi-agent flows::

    await rollout.setup()
    await rollout.start()
    await rollout.install_agent()

    # Coder turn
    await rollout.connect()
    await rollout.execute(prompts=[coder_prompt])
    await rollout.disconnect()

    # Reviewer turn (same sandbox, new ACP session)
    await rollout.connect()
    await rollout.execute(prompts=[reviewer_prompt])
    await rollout.disconnect()

    result = await rollout.verify()
    await rollout.cleanup()

See also: ``RolloutConfig`` for configuration dataclass.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._types import Role, Scene, Turn
from benchflow._utils.config import (
    normalize_agent_name,
    normalize_reasoning_effort,
    normalize_sandbox_user,
)
from benchflow._utils.result_metadata import (
    final_metrics_from_agent_result,
    trajectory_summary_from_events,
)
from benchflow._utils.scoring import classify_error, classify_verifier_error
from benchflow.contracts import (
    AgentProtocolError,
    BaseUser,
    Environment,
    RolloutPlanes,
    RoundResult,
    SandboxStartupFailure,
    default_rollout_planes,
)
from benchflow.diagnostics import RolloutDiagnostics, VerifierTimeoutDiagnostic
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.models import RolloutResult, TrajectorySource
from benchflow.rewards.validation import validate_reward_map
from benchflow.rollout_branch import ChildRunner
from benchflow.rollout_branch import branch as _branch_engine
from benchflow.scenes import (
    compile_scenes_to_steps,
    scene_step_prompt,
    scene_step_role,
    scene_step_skills_dir,
)
from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_SELF_GEN,
    SKILL_MODE_WITH_SKILL,
    TaskSkillPolicy,
    normalize_skill_mode,
    resolve_task_skill_policy,
    strip_task_bundled_skills,
    task_bundled_skills_dir,
)
from benchflow.trajectories._capture import (
    TrajectoryWriter,
    _capture_session_trajectory,
    _scrape_agent_trajectory,
    make_trajectory_sink,
)
from benchflow.trajectories.metrics import count_skill_invocations
from benchflow.trajectories.tree import RolloutNode, RolloutTree, Step
from benchflow.usage_tracking import UsageTrackingConfig

logger = logging.getLogger(__name__)

_DISALLOW_WEB_TOOLS_ENV = "BENCHFLOW_DISALLOW_WEB_TOOLS"
GENERATED_SKILLS_ROOT = "/app/generated-skills"


def _task_disallows_internet(task: Any) -> bool:
    """Return True when task config requests no internet for the agent task."""
    env_config = getattr(getattr(task, "config", None), "environment", None)
    return getattr(env_config, "allow_internet", True) is False


def _read_task_instruction(task_path: Path) -> str:
    """Read the agent-facing instruction from the selected task entrypoint."""
    from benchflow.task.package import TaskRuntimeView

    return TaskRuntimeView.from_task_dir(task_path).materialize_instruction_md()


def _environment_uses_prebuilt_image(
    env_config: object | None, environment_manifest: EnvironmentManifest | None
) -> bool:
    """Return True when sandbox startup will skip the task Dockerfile build."""
    if env_config is not None and getattr(env_config, "docker_image", None):
        return True
    if environment_manifest is None:
        return False
    from benchflow.environment.manifest import resolve_manifest_image

    return bool(resolve_manifest_image(environment_manifest))


def _apply_web_policy(agent_env: dict[str, str], *, disallow: bool) -> dict[str, str]:
    """Inject BenchFlow's no-web policy marker into agent env when requested."""
    if not disallow:
        return agent_env
    return {**agent_env, _DISALLOW_WEB_TOOLS_ENV: "1"}


def _agent_launch_with_web_policy(
    agent: str, *, disallow: bool, planes: RolloutPlanes | None = None
) -> str:
    """Return launch command, appending the agent's no-web launch knob if any."""
    return (planes or default_rollout_planes()).agent_launch(
        agent, disallow_web_tools=disallow
    )


def _agent_process_kill_pattern(agent_launch: str) -> str | None:
    """Return a pkill -f pattern for the launched agent binary."""
    if not agent_launch.strip():
        return None
    agent_cmd = agent_launch.split()[0].split("/")[-1]
    if not agent_cmd:
        return None
    return rf"(^|[ /]){re.escape(agent_cmd)}( |$)"


def _skill_nudge(agent_env: dict[str, str] | None) -> str:
    """Read skill nudge from explicit agent env or the host environment."""
    return (agent_env or {}).get("BENCHFLOW_SKILL_NUDGE") or os.environ.get(
        "BENCHFLOW_SKILL_NUDGE", ""
    )


def _safe_skill_name(value: str) -> str:
    """Return an AgentSkills-compatible generated skill directory name."""
    import re

    name = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    name = re.sub(r"-+", "-", name).strip("-")
    return name or "generated-task"


def _skill_frontmatter_name(skill_dir: Path) -> str:
    """Read a skill's frontmatter name, falling back to the directory name."""
    from benchflow.skills import parse_skill

    info = parse_skill(skill_dir / "SKILL.md")
    return info.name if info and info.name else skill_dir.name


def _resolve_skill_creator_root(path: str | Path | None) -> tuple[Path, str]:
    """Resolve a skills root containing the official skill-creator directory.

    BenchFlow mounts skills as a root directory whose children are individual
    skill directories. If the caller points directly at skill-creator, use its
    parent as the mounted root and leave the skill pack contents unchanged.
    """
    candidates: list[Path] = []
    scan_single_skill_roots: set[Path] = set()
    if path:
        explicit_path = Path(path).expanduser()
        candidates.append(explicit_path)
        scan_single_skill_roots.add(explicit_path)
    env_path = os.environ.get("BENCHFLOW_SKILL_CREATOR_DIR")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        candidates.append(env_candidate)
        scan_single_skill_roots.add(env_candidate)
    repo_skill_creator = (
        Path(__file__).resolve().parents[2] / ".claude" / "skills" / "skill-creator"
    )
    cwd_skill_creator = Path.cwd() / ".claude" / "skills" / "skill-creator"
    candidates.append(repo_skill_creator)
    if cwd_skill_creator != repo_skill_creator:
        candidates.append(cwd_skill_creator)
    candidates.extend(
        [
            Path.home() / ".claude" / "skills" / "skill-creator",
            Path.home() / ".codex" / "skills" / ".system" / "skill-creator",
            Path.home() / ".agents" / "skills" / "skill-creator",
        ]
    )

    for candidate in candidates:
        if (candidate / "SKILL.md").exists():
            return candidate.parent, _skill_frontmatter_name(candidate)
        if (candidate / "skill-creator" / "SKILL.md").exists():
            skill_dir = candidate / "skill-creator"
            return candidate, _skill_frontmatter_name(skill_dir)
        if candidate in scan_single_skill_roots and candidate.is_dir():
            skill_dirs = [
                child
                for child in candidate.iterdir()
                if child.is_dir() and (child / "SKILL.md").exists()
            ]
            if len(skill_dirs) == 1:
                skill_dir = skill_dirs[0]
                return candidate, _skill_frontmatter_name(skill_dir)

    checked = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find skill-creator. Pass --skill-creator-dir or set "
        f"BENCHFLOW_SKILL_CREATOR_DIR. Checked: {checked}"
    )


def _self_gen_prompt(
    task_path: Path, generated_skills_root: str, skill_creator_name: str
) -> str:
    """Prompt the clean creator agent to use the mounted skill-creator skill."""
    skill_dir_name = f"{_safe_skill_name(task_path.name)}-skill"
    target_dir = f"{generated_skills_root}/{skill_dir_name}"
    return f"""Use the {skill_creator_name} skill exactly as provided.

Read /instruction.md and inspect the task environment only as needed to understand the reusable workflow. Do not solve the task directly.

Create one or more complete Anthropic-standard skill packs as immediate child directories under:

{generated_skills_root}

Use this suggested path if one skill is enough:

{target_dir}

Each generated skill pack path must look like {generated_skills_root}/<skill-name>/SKILL.md. It may include scripts/, references/, assets/, examples, or other bundled resources when they help a fresh solver avoid repeated work.

The solver context will start with a clean agent session and only the generated skill packs mounted. Make the skills useful for solving this task type from the same sandbox environment."""


async def _ensure_sandbox_dir(
    env: Any, path: str | Path, sandbox_user: str | None = None
) -> None:
    """Create a sandbox directory and optionally make it writable by the agent."""
    q_path = shlex.quote(str(path))
    cmd = f"mkdir -p {q_path}"
    if sandbox_user:
        q_user = shlex.quote(sandbox_user)
        cmd += f" && chown -R {q_user}:{q_user} {q_path}"
    result = await env.exec(cmd, timeout_sec=10)
    if result.return_code != 0:
        raise RuntimeError(
            f"Failed to create sandbox directory {path}: "
            f"{result.stderr or result.stdout}"
        )


_DIAG_TRUNCATE = 2000


def _write_rewards_jsonl(
    rollout_dir: Path,
    rewards: dict | None,
    finished_at: datetime,
) -> None:
    """Write rewards.jsonl — one JSON line per reward event.

    Architecture (``docs/architecture.md``, "Evaluation — the five spaces"):
    every reward record is tagged ``(space, granularity, value)``. Promote
    those tags from any verifier-supplied per-item dict to first-class
    fields on each line, falling back to the ``RewardEvent`` defaults
    (``space="output"``; ``granularity="step"`` for rubric items,
    ``"terminal"`` for the scalar reward) when the verifier omits them.
    """
    from typing import cast

    if not rewards:
        return
    events: list[dict[str, Any]] = []
    rubric = rewards.get("rubric")
    if isinstance(rubric, list):
        for i, item in enumerate(rubric):
            if not isinstance(item, dict):
                continue
            rubric_item = cast(dict[str, Any], item)
            events.append(
                {
                    "ts": finished_at.isoformat(),
                    "type": "process",
                    "source": "verifier_rubric",
                    "value": rubric_item.get("score", 0.0),
                    "tag": rubric_item.get("name", f"rubric_{i}"),
                    "step_index": i,
                    "space": rubric_item.get("space", "output"),
                    "granularity": rubric_item.get("granularity", "step"),
                    "meta": {
                        k: v
                        for k, v in rubric_item.items()
                        if k not in ("score", "name", "space", "granularity")
                    },
                }
            )
    scalar = rewards.get("reward")
    if scalar is not None:
        non_event_keys = {"reward", "rubric", "space", "granularity"}
        events.append(
            {
                "ts": finished_at.isoformat(),
                "type": "terminal",
                "source": "verifier",
                "value": scalar,
                "tag": "reward",
                "step_index": None,
                "space": rewards.get("space", "output"),
                "granularity": rewards.get("granularity", "terminal"),
                "meta": {k: v for k, v in rewards.items() if k not in non_event_keys},
            }
        )
    if events:
        path = rollout_dir / "rewards.jsonl"
        path.write_text("\n".join(json.dumps(e, default=str) for e in events) + "\n")


def _init_rollout(
    task_path: Path,
    job_name: str | None,
    rollout_name: str | None,
    jobs_dir: str | Path,
) -> tuple[Any, Path, Any, datetime, str, str]:
    """Set up trial directory tree and return core trial objects."""
    from uuid import uuid4

    from benchflow.task import RolloutPaths, Task

    task = Task(task_path)
    job_name = job_name or datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    rollout_name = rollout_name or f"{task_path.name}__{uuid4().hex[:8]}"
    rollout_dir = Path(jobs_dir) / job_name / rollout_name
    rollout_paths = RolloutPaths(rollout_dir=rollout_dir)
    started_at = datetime.now()
    rollout_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("agent", "verifier", "artifacts", "trajectory"):
        (rollout_dir / subdir).mkdir(exist_ok=True)
    return task, rollout_dir, rollout_paths, started_at, job_name, rollout_name


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
) -> None:
    """Write config.json to rollout_dir with secrets filtered out."""
    recorded_env = {
        k: v for k, v in agent_env.items() if _should_record_env_entry(k, v)
    }
    config_data = {
        "task_path": str(task_path),
        "agent": agent,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "environment": environment,
        **skill_policy.config_metadata(),
        "sandbox_user": sandbox_user,
        "sandbox_locked_paths": sandbox_locked_paths,
        "sandbox_setup_timeout": sandbox_setup_timeout,
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
    usage_tracking: dict[str, Any] | None = None,
    evolved_skills: dict[str, str] | None = None,
    source_provenance: dict[str, Any] | None = None,
    diagnostics: RolloutDiagnostics | None = None,
    skill_policy: TaskSkillPolicy | None = None,
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
        prompts=prompts,
        trajectory=trajectory,
        rewards=rewards,
        model=model,
        verifier_error=verifier_error,
    )
    return result


def _write_trainer_artifact(
    rollout_dir: Path,
    *,
    task_name: str,
    prompts: list[str],
    trajectory: list[dict],
    rewards: dict | None,
    model: str | None,
    verifier_error: str | None,
) -> None:
    """Emit ``rollout_dir/trainer/verifiers.jsonl`` for this scored rollout.

    The architecture's train-mode seam (issue #385): every scored rollout
    that reaches result-building should produce a trainer-ready Verifiers /
    ORS record so prime-rl / Verifiers can ingest the run directly. Failures
    here are logged but never block result writing.
    """
    from benchflow.trajectories.export import write_rollout_verifiers_jsonl

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


def _resolve_prompts(
    task_path: Path,
    prompts: list[str | None] | None,
    skills_dir: str | Path | None = None,
    task_skills_dir: str | Path | None = None,
    skill_nudge: str = "",
    agent: str | None = None,
    planes: RolloutPlanes | None = None,
) -> list[str]:
    """Read the task instruction and resolve prompt list."""
    instruction = _read_task_instruction(task_path)

    if skill_nudge:
        skill_display_path = "~/.claude/skills"
        if agent:
            agent_cfg = (planes or default_rollout_planes()).agent_config(agent)
            if agent_cfg and agent_cfg.skill_paths:
                skill_display_path = agent_cfg.skill_paths[0].replace("$HOME", "~")

        skills = []
        for src in [skills_dir, task_skills_dir]:
            if src and Path(src).is_dir():
                for d in sorted(Path(src).iterdir()):
                    if d.is_dir() and (d / "SKILL.md").exists():
                        content = d.joinpath("SKILL.md").read_text()
                        name = d.name
                        desc = ""
                        if content.startswith("---"):
                            parts = content.split("---", 2)
                            if len(parts) >= 3:
                                import yaml

                                try:
                                    fm = yaml.safe_load(parts[1])
                                    desc = fm.get("description", "") if fm else ""
                                except Exception:
                                    pass
                        skills.append({"name": name, "desc": desc, "content": content})
                if skills:
                    break

        if skills:
            if skill_nudge == "name":
                names = ", ".join(s["name"] for s in skills)
                nudge = f"Skills available at {skill_display_path}: {names}. Read them before starting."
                instruction = nudge + "\n\n" + instruction
            elif skill_nudge == "description":
                lines = [f"Skills available at {skill_display_path}:\n"]
                for s in skills:
                    lines.append(f"- **{s['name']}**: {s['desc']}")
                lines.append("\nRead the relevant skills before starting.")
                instruction = "\n".join(lines) + "\n\n" + instruction
            elif skill_nudge == "full":
                blocks = []
                for s in skills:
                    blocks.append(
                        f'<skill name="{s["name"]}">\n{s["content"]}\n</skill>'
                    )
                instruction = "\n\n".join(blocks) + "\n\n" + instruction

    if prompts is None:
        return [instruction]
    return [p if p is not None else instruction for p in prompts]


async def _start_env_and_upload(
    env: Any, task_path: Path, timing: dict, *, skip_start: bool = False
) -> None:
    """Start environment and upload task files.

    ``skip_start=True`` is used when the sandbox was created and started
    by the caller (Runtime with a live Environment, #388) — we still
    upload task files but must not re-run ``start()`` since most sandbox
    backends (e.g. daytona) are not idempotent.
    """
    if skip_start:
        logger.info(f"Reusing caller-owned environment: {task_path.name}")
        timing["environment_setup"] = 0.0
    else:
        logger.info(f"Starting environment: {task_path.name}")
        t0 = datetime.now()
        await env.start(force_build=False)
        timing["environment_setup"] = (datetime.now() - t0).total_seconds()
    from benchflow.task.package import TaskRuntimeView
    from benchflow.task.paths import SandboxPaths

    sandbox_paths = SandboxPaths()
    runtime_view = TaskRuntimeView.from_task_dir(task_path)
    paths = runtime_view.paths
    if (
        runtime_view.entrypoint == "legacy-split"
        and paths.instruction_path.exists()
    ):
        await env.upload_file(
            paths.instruction_path, str(sandbox_paths.instruction_path)
        )
    else:
        instruction = runtime_view.materialize_instruction_md()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write(instruction)
            f.write("\n")
            temp_instruction = Path(f.name)
        try:
            await env.upload_file(
                temp_instruction, str(sandbox_paths.instruction_path)
            )
        finally:
            temp_instruction.unlink(missing_ok=True)
    if paths.task_document_path.exists():
        await env.upload_file(
            paths.task_document_path, str(sandbox_paths.task_document_path)
        )
    if runtime_view.oracle_dir.is_dir():
        target_dir = (
            sandbox_paths.oracle_dir
            if runtime_view.uses_native_oracle_dir
            else sandbox_paths.solution_dir
        )
        await env.upload_dir(runtime_view.oracle_dir, str(target_dir))


async def _run_oracle(
    env: Any, task_path: Path, timeout: int, sandbox_user: str | None = None
) -> tuple[list[dict], str]:
    """Run oracle mode (oracle/solve.sh or legacy solution/solve.sh)."""
    from benchflow.task import Task, resolve_env_vars
    from benchflow.task.paths import SandboxPaths

    logger.info("Oracle mode: running oracle solve.sh")
    task = Task(task_path)
    if not task.paths.solve_path.exists():
        raise FileNotFoundError(
            f"Oracle requires oracle/solve.sh or legacy solution/solve.sh: {task_path}"
        )
    sandbox_paths = SandboxPaths()
    oracle_dir = (
        sandbox_paths.oracle_dir
        if task.paths.uses_native_oracle_dir
        else sandbox_paths.solution_dir
    )
    oracle_command_label = (
        "oracle/solve.sh"
        if task.paths.uses_native_oracle_dir
        else "solution/solve.sh"
    )
    oracle_script = shlex.quote(str(oracle_dir / "solve.sh"))
    if sandbox_user:
        oracle_cmd = f"DEBIAN_FRONTEND=noninteractive bash {oracle_script}"
        cmd = (
            f"su -s /bin/bash {shlex.quote(sandbox_user)} -c {shlex.quote(oracle_cmd)}"
        )
    else:
        cmd = f"bash {oracle_script}"
    oracle_env: dict[str, str] = {"DEBIAN_FRONTEND": "noninteractive"}
    if task.config.solution.env:
        oracle_env.update(resolve_env_vars(task.config.solution.env))
    result = await env.exec(
        f"{cmd} > /logs/agent/oracle.txt 2>&1",
        env=oracle_env,
        timeout_sec=timeout,
    )
    if result.return_code != 0:
        logger.warning(f"Oracle solve.sh exited with rc={result.return_code}")
    preview = await env.exec(
        f"tail -c {shlex.quote(str(_DIAG_TRUNCATE))} /logs/agent/oracle.txt 2>/dev/null || true",
        user="root",
        timeout_sec=10,
    )
    trajectory = [
        {
            "type": "oracle",
            "command": oracle_command_label,
            "return_code": result.return_code,
            "stdout": (preview.stdout or "")[:_DIAG_TRUNCATE],
        }
    ]
    return trajectory, "oracle"


async def _publish_trajectory_for_verifier(env, trajectory: list[dict]) -> None:
    """Make the captured ACP trajectory available inside /logs for verifiers."""
    if not trajectory:
        return
    await env.exec("mkdir -p /logs/agent", user="root", timeout_sec=10)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(json.dumps(e, default=str) for e in trajectory))
        f.write("\n")
        tmp_path = f.name
    try:
        await env.upload_file(tmp_path, "/logs/agent/acp_trajectory.jsonl")
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)


async def _verify_rollout(
    env: Any,
    task: Any,
    rollout_paths: Any,
    timing: dict,
    planes: RolloutPlanes,
    sandbox_user: str | None = None,
    workspace: str | None = None,
) -> tuple[dict | None, str | None, VerifierTimeoutDiagnostic | None]:
    """Run verifier with pre-verification hardening.

    Returns ``(rewards, verifier_error, verifier_timeout_diagnostic)``. The
    diagnostic is non-``None`` only when the verifier exceeded its timeout
    budget — the agent-error channel is unused (issue #503).
    """
    rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    verifier_error = None
    verifier_timeout: VerifierTimeoutDiagnostic | None = None
    timeout_budget = task.config.verifier.timeout_sec
    try:
        await planes.harden_before_verify(env, task, sandbox_user, workspace=workspace)
        logger.info("Running verifier...")
        verifier = planes.verifier(task=task, rollout_paths=rollout_paths, sandbox=env)
        verifier_result = await asyncio.wait_for(
            verifier.verify(),
            timeout=timeout_budget,
        )
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        rewards = _ensure_canonical_rewards(verifier_result.rewards)
        logger.info(f"Rewards: {rewards}")
    except TimeoutError:
        elapsed = (datetime.now() - t0).total_seconds()
        timing["verifier"] = elapsed
        verifier_error = f"verifier timed out after {timeout_budget}s"
        verifier_timeout = VerifierTimeoutDiagnostic(
            timeout_budget_sec=timeout_budget,
            elapsed_sec=round(elapsed, 1),
            task_name=task.name,
        )
        rewards = None
        logger.error(verifier_error)
    except Exception as e:
        timing["verifier"] = (datetime.now() - t0).total_seconds()
        verifier_error = f"verifier crashed: {e}"
        rewards = None
        logger.error(verifier_error)
    return rewards, verifier_error, verifier_timeout


def _ensure_canonical_rewards(rewards: dict | None) -> dict:
    return validate_reward_map(rewards, source="verifier")


def _install_docker_compat(planes: RolloutPlanes | None = None) -> None:
    """Activate the Docker DinD compatibility shim.

    Called from ``Rollout.__init__`` so importing ``benchflow.rollout`` has
    no side effects on the Docker sandbox. The underlying patch is
    idempotent — safe to call once per rollout construction.
    """
    (planes or default_rollout_planes()).install_docker_compat()


def _task_document_scenes(
    task_path: Path,
    *,
    prompts: list[str | None] | None,
    skill_mode: str,
) -> list[Scene]:
    """Load scene declarations from ``task.md`` when it is the task entrypoint."""
    if prompts is not None or skill_mode == SKILL_MODE_SELF_GEN:
        return []
    document_path = task_path / "task.md"
    if not document_path.exists():
        return []

    from benchflow.task.document import TaskDocument

    return list(TaskDocument.from_path(document_path).scenes)


__all__ = [
    "Role",
    "Scene",
    "Turn",
    "Rollout",
    "RolloutConfig",
]


@dataclass
class RolloutConfig:
    """Declarative rollout configuration.

    A rollout is a sequence of scenes executed in a shared sandbox.
    Single-agent runs are a rollout with one scene containing one role.
    """

    task_path: Path
    scenes: list[Scene] = field(default_factory=list)
    environment: str = "docker"
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    sandbox_setup_timeout: int = 120
    services: list[str] | None = None
    job_name: str | None = None
    rollout_name: str | None = None
    jobs_dir: str | Path = "jobs"
    concurrency: int = 1
    context_root: str | Path | None = None
    pre_agent_hooks: list | None = None
    # Environment plane — when set, Rollout provisions a manifest-declared
    # stateful environment, gates on readiness before the agent runs, and
    # tears it down in cleanup. None => no separate environment plane (default).
    environment_manifest: EnvironmentManifest | None = None
    # Abort the prompt if no tool call arrives for this many seconds.
    # Catches agents that hung silently while the local process is alive
    # (e.g. gemini-cli not responding). None disables idle detection.
    agent_idle_timeout: int | None = 600
    # Hard wall-clock budget for the agent prompt, in seconds. When set,
    # overrides the per-task default (task config [agent] timeout_sec).
    # ``None`` keeps the task's own default — this is the seam through which
    # ``Runtime(RuntimeConfig(timeout=...))`` enforces a caller-supplied
    # budget without editing every task definition (#378).
    timeout: int | None = None
    usage_tracking: UsageTrackingConfig = field(default_factory=UsageTrackingConfig)

    # User-driven progressive-disclosure loop
    user: BaseUser | None = None
    max_user_rounds: int = 5
    oracle_access: bool = False

    # Legacy compat fields — used by SDK.run() shim. Ignored when scenes is set.
    agent: str = "claude-agent-acp"
    prompts: list[str | None] | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    agent_env: dict[str, str] | None = None
    skills_dir: str | Path | None = None
    skill_mode: str = SKILL_MODE_NO_SKILL
    artifact_skill_mode: str | None = None
    skill_creator_dir: str | Path | None = None
    generated_skills_root: str = GENERATED_SKILLS_ROOT
    self_gen_no_internet: bool = False
    skip_verify: bool = False
    export_generated_skills_to: str | Path | None = None
    source_provenance: dict[str, Any] | None = None
    planes: RolloutPlanes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        from benchflow._utils.config import normalize_agent_idle_timeout

        if not isinstance(self.task_path, Path):
            self.task_path = Path(self.task_path)
        if self.context_root is not None and not isinstance(self.context_root, Path):
            self.context_root = Path(self.context_root)
        if self.skills_dir is not None and not isinstance(self.skills_dir, Path):
            self.skills_dir = Path(self.skills_dir)
        self.skill_mode = normalize_skill_mode(self.skill_mode)
        if self.artifact_skill_mode is not None:
            self.artifact_skill_mode = normalize_skill_mode(self.artifact_skill_mode)
        if not self.scenes:
            self.scenes = _task_document_scenes(
                self.task_path,
                prompts=self.prompts,
                skill_mode=self.skill_mode,
            )
        if self.skills_dir is not None and self.skill_mode != SKILL_MODE_WITH_SKILL:
            raise ValueError("skills_dir requires skill_mode='with-skill'")
        if not isinstance(self.jobs_dir, Path):
            self.jobs_dir = Path(self.jobs_dir)

        self.agent = normalize_agent_name(self.agent)
        self.sandbox_user = normalize_sandbox_user(self.sandbox_user)
        self.reasoning_effort = normalize_reasoning_effort(self.reasoning_effort)
        self.agent_idle_timeout = normalize_agent_idle_timeout(self.agent_idle_timeout)
        self.usage_tracking = UsageTrackingConfig.coerce(self.usage_tracking)
        for scene in self.scenes:
            for role in scene.roles:
                role.agent = normalize_agent_name(role.agent)
                role.reasoning_effort = normalize_reasoning_effort(
                    role.reasoning_effort
                )

    @classmethod
    def from_legacy(
        cls,
        *,
        task_path: Path,
        agent: str = "claude-agent-acp",
        model: str | None = None,
        reasoning_effort: str | None = None,
        prompts: list[str | None] | None = None,
        skills_dir: str | Path | None = None,
        skill_mode: str = SKILL_MODE_NO_SKILL,
        skill_creator_dir: str | Path | None = None,
        generated_skills_root: str = GENERATED_SKILLS_ROOT,
        self_gen_no_internet: bool = False,
        **kwargs,
    ) -> RolloutConfig:
        """Construct from flat SDK.run()-style args."""
        mode = normalize_skill_mode(skill_mode)
        document_scenes = _task_document_scenes(
            Path(task_path),
            prompts=prompts,
            skill_mode=mode,
        )
        if mode == SKILL_MODE_SELF_GEN:
            scenes = []
        elif document_scenes:
            scenes = document_scenes
        else:
            scenes = [
                Scene.single(
                    agent=agent,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    prompts=prompts,
                )
            ]
        return cls(
            task_path=task_path,
            scenes=scenes,
            agent=agent,
            model=model,
            reasoning_effort=reasoning_effort,
            prompts=prompts,
            skills_dir=skills_dir,
            skill_mode=mode,
            artifact_skill_mode=mode,
            skill_creator_dir=skill_creator_dir,
            generated_skills_root=generated_skills_root,
            self_gen_no_internet=self_gen_no_internet,
            **kwargs,
        )

    @property
    def effective_scenes(self) -> list[Scene]:
        """Scenes to execute — falls back to legacy fields if scenes is empty."""
        if self.scenes:
            return self.scenes
        if self.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout scenes."
            )
        if self.skill_mode not in {SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL}:
            raise ValueError(f"Unknown skill_mode: {self.skill_mode}")
        return [
            Scene.single(
                agent=self.agent,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                prompts=self.prompts,
            )
        ]

    @property
    def recorded_skill_mode(self) -> str:
        return self.artifact_skill_mode or self.skill_mode

    @property
    def primary_agent(self) -> str:
        """Agent name for the first role of the first scene."""
        scenes = self.effective_scenes
        if scenes and scenes[0].roles:
            return scenes[0].roles[0].agent
        return self.agent

    @property
    def primary_model(self) -> str | None:
        """Model for the first role of the first scene."""
        scenes = self.effective_scenes
        if scenes and scenes[0].roles:
            return scenes[0].roles[0].model
        return self.model

    @property
    def primary_reasoning_effort(self) -> str | None:
        """Reasoning effort for the first role of the first scene."""
        scenes = self.effective_scenes
        if scenes and scenes[0].roles:
            return scenes[0].roles[0].reasoning_effort
        return self.reasoning_effort


class Rollout:
    """Decomposed trial lifecycle with independently-callable phases."""

    def __init__(self, config: RolloutConfig) -> None:
        self._planes = config.planes or default_rollout_planes()
        # Activate Docker DinD compatibility shim on first rollout
        # construction (idempotent). Keeps `import benchflow.rollout`
        # side-effect free with respect to sandbox/provider behavior.
        _install_docker_compat(self._planes)

        self._config = config
        self._phase = "created"

        # Populated by setup()
        self._task: Any = None
        self._rollout_dir: Path | None = None
        self._rollout_paths: Any = None
        self._started_at: datetime | None = None
        self._job_name: str | None = None
        self._rollout_name: str | None = None
        self._agent_env: dict[str, str] = {}
        self._resolved_prompts: list[str] = []
        self._agent_launch: str = ""
        self._env: Any = None
        # When True, Rollout treats _env as caller-owned: setup() skips
        # creating a new sandbox and cleanup() skips stopping it. Set via
        # use_prebuilt_env() — see Runtime.execute() for the public path.
        self._env_externally_owned: bool = False
        self._environment: Environment | None = None
        self._timeout: int = 0
        self._timing: dict[str, float] = {}
        self._effective_locked: list[str] = []
        self._disallow_web_tools: bool = False
        self._effective_skills_dir: Path | None = None
        self._effective_skills_sandbox_dir: str | None = None
        self._task_skill_policy: TaskSkillPolicy | None = None
        self._usage_runtime: Any = None
        self._usage_metrics: dict[str, Any] = self._planes.extract_usage(None)

        # Populated by install_agent()
        self._agent_cfg: Any = None
        self._agent_cwd: str = "/app"

        # Populated by connect()
        self._acp_client: Any = None
        self._session: Any = None
        # ``_session_adapter`` carries the Agent-plane :class:`Session` contract
        # over the live ACP client (architecture.md, "The four contracts").
        # The kernel registers ``on_ask_user`` handlers through it so the live
        # ``session/request_permission`` path runs the handler instead of the
        # auto-approve fallback (#382 follow-up — instantiating the adapter
        # here is what makes the wire path honour the contract).
        self._session_adapter: Any = None
        # Sticky ``on_ask_user`` handler. Stored on the rollout (not the
        # adapter) because connect()/_reconnect_for_role() rebuild the adapter
        # each time we attach a new agent process; the handler the caller
        # registered before the first connect must follow across reconnects.
        # The "ever set" flag distinguishes "never registered" (default
        # auto-approve policy) from "explicitly cleared" (caller passed
        # ``None`` to roll back to auto-approve).
        self._ask_user_handler: Any = None
        self._ask_user_handler_set: bool = False
        self._agent_name: str = ""
        self._active_role: Role | None = None

        # Populated by execute()
        self._trajectory: list[dict] = []
        self._n_tool_calls: int = 0
        self._trajectory_source: TrajectorySource | None = None
        self._partial_trajectory: bool = False
        # Every prompt actually sent to the agent across all execute() calls —
        # this is what `n_prompts` and `prompts.json` should reflect for Scene
        # rollouts where each turn issues its own prompt. The original
        # `_resolved_prompts` is only the static base task prompt set.
        self._executed_prompts: list[str] = []

        # The tree-native execution model (architecture.md, "tree-native").
        # A linear rollout is a degree-1 tree; execute() grows it one Step at a
        # time, and branch() forks a node into N children. The tree is additive
        # — it never alters linear behaviour or output.
        self._tree: RolloutTree = RolloutTree()
        self._cursor: RolloutNode = self._tree.root

        # Populated by verify()
        self._rewards: dict | None = None
        self._verifier_error: str | None = None
        self._error: str | None = None
        # Populated by _export_generated_skills() on failure (#389 follow-up).
        # Kept separate from self._error so classify_error() does not mis-tag
        # an export-time infra failure ("connection lost") as the agent's own
        # infra_failure category in dashboards.
        self._export_error: str | None = None
        # Single bag for the four parallel diagnostic fields the old code
        # carried as separate attrs (issue #503). Each callsite that used
        # to assign to one of those slots now calls ``self._diagnostics.set(...)``
        # with a typed Diagnostic value.
        self._diagnostics: RolloutDiagnostics = RolloutDiagnostics()

        # Populated by _export_generated_skills() — the skills the agent
        # generated/evolved, captured for a continual-learning LearnerStore.
        self._evolved_skills: dict[str, str] | None = None

    @classmethod
    async def create(cls, config: RolloutConfig) -> Rollout:
        """Create a Rollout instance. Preferred over __init__ for consistency."""
        if config.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout.create()."
            )
        return cls(config)

    def use_prebuilt_env(self, inner: Any) -> None:
        """Inject a caller-owned sandbox; skip creation and teardown.

        When the public Runtime API receives a live ``Environment`` (one
        the caller already constructed, possibly started, and may want to
        reuse), the Rollout must evaluate inside that same sandbox rather
        than spinning up a second one. Call this before ``setup()``/
        ``run()``: ``setup()`` will then skip ``_create_environment`` and
        ``cleanup()`` will skip stopping the sandbox — the caller owns the
        lifecycle. Fixes #388.
        """
        if inner is None:
            raise ValueError("use_prebuilt_env() requires a non-None sandbox")
        self._env = inner
        self._env_externally_owned = True

    @property
    def env(self) -> Any:
        return self._env

    @property
    def acp_client(self) -> Any:
        return self._acp_client

    @property
    def trajectory(self) -> list[dict]:
        return self._trajectory

    @property
    def tree(self) -> RolloutTree:
        """The RolloutTree this rollout grows as it executes.

        A linear rollout is a degree-1 tree; :meth:`branch` forks a node into
        N children. ``tree.root`` is the start state s₀.
        """
        return self._tree

    @property
    def timing(self) -> dict[str, float]:
        return self._timing

    @property
    def result(self) -> RolloutResult | None:
        if self._phase not in ("verified", "cleaned"):
            return None
        return self._build_result()

    def _require_rollout_dir(self) -> Path:
        if self._rollout_dir is None:
            raise RuntimeError("Rollout.setup() must run before this phase")
        return self._rollout_dir

    def _require_started_at(self) -> datetime:
        if self._started_at is None:
            raise RuntimeError("Rollout.setup() must run before building a result")
        return self._started_at

    # ── Phase 1: SETUP (host-side, no container yet) ──

    async def setup(self) -> None:
        """Resolve config, create environment object (not yet started)."""
        cfg = self._config

        if cfg.sandbox_user is None:
            logger.warning(
                "sandbox_user=None — agent runs as root with no path lockdown."
            )
        if cfg.oracle_access and cfg.user is None:
            logger.warning(
                "oracle_access=True without a User — oracle files stay visible "
                "to the agent for the entire trial."
            )

        self._effective_locked = self._planes.resolve_locked_paths(
            cfg.sandbox_user, cfg.sandbox_locked_paths
        )

        (
            self._task,
            self._rollout_dir,
            self._rollout_paths,
            self._started_at,
            self._job_name,
            self._rollout_name,
        ) = _init_rollout(cfg.task_path, cfg.job_name, cfg.rollout_name, cfg.jobs_dir)

        from benchflow.task.runtime_capabilities import ensure_task_runtime_support

        ensure_task_runtime_support(self._task, cfg.environment, cfg.task_path)

        self._disallow_web_tools = (
            _task_disallows_internet(self._task) or cfg.self_gen_no_internet
        ) and cfg.primary_agent != "oracle"
        self._agent_env = _apply_web_policy(
            self._planes.resolve_agent_env(
                cfg.primary_agent, cfg.primary_model, cfg.agent_env
            ),
            disallow=self._disallow_web_tools,
        )
        env_config = getattr(getattr(self._task, "config", None), "environment", None)
        task_skill_policy = resolve_task_skill_policy(
            task_path=cfg.task_path,
            skill_mode=cfg.recorded_skill_mode,
            runtime_skills_dir=cfg.skills_dir,
            declared_sandbox_skills_dir=getattr(env_config, "skills_dir", None),
        )
        self._task_skill_policy = task_skill_policy
        self._resolved_prompts = _resolve_prompts(
            cfg.task_path,
            cfg.prompts,
            skills_dir=task_skill_policy.prompt_dir,
            skill_nudge=_skill_nudge(cfg.agent_env),
            agent=cfg.primary_agent,
            planes=self._planes,
        )
        self._agent_launch = self._planes.agent_launch(
            cfg.primary_agent,
            disallow_web_tools=self._disallow_web_tools,
        )

        # Copy task dir to temp when Dockerfile mutations are needed
        # (_inject_skills writes into environment/_deps/, stage_dockerfile
        # rewrites COPY paths — neither should modify the source tree)
        effective_task_path = cfg.task_path
        if cfg.context_root or task_skill_policy.needs_task_copy:
            import shutil
            import tempfile

            tmp = Path(tempfile.mkdtemp(prefix="benchflow-task-"))
            shutil.copytree(cfg.task_path, tmp / cfg.task_path.name, dirs_exist_ok=True)
            effective_task_path = tmp / cfg.task_path.name
            self._task_tmp = tmp
            if task_skill_policy.strip_bundled_dir_from_copy:
                strip_task_bundled_skills(effective_task_path)

        if cfg.context_root:
            self._planes.stage_dockerfile_deps(
                effective_task_path, Path(cfg.context_root)
            )
        effective_skills_dir = task_skill_policy.host_dir
        if (
            effective_skills_dir is not None
            and task_skill_policy.host_dir_is_bundled
            and effective_task_path != cfg.task_path
        ):
            effective_skills_dir = task_bundled_skills_dir(effective_task_path)
        if effective_skills_dir is not None and not _environment_uses_prebuilt_image(
            env_config, cfg.environment_manifest
        ):
            self._planes.inject_skills_into_dockerfile(
                effective_task_path,
                effective_skills_dir,
                sandbox_dir=task_skill_policy.sandbox_dir or "/skills",
            )

        task_skill_policy = replace(task_skill_policy, host_dir=effective_skills_dir)
        self._task_skill_policy = task_skill_policy
        self._effective_task_path = effective_task_path
        self._effective_skills_dir = effective_skills_dir
        self._effective_skills_sandbox_dir = task_skill_policy.sandbox_dir

        # Honour an externally-supplied sandbox (use_prebuilt_env, set by
        # Runtime.execute() when the caller passes a live Environment).
        # Without this guard, every Runtime.execute() would build a second
        # sandbox and silently discard the caller's prepared one — #388.
        if self._env is None:
            self._env = self._planes.create_environment(
                cfg.environment,
                self._task,
                effective_task_path,
                self._rollout_name,
                self._rollout_paths,
                preserve_agent_network=self._disallow_web_tools,
                environment_manifest=cfg.environment_manifest,
            )
        # Caller-supplied wall-clock budget (e.g. RuntimeConfig.timeout)
        # wins over the task's own default. Without this override there is
        # no way to tighten/loosen the agent budget per run — see #378.
        if cfg.timeout is not None:
            self._timeout = int(cfg.timeout)
        else:
            self._timeout = int(self._task.config.agent.timeout_sec or 0)

        _write_config(
            self._rollout_dir,
            task_path=cfg.task_path,
            agent=cfg.primary_agent,
            model=cfg.primary_model,
            reasoning_effort=cfg.primary_reasoning_effort,
            environment=cfg.environment,
            skill_policy=task_skill_policy,
            sandbox_user=cfg.sandbox_user,
            context_root=cfg.context_root,
            sandbox_locked_paths=self._effective_locked,
            sandbox_setup_timeout=cfg.sandbox_setup_timeout,
            timeout=self._timeout,
            started_at=self._started_at,
            agent_env=self._agent_env,
            usage_tracking=cfg.usage_tracking.with_env_defaults(),
            concurrency=cfg.concurrency,
            agent_idle_timeout=cfg.agent_idle_timeout,
            scenes=cfg.effective_scenes,
            source_provenance=cfg.source_provenance,
        )

        self._phase = "setup"

    # ── Phase 2: START (container comes up) ──

    async def start(self) -> None:
        """Start the environment and upload task files."""
        await _start_env_and_upload(
            self._env,
            self._config.task_path,
            self._timing,
            skip_start=self._env_externally_owned,
        )

        for hook in self._config.pre_agent_hooks or []:
            await hook(self._env)

        # Environment plane: provision the manifest-declared stateful
        # environment and gate on its readiness before the agent runs.
        if self._config.environment_manifest is not None:
            self._environment = self._planes.manifest_environment(
                self._config.environment_manifest, sandbox=self._env
            )
            await self._environment.provision(
                ctx={"task_id": self._config.task_path.name}
            )
            probe = await self._environment.readiness()
            if not probe.ready:
                raise RuntimeError(
                    f"environment plane not ready: {probe.error} "
                    f"(checked: {probe.checked})"
                )
            logger.info(
                "environment '%s' ready (%d probe(s))",
                self._config.environment_manifest.name,
                len(probe.checked),
            )

        self._phase = "started"

    # ── Phase 3: INSTALL AGENT ──

    async def install_agent(self) -> None:
        """Install the primary agent binary, set up credentials, sandbox user, skills, lockdown.

        For heterogeneous scene-authored steps (different agents per role),
        each role's agent is installed on-demand in connect_as().
        This method installs the primary agent to set up the sandbox baseline.
        """
        cfg = self._config
        rollout_dir = self._require_rollout_dir()

        cwd_result = await self._env.exec("pwd", timeout_sec=10)
        agent_cwd = (cwd_result.stdout or "").strip() or "/app"
        self._agent_cwd = agent_cwd

        if cfg.primary_agent == "oracle":
            if cfg.sandbox_user:
                await self._planes.setup_sandbox_user(
                    self._env,
                    cfg.sandbox_user,
                    workspace=self._agent_cwd,
                    timeout_sec=cfg.sandbox_setup_timeout,
                )
            await self._planes.snapshot_build_config(
                self._env, workspace=self._agent_cwd
            )
            await self._planes.seed_verifier_workspace(
                self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
            )
            await self._planes.deploy_skills(
                self._env,
                getattr(self, "_effective_task_path", cfg.task_path),
                self._effective_skills_dir,
                None,
                cfg.sandbox_user,
                self._agent_cwd,
                skills_sandbox_dir=self._effective_skills_sandbox_dir,
            )
            if cfg.export_generated_skills_to:
                await _ensure_sandbox_dir(
                    self._env, cfg.generated_skills_root, cfg.sandbox_user
                )
            await self._planes.lockdown_paths(self._env, self._effective_locked)
            self._phase = "installed"
            return

        agent_name = cfg.primary_agent
        self._agent_cfg = await self._planes.install_agent(
            self._env, agent_name, rollout_dir
        )
        if cfg.sandbox_user:
            self._agent_cwd = await self._planes.setup_sandbox_user(
                self._env,
                cfg.sandbox_user,
                workspace=self._agent_cwd,
                timeout_sec=cfg.sandbox_setup_timeout,
            )
        cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
        await self._planes.write_credential_files(
            self._env,
            agent_name,
            self._agent_env,
            self._agent_cfg,
            cfg.primary_model,
            cred_home,
        )
        if self._agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
            await self._planes.upload_subscription_auth(
                self._env, agent_name, cred_home
            )
        await self._planes.apply_web_tool_policy(
            self._env,
            agent_name,
            self._agent_cfg,
            cred_home,
            disallow=self._disallow_web_tools,
        )
        await self._planes.snapshot_build_config(self._env, workspace=self._agent_cwd)
        await self._planes.seed_verifier_workspace(
            self._env, workspace=self._agent_cwd, sandbox_user=cfg.sandbox_user
        )

        await self._planes.deploy_skills(
            self._env,
            getattr(self, "_effective_task_path", cfg.task_path),
            self._effective_skills_dir,
            self._agent_cfg,
            cfg.sandbox_user,
            self._agent_cwd,
            skills_sandbox_dir=self._effective_skills_sandbox_dir,
        )
        if cfg.export_generated_skills_to:
            await _ensure_sandbox_dir(
                self._env, cfg.generated_skills_root, cfg.sandbox_user
            )
        await self._planes.lockdown_paths(self._env, self._effective_locked)

        self._phase = "installed"

    # ── Phase 3b: CONNECT (ACP session — re-entrant) ──

    async def connect(self) -> None:
        """Open an ACP connection to the agent. Can be called multiple times."""
        cfg = self._config
        rollout_dir = self._require_rollout_dir()
        t0 = datetime.now()

        (
            self._agent_env,
            self._usage_runtime,
        ) = await self._planes.ensure_litellm_runtime(
            agent=cfg.primary_agent,
            agent_env=self._agent_env,
            model=cfg.primary_model,
            runtime=getattr(self, "_usage_runtime", None),
            environment=cfg.environment,
            session_id=getattr(self, "_rollout_name", "") or "",
            usage_tracking=cfg.usage_tracking,
            sandbox=self._env,
        )
        (
            self._acp_client,
            self._session,
            self._session_adapter,
            self._agent_name,
        ) = await self._planes.connect_acp(
            env=self._env,
            agent=cfg.primary_agent,
            agent_launch=self._agent_launch,
            agent_env=self._agent_env,
            sandbox_user=cfg.sandbox_user,
            model=cfg.primary_model,
            rollout_dir=rollout_dir,
            environment=cfg.environment,
            agent_cwd=self._agent_cwd,
            reasoning_effort=cfg.primary_reasoning_effort,
        )
        self._reapply_ask_user_handler()
        self._attach_trajectory_writer(rollout_dir)

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    def _attach_trajectory_writer(self, rollout_dir: Path) -> None:
        """Wire the current session's ``on_change`` to stream cumulative
        trajectory to ``rollout_dir/trajectory/acp_trajectory.jsonl``.

        The sink prepends ``self._trajectory`` (events from prior scenes,
        captured by value at wire-up time) so multi-scene rollouts don't
        overwrite earlier scenes' events with the current session's
        snapshot.
        """
        if self._session is None or rollout_dir is None:
            return
        prior: list[dict] = getattr(self, "_trajectory", []) or []
        traj_path = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
        self._session.on_change = make_trajectory_sink(
            TrajectoryWriter(traj_path), prior
        )

    async def disconnect(self) -> None:
        """Close the ACP client and clean up agent process, keeping the environment alive."""
        self._capture_partial_acp_trajectory()
        if self._acp_client:
            try:
                await self._acp_client.close()
            except Exception as e:
                logger.warning(f"ACP client close failed: {e}")
            self._acp_client = None
            self._session = None
            self._session_adapter = None
        # Kill any lingering agent processes to prevent context bleed between scenes
        agent_pattern = _agent_process_kill_pattern(self._agent_launch)
        if self._env and agent_pattern:
            with contextlib.suppress(Exception):
                await self._env.exec(
                    f"pkill -f {shlex.quote(agent_pattern)} || true",
                    timeout_sec=10,
                )
        self._active_role = None
        self._session_tool_count = 0
        self._session_traj_count = 0
        self._phase = "installed"

    def on_ask_user(self, handler: Any) -> None:
        """Register the agent-initiated ``session/request_permission`` handler.

        Forwards to :meth:`ACPSessionAdapter.on_ask_user` on the live adapter
        so the handler runs on the wire path; before #382's follow-up the
        adapter was never instantiated in production and the auto-approve
        policy ran unconditionally. The handler is sticky — stored on the
        rollout so reconnects (e.g. ``_reconnect_for_role``) re-register it
        on the freshly bound adapter via :meth:`_reapply_ask_user_handler`.

        Pass ``None`` to clear; the client's most-permissive auto-approve
        policy takes over (preserves the benchmark-mode default).
        """
        self._ask_user_handler = handler
        self._ask_user_handler_set = True
        self._reapply_ask_user_handler()

    def _reapply_ask_user_handler(self) -> None:
        """Re-bind any registered ``on_ask_user`` handler to the live adapter.

        ``getattr`` defaults guard ``Rollout`` instances built via
        ``__new__`` in tests that pre-date the on_ask_user field — they
        skip ``__init__`` and only set the attributes their scenarios use.
        """
        adapter = getattr(self, "_session_adapter", None)
        if adapter is None:
            return
        # No-op when the caller never touched on_ask_user — leaves the
        # default auto-approve path alone and avoids redundant client calls
        # from the connect()/_reconnect_for_role() hot paths.
        if not getattr(self, "_ask_user_handler_set", False):
            return
        handler = getattr(self, "_ask_user_handler", None)
        if handler is None:
            # Explicit clear — drop the bridge closure on the client so
            # the default most-permissive policy takes over.
            client = getattr(self, "_acp_client", None)
            if client is not None:
                client.on_ask_user(None)
            return
        adapter.on_ask_user(handler)

    def _capture_partial_acp_trajectory(self) -> None:
        """Append the live session's uncaptured tail to ``self._trajectory``.

        Runs on the disconnect / cleanup path when ``execute_prompts`` may
        have raised before the normal extend in :meth:`execute`. Uses
        ``_session_traj_count`` as the pointer to events already extended
        from this session so a partial scene's events are preserved on top
        of any prior scenes' (already-captured) events — see PR #566 review.
        """
        # Defensive lookup tolerates bare ``object()`` stubs used by older
        # rollout tests that pre-date the live-session partial-capture path.
        session = (
            getattr(self._acp_client, "session", None) if self._acp_client else None
        )
        if session is None:
            return
        try:
            captured = _capture_session_trajectory(session)
        except Exception as e:
            logger.warning(f"Partial trajectory capture failed: {e}")
            return
        delta = captured[getattr(self, "_session_traj_count", 0) :]
        if not delta:
            return
        self._trajectory.extend(delta)
        self._session_traj_count = len(captured)
        self._partial_trajectory = True
        self._trajectory_source = "partial_acp"
        prior_session_tools = getattr(self, "_session_tool_count", 0)
        new_tools = len(session.tool_calls) - prior_session_tools
        if new_tools > 0:
            self._n_tool_calls += new_tools
        self._session_tool_count = len(session.tool_calls)

    # ── Phase 3c: EXECUTE ──

    async def execute(
        self, prompts: list[str] | None = None, *, node: RolloutNode | None = None
    ) -> tuple[list[dict], int]:
        """Run prompts through the ACP session. Returns (new trajectory, new tool calls).

        execute_prompts returns cumulative session trajectory. We track
        what we've already captured to avoid duplication when the same
        session is reused across multiple turns.

        ``node`` — when given, a *pending* tree node (no incoming Step yet,
        from :meth:`RolloutTree.attach`) whose Step this call fills in place,
        instead of advancing the tree with a fresh child. The Branch engine
        passes a pre-attached branch-child node here so the child's real
        continuation Step lands on the child node itself.
        """
        effective_prompts = prompts or self._resolved_prompts
        if self._acp_client is None:
            raise RuntimeError("Rollout.connect() must run before execute()")
        prev_session_tools = getattr(self, "_session_tool_count", 0)
        t0 = datetime.now()
        active_role = getattr(self, "_active_role", None)
        timeout = (
            active_role.timeout_sec
            if active_role and active_role.timeout_sec is not None
            else self._timeout
        )
        idle_timeout = (
            active_role.idle_timeout_sec
            if active_role and active_role.idle_timeout_sec is not None
            else self._config.agent_idle_timeout
        )

        trajectory, n_tool_calls = await self._planes.execute_prompts(
            self._acp_client,
            self._session,
            effective_prompts,
            timeout,
            idle_timeout=idle_timeout,
        )

        # trajectory and n_tool_calls are cumulative for this session.
        # Compute the delta since last execute() on this session.
        new_tools = n_tool_calls - prev_session_tools
        new_events = trajectory[getattr(self, "_session_traj_count", 0) :]
        self._session_tool_count = n_tool_calls
        self._session_traj_count = len(trajectory)

        self._trajectory.extend(new_events)
        self._n_tool_calls += new_tools
        self._executed_prompts.extend(effective_prompts)
        self._trajectory_source = "acp"

        # Grow the tree at Step-level granularity — one Step per ACP event
        # (tool_call, agent_message, agent_thought, user_message). A single
        # execute() call walks the cursor down N nodes when it produced N
        # events. Closes #414: branch/process-reward/value targets the
        # individual action, not a collapsed turn.
        #
        # Empty-event executes still emit one Step so the tree advances at
        # least once per execute() call — the cursor must move, and a branch
        # child's pending node must get populated (see rollout_branch).
        steps = self._build_step_batch(new_events, new_tools)
        first_step, *rest_steps = steps
        if node is not None:
            # Fill a pre-attached pending node (a branch child) in place — the
            # child's real continuation Step lands on the child node itself.
            self._cursor = self._tree.populate(node, first_step)
        else:
            self._cursor = self._tree.advance(self._cursor, first_step)
        for step in rest_steps:
            self._cursor = self._tree.advance(self._cursor, step)

        # Accumulate execution time across all execute() calls — Scene rollouts
        # invoke execute() once per turn, and the previous "set only on first
        # call" behaviour undercounted multi-turn agent time.
        elapsed = (datetime.now() - t0).total_seconds()
        self._timing["agent_execution"] = (
            self._timing.get("agent_execution", 0.0) + elapsed
        )

        self._phase = "executed"
        return trajectory, n_tool_calls

    def _build_step_batch(self, new_events: list[dict], new_tools: int) -> list[Step]:
        """Build one Step per ACP event from the events appended this execute.

        Step-level granularity (closes #414) — each ACP event (tool_call,
        agent_message, agent_thought, user_message) becomes a Step the tree
        can address for branching, reward shaping, and value estimation.
        Empty-event executes still produce one Step so the cursor advances
        and any pending branch-child node gets populated.
        """
        base = len(self._trajectory) - len(new_events)
        if not new_events:
            return [
                Step(
                    id=f"step-{base}-empty",
                    data={"event": None, "n_tool_calls": 0},
                )
            ]
        steps: list[Step] = []
        for offset, event in enumerate(new_events):
            traj_index = base + offset
            event_type = (
                event.get("type", "event") if isinstance(event, dict) else "event"
            )
            is_tool_call = event_type == "tool_call"
            steps.append(
                Step(
                    id=f"step-{traj_index}-{event_type}",
                    data={
                        "event": event,
                        "event_type": event_type,
                        "n_tool_calls": 1 if is_tool_call else 0,
                    },
                )
            )
        # n_tool_calls across the batch should equal the new_tools reported
        # by execute_prompts. If they disagree (legacy/non-tool_call events
        # counted as tools by the agent shim) attribute the remainder to the
        # last step so the per-execute total still matches.
        batch_tools = sum(s.data["n_tool_calls"] for s in steps)
        if batch_tools != new_tools and steps:
            steps[-1].data["n_tool_calls"] += new_tools - batch_tools
        return steps

    # ── Phase 3d: BRANCH ──

    async def branch(
        self,
        n: int,
        run_child: ChildRunner | None = None,
        *,
        require_sandbox_snapshot: bool = False,
    ) -> float:
        """Branch the rollout at the cursor into ``n`` child continuations.

        Thin entry point — the Branch engine lives in
        :mod:`benchflow.rollout_branch`. It checkpoints the Environment at the
        cursor, runs each forked child as an isolated sub-rollout (its own
        scoped state, a fresh agent session), and aggregates the children's
        returns into V(parent). After this returns, the rollout's linear state
        is exactly what it was before.

        ``run_child`` is the per-child runner — injected for unit tests; the
        default restores the env, connects a fresh agent, runs the continuation,
        and scores it. A caller that needs per-child prompts binds them into the
        ``run_child`` closure.

        ``require_sandbox_snapshot`` gates the branch on the active Sandbox
        implementing container-level snapshot/restore. When True, providers
        without that capability (Modal, Daytona DinD) fail closed with a clear
        diagnostic rather than running with a half-consistent checkpoint
        (#384, Branch lifecycle in docs/architecture.md).
        """
        return await _branch_engine(
            self, n, run_child, require_sandbox_snapshot=require_sandbox_snapshot
        )

    # ── Phase 4: VERIFY ──

    async def verify(self) -> dict | None:
        """Run the verifier and return rewards."""
        cfg = self._config

        if not self._trajectory and cfg.primary_agent != "oracle":
            scraped = await _scrape_agent_trajectory(
                self._env, cfg.primary_agent, cfg.sandbox_user
            )
            if scraped:
                self._trajectory = scraped
                self._trajectory_source = "scraped"
                logger.warning(
                    f"Using scraped trajectory ({len(scraped)} events) — UNTRUSTED"
                )

        await _publish_trajectory_for_verifier(self._env, self._trajectory)

        (
            self._rewards,
            self._verifier_error,
            verifier_timeout_diag,
        ) = await _verify_rollout(
            self._env,
            self._task,
            self._rollout_paths,
            self._timing,
            self._planes,
            sandbox_user=cfg.sandbox_user,
            workspace=self._agent_cwd,
        )
        if verifier_timeout_diag is not None:
            self._diagnostics.set(verifier_timeout_diag)

        self._phase = "verified"
        return self._rewards

    async def soft_verify(self) -> tuple[dict | None, str | None, str | None]:
        """Run the verifier without full hardening — for intermediate feedback.

        Skips process kill and workspace restore/chown (so the sandbox
        stays usable for the next round), but DOES purge agent-injected
        conftest.py / sitecustomize.py / .pth files to prevent the agent
        from gaming intermediate test results.

        Returns (rewards, verifier_output, verifier_error). The final
        verify() still does full hardening.
        """
        self._rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        # Clean verifier output dir — chmod 777 so non-root verifier processes can write.
        # Keep /app present for task/verifier paths that still use the legacy
        # rootdir fallback; tasks that populate /app are unaffected.
        try:
            await self._planes.clear_verifier_output_dir(
                self._env,
                "Soft verifier setup failed: clearing verifier output directory",
                user="root",
                timeout_sec=10,
            )
            await self._planes.ensure_legacy_app_dir(
                self._env,
                "Soft verifier setup failed: preparing /app",
                user="root",
                timeout_sec=10,
            )
            # Purge agent-injected conftest/sitecustomize/.pth without
            # killing processes or restoring workspace.
            # Honor per-task [verifier.hardening] opt-outs from task config.
            await self._planes.cleanup_verifier_python_hooks(
                self._env,
                getattr(self._task, "task_dir", None),
                "Soft verifier setup failed: purging Python injection hooks",
                user="root",
                timeout_sec=10,
            )
        except Exception as e:
            verifier_error = f"soft verifier crashed: {e}"
            logger.error(verifier_error)
            return None, None, verifier_error

        rewards = None
        verifier_output = None
        verifier_error = None
        try:
            verifier = self._planes.verifier(
                task=self._task,
                rollout_paths=self._rollout_paths,
                sandbox=self._env,
            )
            verifier_result = await asyncio.wait_for(
                verifier.verify(),
                timeout=self._task.config.verifier.timeout_sec,
            )
            rewards = _ensure_canonical_rewards(verifier_result.rewards)
            # Capture raw verifier output for the user
            cat = await self._env.exec(
                "cat /logs/verifier/*.log 2>/dev/null || "
                "cat /logs/verifier/output.txt 2>/dev/null || true",
                timeout_sec=10,
            )
            verifier_output = (cat.stdout or "").strip() or None
            logger.info(f"[soft_verify] rewards={rewards}")
        except TimeoutError:
            verifier_error = (
                f"soft verifier timed out after "
                f"{self._task.config.verifier.timeout_sec}s"
            )
            logger.error(verifier_error)
        except Exception as e:
            verifier_error = f"soft verifier crashed: {e}"
            logger.error(verifier_error)
        return rewards, verifier_output, verifier_error

    # ── Phase 5: CLEANUP ──

    async def cleanup(self) -> None:
        """Close ACP client and stop the environment."""
        self._capture_partial_acp_trajectory()
        await self.disconnect()

        if self._env and self._config.export_generated_skills_to:
            try:
                await self._export_generated_skills()
            except Exception as e:
                # Surface export failure on a dedicated sibling channel
                # (#389 follow-up). Routing it through self._error caused
                # classify_error("Skill export failed: ... connection lost")
                # to mis-tag the rollout as agent infra_failure, polluting
                # the agent-error dashboards. Keep the agent/verifier error
                # channels untouched: export runs during cleanup, after the
                # agent already finished.
                export_error = f"Skill export failed: {e}"
                logger.error(export_error)
                if self._export_error is None:
                    self._export_error = export_error
                self._evolved_skills = None

        usage_runtime = getattr(self, "_usage_runtime", None)
        if usage_runtime is not None:
            try:
                await self._planes.stop_provider_runtime(usage_runtime)
                self._usage_metrics = self._planes.extract_usage(usage_runtime)
            except Exception as e:
                logger.warning(f"Usage telemetry runtime stop failed: {e}")
                self._usage_metrics = self._planes.extract_usage(None)
            try:
                self._write_llm_trajectory(usage_runtime)
            except Exception as e:
                logger.warning(f"LLM trajectory write failed: {e}")
            finally:
                self._usage_runtime = None
            self._enforce_required_usage_tracking()

        if self._environment is not None:
            with contextlib.suppress(Exception):
                await self._environment.teardown()
            self._environment = None

        if self._env and not getattr(self, "_env_externally_owned", False):
            # An externally-owned sandbox (use_prebuilt_env) belongs to the
            # caller — leave it running so they can reuse it or stop it
            # themselves. #388. getattr() keeps tests that bypass __init__
            # via Rollout.__new__() working.
            try:
                await self._env.stop(delete=True)
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")

        if hasattr(self, "_task_tmp") and self._task_tmp:
            import shutil

            shutil.rmtree(self._task_tmp, ignore_errors=True)

        self._phase = "cleaned"

    def _enforce_required_usage_tracking(self) -> None:
        usage_cfg = self._config.usage_tracking.with_env_defaults()
        if usage_cfg.mode != "required" or self._config.primary_agent == "oracle":
            return
        if self._usage_metrics.get("usage_source") == "provider_response":
            return
        if self._error is not None:
            return
        self._error = (
            "Token usage tracking is required, but no provider token usage was "
            "captured."
        )
        logger.error(self._error)

    # ── Full run ──

    async def run(self) -> RolloutResult:
        """Run the complete trial lifecycle.

        Iterates over effective_scenes. Single-agent is a trial with one
        scene containing one role — no special case.
        """
        cfg = self._config
        agent_timed_out = False
        if cfg.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use bf.run(), "
                "Evaluation.run(), or bf.run(RolloutConfig(...)) instead of Rollout.run()."
            )
        try:
            await self.setup()
            await self.start()

            if cfg.primary_agent == "oracle":
                await self.install_agent()
                # git safe.directory needed for SWE-bench tasks with sandbox_user
                import shlex

                await self._env.exec(
                    f"git config --global --add safe.directory "
                    f"{shlex.quote(self._agent_cwd)} 2>/dev/null || true",
                    user="root",
                    timeout_sec=10,
                )
                self._trajectory, self._agent_name = await _run_oracle(
                    self._env, cfg.task_path, self._timeout, sandbox_user=None
                )
            else:
                await self.install_agent()
                try:
                    try:
                        if cfg.user is not None:
                            await self._run_user_loop()
                        else:
                            await self._run_steps(
                                compile_scenes_to_steps(
                                    cfg.effective_scenes,
                                    default_prompt=(
                                        self._resolved_prompts[0]
                                        if self._resolved_prompts
                                        else None
                                    ),
                                )
                            )
                    except TimeoutError as e:
                        agent_timed_out = True
                        detail = str(e).strip()
                        self._error = (
                            detail or f"Agent timed out after {self._timeout}s"
                        )
                        self._diagnostics.capture_idle(e)
                        logger.error(self._error)
                finally:
                    if cfg.oracle_access:
                        await self._env.exec(
                            "mv /oracle_backup /oracle 2>/dev/null || true; "
                            "mv /solution_oracle_backup /solution 2>/dev/null || true",
                            user="root",
                            timeout_sec=10,
                        )

            if not cfg.skip_verify:
                await self.verify()
                if (
                    agent_timed_out
                    and self._rewards is None
                    and self._verifier_error is None
                ):
                    self._rewards = {"reward": 0.0}
                    self._verifier_error = None

        except TimeoutError as e:
            # Preserve the watchdog's diagnostic message ("Agent idle for 600s
            # with no new tool call ...") if it raised one. Fall back to the
            # generic wall-clock message only when there's no detail.
            detail = str(e).strip()
            self._error = detail or f"Agent timed out after {self._timeout}s"
            self._diagnostics.capture_idle(e)
            logger.error(self._error)
        except ConnectionError as e:
            self._error = str(e)
            self._diagnostics.capture_transport(e)
            await self._probe_sandbox_health()
            logger.error(f"Agent connection lost: {self._error}")
        except SandboxStartupFailure as e:
            self._error = f"Sandbox startup failed: {e}"
            self._diagnostics.set(e.diagnostic)
            logger.error(self._error)
        except AgentProtocolError as e:
            self._error = self._classify_acp_error(e)
            logger.error(self._error)
        except Exception as e:
            self._error = str(e)
            logger.error("Run failed", exc_info=True)
        finally:
            await self.cleanup()

        if self._rollout_dir is None:
            return RolloutResult(
                task_name=self._config.task_path.name,
                error=self._error or "Setup failed before trial directory was created",
            )
        return self._build_result()

    # ── Scene-authored Step execution ──

    async def _export_generated_skills(self) -> None:
        """Download creator-produced skills before sandbox cleanup.

        Also captures the exported skill packs into ``self._evolved_skills``
        — the ``name -> body`` dict a continual-learning Job commits to its
        persistent LearnerStore (capability 5).

        Retries transient download failures up to 3 times (guards ENG-147).
        """
        export_target = self._config.export_generated_skills_to
        if export_target is None:
            return
        target = Path(export_target)
        target.mkdir(parents=True, exist_ok=True)

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                await self._env.download_dir(self._config.generated_skills_root, target)
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        f"Skill export attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
        else:
            raise RuntimeError(
                f"Skill export failed after 3 attempts: {last_err}"
            ) from last_err

        from benchflow.learner_skills import capture_skills

        self._evolved_skills = capture_skills(target)

    async def _activate_step_skills(self, step: Step) -> None:
        """Activate scene-local skills attached by the Scene desugaring pass."""
        skills_dir = scene_step_skills_dir(step)
        if not skills_dir:
            return
        if self._env is None:
            raise RuntimeError("Environment is not started")

        scene_name = str(step.data.get("scene") or "scene")
        role = scene_step_role(step)
        source = str(skills_dir)
        local_source = Path(source).expanduser()
        # Upload whenever skills_dir resolves to a real directory on the
        # orchestrator host, regardless of whether it arrived as a str or a
        # PathLike. The SkillsBench entrypoint passes an absolute *str* host
        # path (EvaluationConfig.skills_dir is typed str), so gating the upload
        # on isinstance(..., os.PathLike) silently skipped it: the path then
        # fell through to the "already inside the sandbox" branch below and
        # _link_skill_paths produced a dangling symlink, so deployed task skills
        # never reached the agent. An absolute path that does NOT exist on the
        # host is a sandbox path produced by an earlier scene and is linked
        # as-is.
        if local_source.is_dir():
            remote_source = f"/skills/{_safe_skill_name(scene_name)}"
            await _ensure_sandbox_dir(self._env, Path(remote_source).parent)
            await self._env.upload_dir(local_source, remote_source)
        elif source.startswith("/"):
            remote_source = source
        else:
            raise FileNotFoundError(f"Scene skills_dir not found: {skills_dir}")

        home = (
            f"/home/{self._config.sandbox_user}"
            if self._config.sandbox_user
            else "/root"
        )
        agent_cfg = self._planes.agent_config(role.agent)
        if not agent_cfg or not agent_cfg.skill_paths:
            return
        await self._planes.link_skill_paths(
            self._env,
            remote_source,
            agent_cfg.skill_paths,
            home,
            self._agent_cwd,
            self._config.sandbox_user,
        )

    async def _run_steps(self, steps: list[Step]) -> None:
        """Execute already-compiled rollout Steps in declaration order."""
        current_role_key: tuple[Any, ...] | None = None
        try:
            for step in steps:
                role = scene_step_role(step)
                role_key = (
                    step.data.get("scene_index"),
                    role.name,
                    role.agent,
                    role.model,
                    role.reasoning_effort,
                    role.timeout_sec,
                    role.idle_timeout_sec,
                    tuple(sorted(role.env.items())),
                )
                logger.info(
                    "[Step] %s scene=%s role=%s",
                    step.id,
                    step.data.get("scene"),
                    role.name,
                )
                await self._activate_step_skills(step)
                if current_role_key != role_key:
                    if current_role_key is not None:
                        await self.disconnect()
                    await self.connect_as(role)
                    current_role_key = role_key
                await self.execute(prompts=[scene_step_prompt(step)])
        finally:
            if current_role_key is not None:
                await self.disconnect()

    async def _run_user_loop(self) -> None:
        """Execute a user-driven progressive-disclosure loop.

        Each round: user.run() → connect → agent.execute() → disconnect →
        soft_verify() → build RoundResult → repeat. Stops when user.run()
        returns None or max_user_rounds is reached.
        """
        cfg = self._config
        user = cfg.user
        assert user is not None

        if len(cfg.effective_scenes) > 1:
            raise ValueError(
                "User-driven loops operate on a single scene. "
                f"Got {len(cfg.effective_scenes)} scenes."
            )
        scene = cfg.effective_scenes[0]
        if len(scene.roles) != 1:
            raise ValueError(
                "User-driven loops require a single-role scene. "
                f"Got {len(scene.roles)} roles."
            )
        role = scene.roles[0]

        instruction = (
            self._resolved_prompts[0]
            if self._resolved_prompts
            else ("Solve the task described in /app/instruction.md")
        )

        # Oracle access: read /solution before the agent runs, then remove it
        solution: str | None = None
        if cfg.oracle_access:
            cat = await self._env.exec(
                "cat /oracle/solve.sh 2>/dev/null || cat /solution/solve.sh 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )
            solution = (cat.stdout or "").strip() or None

        await user.setup(instruction, solution)

        # Hide oracle files from agent — move rather than delete so the
        # final verify() can still access them if the verifier needs them.
        if cfg.oracle_access:
            await self._env.exec(
                "mv /oracle /oracle_backup 2>/dev/null || true; "
                "mv /solution /solution_oracle_backup 2>/dev/null || true",
                user="root",
                timeout_sec=10,
            )

        round_result: RoundResult | None = None
        rounds_log: list[dict] = []

        for round_num in range(cfg.max_user_rounds):
            try:
                prompt = await user.run(round_num, instruction, round_result)
            except Exception as e:
                self._error = f"user.run() failed at round {round_num}: {e}"
                logger.error(self._error, exc_info=True)
                break

            if prompt is None:
                logger.info(f"[User] stopped at round {round_num}")
                break

            logger.info(
                f"[User] round {round_num}: prompt={prompt[:80]!r}..."
                if len(prompt) > 80
                else f"[User] round {round_num}: prompt={prompt!r}"
            )

            # Fresh ACP session each round — agent starts clean but sees
            # its previous workspace changes in the shared sandbox.
            traj_before = len(self._trajectory)
            try:
                await self.connect_as(role)
                await self.execute(prompts=[prompt])
            finally:
                await self.disconnect()

            round_trajectory = self._trajectory[traj_before:]
            round_tools = sum(
                1
                for e in round_trajectory
                if isinstance(e, dict) and e.get("type") == "tool_call"
            )

            # Soft verify: run tests after agent disconnected but before
            # next round. Temporarily restore /solution so the verifier can
            # access it, then re-hide before the next agent round.
            if cfg.oracle_access:
                await self._env.exec(
                    "mv /oracle_backup /oracle 2>/dev/null || true; "
                    "mv /solution_oracle_backup /solution 2>/dev/null || true",
                    user="root",
                    timeout_sec=10,
                )
            try:
                rewards, verifier_output, verifier_error = await self.soft_verify()
            finally:
                if cfg.oracle_access:
                    await self._env.exec(
                        "mv /oracle /oracle_backup 2>/dev/null || true; "
                        "mv /solution /solution_oracle_backup 2>/dev/null || true",
                        user="root",
                        timeout_sec=10,
                    )

            round_result = RoundResult(
                round=round_num,
                trajectory=round_trajectory,
                rewards=rewards,
                verifier_output=verifier_output,
                verifier_error=verifier_error,
                n_tool_calls=round_tools,
            )

            rounds_log.append(
                {
                    "round": round_num,
                    "prompt": prompt,
                    "rewards": rewards,
                    "verifier_error": verifier_error,
                    "n_tool_calls": round_tools,
                    "n_trajectory_events": len(round_trajectory),
                }
            )

            logger.info(
                f"[User] round {round_num} done: rewards={rewards}, tools={round_tools}"
            )

        # Persist round log
        if rounds_log and self._rollout_dir:
            log_path = self._rollout_dir / "user_rounds.jsonl"
            with log_path.open("w") as f:
                for entry in rounds_log:
                    f.write(json.dumps(entry) + "\n")
            logger.info(f"[User] {len(rounds_log)} rounds → {log_path}")

    async def connect_as(self, role: Role) -> None:
        """Open an ACP connection for a specific role.

        Installs the role's agent binary and credentials if it differs
        from the primary agent (which was set up in install_agent()).
        Updates _agent_launch so disconnect() kills the correct process.
        """
        cfg = self._config
        rollout_dir = self._require_rollout_dir()
        t0 = datetime.now()

        # Merge cfg.agent_env (config-level) with role.env (role-specific) so
        # provider creds from YAML reach the agent. role.env wins on overlap.
        disallow_web_tools = getattr(self, "_disallow_web_tools", None)
        if disallow_web_tools is None:
            disallow_web_tools = _task_disallows_internet(getattr(self, "_task", None))
        disallow_web_tools = bool(disallow_web_tools and role.agent != "oracle")
        agent_launch = self._planes.agent_launch(
            role.agent,
            disallow_web_tools=disallow_web_tools,
        )
        agent_env = _apply_web_policy(
            self._planes.resolve_agent_env(
                role.agent,
                role.model,
                {**(cfg.agent_env or {}), **(role.env or {})},
            ),
            disallow=disallow_web_tools,
        )
        agent_env, self._usage_runtime = await self._planes.ensure_litellm_runtime(
            agent=role.agent,
            agent_env=agent_env,
            model=role.model,
            runtime=getattr(self, "_usage_runtime", None),
            environment=cfg.environment,
            session_id=getattr(self, "_rollout_name", "") or "",
            usage_tracking=cfg.usage_tracking,
            sandbox=self._env,
        )

        role_agent_differs = role.agent != cfg.primary_agent
        needs_role_credentials = (
            role_agent_differs or role.model != cfg.primary_model or bool(role.env)
        )
        if role_agent_differs:
            agent_cfg = await self._planes.install_agent(
                self._env, role.agent, rollout_dir
            )
        else:
            agent_cfg = getattr(self, "_agent_cfg", None)
        if needs_role_credentials:
            cred_home = f"/home/{cfg.sandbox_user}" if cfg.sandbox_user else "/root"
            await self._planes.write_credential_files(
                self._env,
                role.agent,
                agent_env,
                agent_cfg,
                role.model,
                cred_home,
            )
            if agent_env.get("_BENCHFLOW_SUBSCRIPTION_AUTH"):
                await self._planes.upload_subscription_auth(
                    self._env, role.agent, cred_home
                )
            await self._planes.apply_web_tool_policy(
                self._env,
                role.agent,
                agent_cfg,
                cred_home,
                disallow=disallow_web_tools,
            )

        self._agent_launch = agent_launch

        (
            self._acp_client,
            self._session,
            self._session_adapter,
            self._agent_name,
        ) = await self._planes.connect_acp(
            env=self._env,
            agent=role.agent,
            agent_launch=agent_launch,
            agent_env=agent_env,
            sandbox_user=cfg.sandbox_user,
            model=role.model,
            rollout_dir=rollout_dir,
            environment=cfg.environment,
            agent_cwd=self._agent_cwd,
            reasoning_effort=role.reasoning_effort,
        )
        self._reapply_ask_user_handler()
        self._attach_trajectory_writer(rollout_dir)
        self._active_role = role

        if "agent_setup" not in self._timing:
            self._timing["agent_setup"] = (datetime.now() - t0).total_seconds()

        self._phase = "connected"

    # ── Internal helpers ──

    async def _probe_sandbox_health(self) -> None:
        """Quick health probe after transport death. Enriches transport diagnostic.

        Guards ENG-148: distinguishes Daytona session killed vs agent crash.
        """
        diag = self._diagnostics.transport_closed
        if diag is None or self._env is None:
            return
        try:
            result = await asyncio.wait_for(
                self._env.exec("echo __BENCHFLOW_HEALTH_OK__", timeout_sec=10),
                timeout=15,
            )
            stdout = str(getattr(result, "stdout", "") or "").strip()
            raw_rc = getattr(result, "return_code", None)
            rc = int(raw_rc) if isinstance(raw_rc, (int, float)) else None
            if "__BENCHFLOW_HEALTH_OK__" in stdout:
                diag.sandbox_reachable = True
                diag.sandbox_probe_rc = rc
            else:
                diag.sandbox_reachable = False
                diag.sandbox_probe_rc = rc
                diag.sandbox_probe_stdout = stdout[:200]
        except Exception as probe_err:
            import traceback

            logger.exception("sandbox health probe failed")
            diag.sandbox_reachable = False
            diag.sandbox_probe_error = str(probe_err)[:200]
            diag.sandbox_probe_error_type = type(probe_err).__name__
            diag.sandbox_probe_traceback = traceback.format_exc()[-2000:]

    def _classify_acp_error(self, e: AgentProtocolError) -> str:
        if "Invalid API key" in e.message:
            from benchflow.agents.env import check_subscription_auth
            from benchflow.agents.registry import infer_env_key_for_model

            key = (
                infer_env_key_for_model(self._config.primary_model)
                if self._config.primary_model
                else None
            )
            if key and check_subscription_auth(self._config.primary_agent, key):
                return (
                    f"{key} was rejected as invalid. "
                    f"Subscription auth credentials exist — unset the env var "
                    f"to use them: env -u {key} <command>"
                )
        return str(e)

    def _write_llm_trajectory(self, usage_runtime: Any) -> None:
        """Persist captured provider HTTP exchanges as JSONL."""
        if self._rollout_dir is None:
            return
        trajectory = getattr(getattr(usage_runtime, "server", None), "trajectory", None)
        if trajectory is None or not trajectory.exchanges:
            return
        traj_dir = self._rollout_dir / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / "llm_trajectory.jsonl").write_text(
            trajectory.to_jsonl(redact_keys=True)
        )

    def _usage_tracking_metadata(self) -> dict[str, Any]:
        usage_cfg = self._config.usage_tracking.with_env_defaults()
        usage_source = str(self._usage_metrics.get("usage_source", "unavailable"))
        if usage_cfg.mode == "off":
            status = "off"
        elif usage_source == "provider_response":
            status = "enabled"
        else:
            status = "unavailable"
        return usage_cfg.to_result_metadata(
            environment=self._config.environment,
            status=status,
            usage_source=usage_source,
        )

    def _build_result(self) -> RolloutResult:
        rollout_dir = self._require_rollout_dir()
        # For Scene/multi-turn rollouts, each execute() call records the
        # prompt(s) it sent into self._executed_prompts. Use that as the
        # authoritative prompt list so n_prompts and prompts.json reflect
        # every prompt the agent actually received (issue #377). Fall back
        # to the resolved base prompts when no execute() ran (e.g. setup
        # failure paths).
        prompts = self._executed_prompts or self._resolved_prompts
        return _build_rollout_result(
            rollout_dir,
            task_name=self._config.task_path.name,
            rollout_name=self._rollout_name or "",
            agent=self._config.primary_agent,
            agent_name=self._agent_name,
            model=self._config.primary_model,
            n_tool_calls=self._n_tool_calls,
            prompts=prompts,
            error=self._error,
            verifier_error=self._verifier_error,
            export_error=self._export_error,
            trajectory=self._trajectory,
            partial_trajectory=self._partial_trajectory,
            trajectory_source=self._trajectory_source,
            rewards=self._rewards,
            started_at=self._require_started_at(),
            timing=self._timing,
            scenes=self._config.effective_scenes,
            evolved_skills=self._evolved_skills,
            source_provenance=self._config.source_provenance,
            diagnostics=self._diagnostics,
            usage_tracking=self._usage_tracking_metadata(),
            skill_policy=getattr(self, "_task_skill_policy", None)
            or resolve_task_skill_policy(
                task_path=self._config.task_path,
                skill_mode=self._config.recorded_skill_mode,
                runtime_skills_dir=self._config.skills_dir,
                declared_sandbox_skills_dir=None,
            ),
            **self._usage_metrics,
        )
