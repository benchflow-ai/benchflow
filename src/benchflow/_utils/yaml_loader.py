"""YAML rollout config loader.

Parses rollout YAML files into RolloutConfig with Scene support.
Handles both new scene-based format and legacy flat format.

New format::

    task_dir: tasks/
    environment: daytona
    concurrency: 64

    scenes:
      - name: skill-gen
        roles:
          - name: creator
            agent: gemini
            model: gemini-3.1-flash-lite-preview
        turns:
          - role: creator
            prompt: "Generate a skill for this task..."
      - name: solve
        roles:
          - name: solver
            agent: gemini
            model: gemini-3.1-flash-lite-preview
        turns:
          - role: solver

Legacy format (auto-converted)::

    task_dir: tasks/
    agent: gemini
    model: gemini-3.1-flash-lite-preview
    environment: daytona
    concurrency: 64
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from benchflow._types import Role, Scene, Turn
from benchflow.review.options import ReviewerConfig
from benchflow.rollout import RolloutConfig
from benchflow.skill_policy import SKILL_MODE_NO_SKILL
from benchflow.usage_tracking import UsageTrackingConfig

logger = logging.getLogger(__name__)


def load_rollout_yaml(path: str | Path) -> dict:
    """Load and normalize a rollout YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict at top level, got {type(raw).__name__}")
    return raw


def rollout_config_from_yaml(
    path: str | Path,
    task_path: Path | None = None,
) -> RolloutConfig:
    """Parse a YAML file into a RolloutConfig.

    If task_path is provided, it overrides task_dir from the YAML.
    """
    raw = load_rollout_yaml(path)
    return rollout_config_from_dict(raw, task_path=task_path)


def rollout_config_from_dict(
    raw: dict[str, Any],
    task_path: Path | None = None,
) -> RolloutConfig:
    """Convert a raw dict (from YAML or programmatic) into a RolloutConfig."""
    tp = task_path or Path(raw.get("task_dir", raw.get("task_path", ".")))

    # Scene-based format
    if "scenes" in raw:
        scenes = [_parse_scene(s) for s in raw["scenes"]]
    elif "agent" in raw and "prompts" in raw and raw["prompts"] is None:
        # ``prompts: null`` (as RolloutConfig.to_dict writes it): the task's
        # own prompts, derived by RolloutConfig from the task document.
        scenes = []
    elif "agent" in raw:
        # Legacy flat format
        prompts_raw = raw.get("prompts")
        prompts: list[str | None]
        if isinstance(prompts_raw, list):
            prompts = []
            for prompt in prompts_raw:
                if prompt is not None and not isinstance(prompt, str):
                    raise ValueError("YAML prompts entries must be strings or null")
                prompts.append(prompt)
        elif isinstance(prompts_raw, str):
            prompts = [prompts_raw]
        else:
            prompts = [None]
        scenes = [
            Scene.single(
                agent=raw["agent"],
                model=raw.get("model"),
                reasoning_effort=raw.get("reasoning_effort"),
                prompts=prompts,
            )
        ]
    else:
        raise ValueError("YAML must have either 'scenes' or 'agent' at top level")

    return RolloutConfig(
        task_path=tp,
        reviewer=ReviewerConfig.coerce(raw.get("reviewer")),
        scenes=scenes,
        environment=raw.get("environment", "docker"),
        codex_apps_policy=raw.get("codex_apps_policy"),
        sandbox_user=raw.get("sandbox_user", "agent"),
        sandbox_locked_paths=raw.get("sandbox_locked_paths"),
        sandbox_setup_timeout=raw.get("sandbox_setup_timeout", 120),
        job_name=raw.get("job_name"),
        rollout_name=raw.get("rollout_name"),
        jobs_dir=raw.get("jobs_dir", "jobs"),
        concurrency=raw.get("concurrency", 1),
        agent_idle_timeout=raw.get(
            "agent_idle_timeout_sec", raw.get("agent_idle_timeout", 600)
        ),
        harness=raw.get("harness", "acp"),
        context_root=raw.get("context_root"),
        base_image_override=raw.get("base_image_override"),
        agent=raw.get("agent", "claude-agent-acp"),
        model=raw.get("model"),
        reasoning_effort=raw.get("reasoning_effort"),
        agent_env=raw.get("agent_env"),
        skills_dir=raw.get("skills_dir"),
        skill_mode=raw.get("skill_mode", SKILL_MODE_NO_SKILL),
        skill_creator_dir=raw.get("skill_creator_dir"),
        self_gen_no_internet=bool(raw.get("self_gen_no_internet", False)),
        usage_tracking=UsageTrackingConfig.from_mapping(raw),
        **_optional_fields(raw),
    )


def _optional_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Fields RolloutConfig.to_dict writes beyond the original YAML schema."""
    from benchflow.checkpoints import parse_checkpoint_policy
    from benchflow.environment.manifest import EnvironmentManifest, load_manifest

    out: dict[str, Any] = {}
    for key in (
        "timeout",
        "services",
        "config_override",
        "max_user_rounds",
        "oracle_access",
        "loop_strategy",
        "uploads",
        "skip_verify",
        "freeze_workspace",
        "skip_agent_install",
        "source_provenance",
    ):
        if raw.get(key) is not None:
            out[key] = raw[key]
    if isinstance(raw.get("prompts"), list):
        out["prompts"] = raw["prompts"]
    manifest = raw.get("environment_manifest")
    if isinstance(manifest, dict):
        out["environment_manifest"] = EnvironmentManifest.model_validate(manifest)
    elif manifest is not None:
        out["environment_manifest"] = load_manifest(manifest)
    if raw.get("checkpoints"):
        out["checkpoints"] = parse_checkpoint_policy(
            str(raw["checkpoints"]), keep=int(raw.get("checkpoint_keep") or 3)
        )
    return out


def _checkpoint_spec(policy: Any) -> str | None:
    if policy is None:
        return None
    if policy.every:
        return "every-prompt"
    return "prompt:" + ",".join(str(n) for n in sorted(policy.after))


def rollout_config_to_dict(
    config: RolloutConfig, *, include_agent_env: bool = False
) -> dict[str, Any]:
    """``config`` in the shape :func:`rollout_config_from_dict` reads.

    A single-agent config is written flat (``agent``, ``model``, ``prompts``;
    ``prompts: null`` keeps the task's own prompts); explicit multi-role scenes
    are written as ``scenes``. agent_env values are written only with
    ``include_agent_env=True`` (their names are listed under
    ``agent_env_keys``). Raises ``ValueError`` for fields that hold Python
    objects (``user``, ``pre_agent_hooks``, ``planes``).
    """
    import json

    from benchflow.loop_strategies import LoopStrategySpec

    blocked = [
        name
        for name in ("user", "pre_agent_hooks", "planes")
        if getattr(config, name) not in (None, [])
    ]
    if blocked:
        raise ValueError(
            f"RolloutConfig fields {', '.join(blocked)} hold Python objects and "
            "cannot be written to YAML; set them in code after from_yaml()."
        )
    agent_env = dict(config.agent_env or {})
    single = len(config.scenes) <= 1 and all(
        len(scene.roles) == 1
        and scene.roles[0].agent == config.agent
        and scene.roles[0].model == config.model
        for scene in config.scenes
    )
    out: dict[str, Any] = {
        "task_path": str(config.task_path),
        "agent": config.agent,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
    }
    if single:
        out["prompts"] = config.prompts
    else:
        out["scenes"] = [
            {
                "name": scene.name,
                "roles": [
                    {
                        "name": role.name,
                        "agent": role.agent,
                        "model": role.model,
                        "reasoning_effort": role.reasoning_effort,
                        **(
                            {"env": dict(role.env)}
                            if role.env and include_agent_env
                            else {}
                        ),
                    }
                    for role in scene.roles
                ],
                "turns": [{"role": t.role, "prompt": t.prompt} for t in scene.turns],
            }
            for scene in config.scenes
        ]
    reviewer = config.reviewer.to_dict()
    if not include_agent_env:
        reviewer["agent_env"] = {}
    loop = config.loop_strategy
    out.update(
        {
            "environment": config.environment,
            "codex_apps_policy": config.codex_apps_policy,
            "sandbox_user": config.sandbox_user,
            "sandbox_locked_paths": config.sandbox_locked_paths,
            "sandbox_setup_timeout": config.sandbox_setup_timeout,
            "job_name": config.job_name,
            "rollout_name": config.rollout_name,
            "jobs_dir": str(config.jobs_dir),
            "concurrency": config.concurrency,
            "agent_idle_timeout_sec": config.agent_idle_timeout,
            "harness": config.harness,
            "timeout": config.timeout,
            "context_root": str(config.context_root) if config.context_root else None,
            "base_image_override": config.base_image_override,
            "agent_env": agent_env if include_agent_env else None,
            "agent_env_keys": sorted(agent_env),
            "skills_dir": str(config.skills_dir) if config.skills_dir else None,
            "skill_mode": config.skill_mode,
            "skill_creator_dir": str(config.skill_creator_dir)
            if config.skill_creator_dir
            else None,
            "self_gen_no_internet": config.self_gen_no_internet,
            "reviewer": reviewer,
            **config.usage_tracking.to_mapping(),
            "services": config.services,
            "config_override": config.config_override,
            "max_user_rounds": config.max_user_rounds,
            "oracle_access": config.oracle_access,
            "loop_strategy": loop.to_mapping()
            if isinstance(loop, LoopStrategySpec)
            else loop,
            "uploads": dict(config.uploads) or None,
            "skip_verify": config.skip_verify,
            "freeze_workspace": config.freeze_workspace,
            "skip_agent_install": config.skip_agent_install,
            "environment_manifest": config.environment_manifest.model_dump(mode="json")
            if config.environment_manifest is not None
            else None,
            "checkpoints": _checkpoint_spec(config.checkpoints),
            "checkpoint_keep": config.checkpoints.keep
            if config.checkpoints is not None
            else None,
            "source_provenance": config.source_provenance,
        }
    )
    return json.loads(json.dumps(out, default=str))


def _parse_scene(raw: dict) -> Scene:
    """Parse a scene dict from YAML."""
    roles = [_parse_role(r) for r in raw.get("roles", [])]
    turns = [_parse_turn(t) for t in raw.get("turns", [])]

    # If no turns specified but roles exist, create one turn per role
    if not turns and roles:
        turns = [Turn(role=r.name) for r in roles]

    return Scene(
        name=raw.get("name", "default"),
        roles=roles,
        turns=turns,
        skills_dir=raw.get("skills_dir"),
    )


def _parse_role(raw: dict) -> Role:
    """Parse a role dict from YAML."""
    return Role(
        name=raw["name"],
        agent=raw["agent"],
        model=raw.get("model"),
        reasoning_effort=raw.get("reasoning_effort"),
        env=raw.get("env", {}),
    )


def _parse_turn(raw: dict) -> Turn:
    """Parse a turn dict from YAML."""
    return Turn(
        role=raw["role"],
        prompt=raw.get("prompt"),
    )
