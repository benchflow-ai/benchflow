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
from benchflow.rollout import RolloutConfig

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
                prompts=prompts,
                skills_dir=raw.get("skills_dir"),
            )
        ]
    else:
        raise ValueError("YAML must have either 'scenes' or 'agent' at top level")

    return RolloutConfig(
        task_path=tp,
        scenes=scenes,
        environment=raw.get("environment", "docker"),
        sandbox_user=raw.get("sandbox_user", "agent"),
        sandbox_locked_paths=raw.get("sandbox_locked_paths"),
        sandbox_setup_timeout=raw.get("sandbox_setup_timeout", 120),
        job_name=raw.get("job_name"),
        rollout_name=raw.get("rollout_name"),
        jobs_dir=raw.get("jobs_dir", "jobs"),
        context_root=raw.get("context_root"),
        agent=raw.get("agent", "claude-agent-acp"),
        model=raw.get("model"),
        agent_env=raw.get("agent_env"),
        skills_dir=raw.get("skills_dir"),
        skill_mode=raw.get("skill_mode", "default"),
        skill_creator_dir=raw.get("skill_creator_dir"),
        self_gen_no_internet=bool(raw.get("self_gen_no_internet", False)),
    )


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
        env=raw.get("env", {}),
        timeout_sec=raw.get("timeout_sec"),
        idle_timeout_sec=raw.get("idle_timeout_sec"),
        skills_dir=raw.get("skills_dir"),
        capabilities=raw.get("capabilities"),
        transport=raw.get("transport", "acp"),
        endpoint_url=raw.get("endpoint_url"),
    )


def _parse_turn(raw: dict) -> Turn:
    """Parse a turn dict from YAML."""
    return Turn(
        role=raw["role"],
        prompt=raw.get("prompt"),
    )


def job_config_from_yaml(path: str | Path) -> dict:
    """Parse a YAML file and return both job-level and trial-level config.

    Returns a dict with keys: task_dir, concurrency, max_retries,
    trial_config (RolloutConfig), and any other job-level fields.
    """
    raw = load_rollout_yaml(path)
    task_dir = Path(raw.get("task_dir", raw.get("tasks_dir", ".")))
    concurrency = raw.get("concurrency", 4)
    max_retries = raw.get("max_retries", 2)

    return {
        "task_dir": task_dir,
        "concurrency": concurrency,
        "max_retries": max_retries,
        "trial_config": rollout_config_from_dict(raw, task_path=task_dir),
        "raw": raw,
    }
