"""Canonical rollout configuration types.

These types are intentionally small, serializable dataclasses. They are the
single source of truth for v0.4 scene configuration while the execution
lifecycle is migrated from ``Trial`` to ``Rollout``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Role:
    """One participant in a rollout scene.

    Runtime budgets belong on roles, not in the agent registry. ``None`` means
    inherit from the surrounding rollout/task defaults.
    """

    name: str
    agent: str
    model: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_sec: int | None = None
    idle_timeout_sec: int | None = None
    skills_dir: str | Path | None = None
    capabilities: list[str] = field(default_factory=list)
    instruction: str | None = None
    tools: list[str] = field(default_factory=list)


@dataclass
class Turn:
    """One role action opportunity within a scene."""

    role: str
    prompt: str | None = None  # None = expand from task instruction.md


@dataclass
class Scene:
    """A flat rollout phase containing roles and ordered turns."""

    name: str = "default"
    roles: list[Role] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    skills_dir: str | Path | None = None
    parallel_group: str | None = None

    @classmethod
    def single(
        cls,
        *,
        agent: str,
        model: str | None = None,
        prompts: list[str | None] | None = None,
        role_name: str = "agent",
        skills_dir: str | Path | None = None,
        timeout_sec: int | None = None,
        idle_timeout_sec: int | None = None,
    ) -> Scene:
        """Shortcut for a single-role scene using task instructions by default."""

        prompts = prompts or [None]
        return cls(
            roles=[
                Role(
                    name=role_name,
                    agent=agent,
                    model=model,
                    timeout_sec=timeout_sec,
                    idle_timeout_sec=idle_timeout_sec,
                    skills_dir=skills_dir,
                )
            ],
            turns=[Turn(role=role_name, prompt=p) for p in prompts],
            skills_dir=skills_dir,
        )
