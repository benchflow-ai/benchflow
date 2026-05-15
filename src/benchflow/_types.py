"""Canonical Scene / Role / Turn data types for benchflow trials.

These are the *declarative* types — they describe what a trial *will* do.
Runtime classes (e.g. ``_scene.Scene``) consume these but are not defined
here.

Merged from the duplicate definitions that lived in ``trial.py`` and
``_scene.py`` prior to ENG-47.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Role:
    """One agent participant in a scene."""

    name: str
    agent: str
    model: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_sec: int | None = None  # None = inherit from task.toml
    idle_timeout_sec: int | None = None
    skills_dir: str | Path | None = None


@dataclass
class Turn:
    """One prompt in a scene. *role* selects which Role acts."""

    role: str
    prompt: str | None = None  # None = expand from instruction.md


@dataclass
class Scene:
    """One interaction region — roles take turns executing prompts."""

    name: str = "default"
    roles: list[Role] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    skills_dir: str | Path | None = None
    parallel_group: str | None = None  # scenes with same group execute concurrently

    @classmethod
    def single(
        cls,
        *,
        agent: str,
        model: str | None = None,
        prompts: list[str | None] | None = None,
        role_name: str = "agent",
        skills_dir: str | Path | None = None,
    ) -> Scene:
        """Shortcut for single-agent, single-role scene."""
        prompts = prompts or [None]
        return cls(
            roles=[Role(name=role_name, agent=agent, model=model)],
            turns=[Turn(role=role_name, prompt=p) for p in prompts],
            skills_dir=skills_dir,
        )
