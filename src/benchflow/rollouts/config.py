"""Canonical rollout configuration types.

These types are intentionally small, serializable dataclasses. They are the
single source of truth for v0.4 scene configuration while the execution
lifecycle is migrated from ``Trial`` to ``Rollout``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from benchflow.user import BaseUser

SKILL_MODE_DEFAULT = "default"
SKILL_MODE_SELF_GEN = "self-gen"
GENERATED_SKILLS_ROOT = "/app/generated-skills"


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


@dataclass
class RolloutConfig:
    """Declarative config for one task attempt in one sandbox.

    During the v0.4 migration this preserves the current TrialConfig fields so
    execution can move independently from public naming.
    """

    task_path: Path
    scenes: list[Scene] = field(default_factory=list)
    environment: str = "docker"
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    sandbox_setup_timeout: int = 120
    services: list[str] | None = None
    job_name: str | None = None
    trial_name: str | None = None
    jobs_dir: str | Path = "jobs"
    context_root: str | Path | None = None
    pre_agent_hooks: list | None = None
    agent_idle_timeout: int | None = 600

    # User-driven progressive-disclosure loop
    user: BaseUser | None = None
    max_user_rounds: int = 5
    oracle_access: bool = False

    # Legacy flat single-agent fields. Ignored when scenes is set.
    agent: str = "claude-agent-acp"
    prompts: list[str | None] | None = None
    model: str | None = None
    agent_env: dict[str, str] | None = None
    skills_dir: str | Path | None = None
    skill_mode: str = SKILL_MODE_DEFAULT
    skill_creator_dir: str | Path | None = None
    generated_skills_root: str = GENERATED_SKILLS_ROOT
    self_gen_no_internet: bool = False
    include_task_skills: bool = True
    skip_verify: bool = False
    export_generated_skills_to: str | Path | None = None

    @classmethod
    def from_single(
        cls,
        *,
        task_path: Path,
        agent: str = "claude-agent-acp",
        model: str | None = None,
        prompts: list[str | None] | None = None,
        skills_dir: str | Path | None = None,
        skill_mode: str = SKILL_MODE_DEFAULT,
        skill_creator_dir: str | Path | None = None,
        generated_skills_root: str = GENERATED_SKILLS_ROOT,
        self_gen_no_internet: bool = False,
        **kwargs,
    ) -> RolloutConfig:
        """Construct from flat single-agent arguments."""

        scenes = []
        if skill_mode not in {SKILL_MODE_DEFAULT, SKILL_MODE_SELF_GEN}:
            raise ValueError(f"Unknown skill_mode: {skill_mode}")
        if skill_mode == SKILL_MODE_DEFAULT:
            scenes = [
                Scene.single(
                    agent=agent,
                    model=model,
                    prompts=prompts,
                    skills_dir=skills_dir,
                )
            ]
        return cls(
            task_path=task_path,
            scenes=scenes,
            agent=agent,
            model=model,
            prompts=prompts,
            skills_dir=skills_dir,
            skill_mode=skill_mode,
            skill_creator_dir=skill_creator_dir,
            generated_skills_root=generated_skills_root,
            self_gen_no_internet=self_gen_no_internet,
            **kwargs,
        )

    @classmethod
    def from_legacy(cls, **kwargs) -> RolloutConfig:
        """Compatibility constructor while callsites migrate to from_single()."""

        return cls.from_single(**kwargs)

    @property
    def effective_scenes(self) -> list[Scene]:
        """Scenes to execute, falling back to flat single-agent fields."""

        if self.skill_mode == SKILL_MODE_SELF_GEN:
            raise ValueError(
                "self-gen requires the runtime orchestrator. Use the high-level "
                "run path instead of direct rollout scenes."
            )
        if self.scenes:
            return self.scenes
        if self.skill_mode != SKILL_MODE_DEFAULT:
            raise ValueError(f"Unknown skill_mode: {self.skill_mode}")
        return [
            Scene.single(
                agent=self.agent,
                model=self.model,
                prompts=self.prompts,
                skills_dir=self.skills_dir,
            )
        ]

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
