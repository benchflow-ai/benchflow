"""Trial config dataclasses — Role, Turn, Scene, TrialConfig.

CONTRACT SURFACE — semver-stable. Changes here break downstream importers.
Prefer extending in periphery (``benchflow.trial``) unless the shape itself
must change.

Declarative inputs to ``benchflow.trial.Trial``. Kept here (``contracts/``) so the
orchestrator in ``trial.py`` can evolve without dragging semver guarantees
into every lifecycle refactor. ``trial.py`` re-exports these names for
backward compatibility with the 71 external import sites.
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


@dataclass
class Turn:
    """One prompt in a scene. role selects which Role acts."""

    role: str
    prompt: str | None = None  # None = expand from instruction.md


@dataclass
class Scene:
    """One interaction region — roles take turns executing prompts.

    Single-role multi-turn execution is supported by ``Trial._run_scene``
    (same ACP session preserved across turns). Multi-role execution is
    NOT wired in graduated code: ``Trial._run_scene`` rejects scenes
    with ``len(roles) > 1`` because there is no inter-role coordination
    layer (no inbox, no message routing, no shared prompt state). For
    multi-agent runs use ``benchflow.experimental.mailbox.MailboxRunner``
    directly — it is a separate API and does not consume ``Scene``.
    """

    name: str = "default"
    roles: list[Role] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)

    @classmethod
    def single(
        cls,
        *,
        agent: str,
        model: str | None = None,
        prompts: list[str | None] | None = None,
        role_name: str = "agent",
    ) -> "Scene":
        """Shortcut for single-agent, single-role scene."""
        prompts = prompts or [None]
        return cls(
            roles=[Role(name=role_name, agent=agent, model=model)],
            turns=[Turn(role=role_name, prompt=p) for p in prompts],
        )


@dataclass
class TrialConfig:
    """Declarative trial configuration.

    A trial is a sequence of scenes executed in a shared sandbox.
    Single-agent runs are a trial with one scene containing one role.
    """

    task_path: Path
    scenes: list[Scene] = field(default_factory=list)
    environment: str = "docker"
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    services: list[str] | None = None
    job_name: str | None = None
    trial_name: str | None = None
    jobs_dir: str | Path = "jobs"
    context_root: str | Path | None = None
    pre_agent_hooks: list | None = None

    # Legacy compat fields — used by SDK.run() shim. Ignored when scenes is set.
    agent: str = "claude-agent-acp"
    prompts: list[str | None] | None = None
    model: str | None = None
    agent_env: dict[str, str] | None = None
    skills_dir: str | Path | None = None

    @classmethod
    def from_legacy(
        cls,
        *,
        task_path: Path,
        agent: str = "claude-agent-acp",
        model: str | None = None,
        prompts: list[str | None] | None = None,
        skills_dir: str | Path | None = None,
        **kwargs,
    ) -> "TrialConfig":
        """Construct from flat SDK.run()-style args."""
        return cls(
            task_path=task_path,
            scenes=[
                Scene.single(agent=agent, model=model, prompts=prompts)
            ],
            agent=agent,
            model=model,
            prompts=prompts,
            skills_dir=skills_dir,
            **kwargs,
        )

    @property
    def effective_scenes(self) -> list[Scene]:
        """Scenes to execute — falls back to legacy fields if scenes is empty."""
        if self.scenes:
            return self.scenes
        return [
            Scene.single(
                agent=self.agent,
                model=self.model,
                prompts=self.prompts,
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
