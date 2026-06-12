"""``RolloutConfig`` — the declarative rollout configuration dataclass.

Carries both the scene-native configuration and the legacy flat fields that
``SDK.run()`` depends on (the back-compat contract in ``from_legacy`` and
``__post_init__``). The flat fields and ``from_legacy`` are intentionally left
unchanged: they are the SDK back-compat surface (BR6).

Split into its own module from ``rollout.py`` for cohesion; re-exported from
:mod:`benchflow.rollout` so ``from benchflow.rollout import RolloutConfig``
keeps resolving unchanged. The two ``task.md`` document compilers it depends on
travel with it here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchflow._types import Role, Scene
from benchflow._utils.config import (
    normalize_agent_name,
    normalize_reasoning_effort,
    normalize_sandbox_user,
)
from benchflow.contracts import BaseUser, RolloutPlanes
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.skill_policy import (
    SKILL_MODE_NO_SKILL,
    SKILL_MODE_SELF_GEN,
    SKILL_MODE_WITH_SKILL,
    normalize_skill_mode,
)
from benchflow.usage_tracking import UsageTrackingConfig

logger = logging.getLogger(__name__)

GENERATED_SKILLS_ROOT = "/app/generated-skills"


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
    from benchflow.task.prompts import materialize_prompt_plan_scenes

    document = TaskDocument.from_path(document_path)
    return materialize_prompt_plan_scenes(document)


def _task_document_user_runtime(
    task_path: Path,
    *,
    prompts: list[str | None] | None,
    skill_mode: str,
) -> tuple[BaseUser, int | None] | None:
    """Compile a supported document-declared user into runtime configuration."""

    if prompts is not None or skill_mode == SKILL_MODE_SELF_GEN:
        return None
    document_path = task_path / "task.md"
    if not document_path.exists():
        return None

    from benchflow.task.document import TaskDocument
    from benchflow.task.prompts import compile_document_user_runtime

    runtime = compile_document_user_runtime(TaskDocument.from_path(document_path))
    if runtime.user is None:
        return None
    return runtime.user, runtime.max_rounds


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
    # Registry dataset identity ({"name","version","task_digest"}) for the
    # `bench eval create -d` path; stamped into config.json/result.json.
    dataset: dict[str, Any] | None = None
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
        explicit_scenes = bool(self.scenes)
        if not explicit_scenes:
            self.scenes = _task_document_scenes(
                self.task_path,
                prompts=self.prompts,
                skill_mode=self.skill_mode,
            )
        if self.user is None and not explicit_scenes:
            document_user = _task_document_user_runtime(
                self.task_path,
                prompts=self.prompts,
                skill_mode=self.skill_mode,
            )
            if document_user is not None:
                self.user, max_rounds = document_user
                if max_rounds is not None:
                    self.max_user_rounds = max_rounds
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
            if agent == "oracle":
                # The task.md pins its own scene agents (document_scenes), which
                # become the primary agent — so an explicit `--agent oracle`
                # is silently dropped and oracle mode never engages. Warn loudly
                # rather than running the role agent under the guise of oracle.
                logger.warning(
                    "--agent oracle was requested, but %s pins scene agents "
                    "(agents.roles / document scenes), which take precedence. "
                    "Oracle mode will NOT run; the task's scene agent(s) will "
                    "execute instead.",
                    Path(task_path).name,
                )
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
        if "user" not in kwargs:
            document_user = _task_document_user_runtime(
                Path(task_path),
                prompts=prompts,
                skill_mode=mode,
            )
            if document_user is not None:
                kwargs["user"], max_rounds = document_user
                if max_rounds is not None and "max_user_rounds" not in kwargs:
                    kwargs["max_user_rounds"] = max_rounds
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
    def _primary_role(self) -> Role | None:
        """First role of the first scene, if any."""
        scenes = self.effective_scenes
        if scenes and scenes[0].roles:
            return scenes[0].roles[0]
        return None

    @property
    def primary_agent(self) -> str:
        """Agent name for the first role of the first scene."""
        role = self._primary_role
        return role.agent if role else self.agent

    @property
    def primary_model(self) -> str | None:
        """Model for the first role of the first scene."""
        role = self._primary_role
        return role.model if role else self.model

    @property
    def primary_reasoning_effort(self) -> str | None:
        """Reasoning effort for the first role of the first scene."""
        role = self._primary_role
        return role.reasoning_effort if role else self.reasoning_effort
