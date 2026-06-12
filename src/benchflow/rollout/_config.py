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
from benchflow.loop_strategies import (
    LoopStrategySpec,
    build_loop_user,
    parse_loop_strategy_spec,
)
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
    # Harness-level loop strategy (``--loop-strategy``). Accepts a spec string
    # ("verify-retry:k=3,feedback=names") or a parsed LoopStrategySpec;
    # __post_init__ materializes it into user/max_user_rounds. Mutually
    # exclusive with an explicit ``user``.
    loop_strategy: LoopStrategySpec | str | None = None
    # Whether a task.md-declared user may be adopted when neither an explicit
    # ``user`` nor a loop strategy materializes one. ``None`` (default)
    # derives the answer from scene authorship: programmatic scene callers
    # opt out of document user loops. ``from_legacy`` passes True — its
    # scenes come from the task document / flat args, not the caller, so the
    # document user keeps engaging on that path.
    allow_document_user: bool | None = None

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
    # Registry dataset identity: {"name", "version"} — stamped into
    # config.json/result.json as dataset_name/dataset_version so published
    # runs declare the dataset version they ran (dataset-versioning plan).
    dataset: dict[str, Any] | None = None
    # Content digest of the task directory ("sha256:<hex>", the skillsbench
    # registry algorithm). Registry-pinned for dataset runs, computed live
    # for dev runs — every result stays attributable to exact task content.
    task_digest: str | None = None
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
        if self.allow_document_user is None:
            self.allow_document_user = not explicit_scenes
        if isinstance(self.loop_strategy, str):
            self.loop_strategy = parse_loop_strategy_spec(self.loop_strategy)
        self._resolve_user()
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

    def _resolve_user(self) -> None:
        """The single user-materialization point for every construction path.

        One decision ladder — explicit user → strategy user → task-document
        user → none — applied identically for direct construction and
        ``from_legacy``, so single-shot semantics cannot fork between them:
        an explicit ``loop_strategy="single-shot"`` forces a true single-shot
        run and suppresses a task-document-declared user (with a warning)
        instead of keeping it on one path and silently dropping it on the
        other. ``_task_document_user_runtime`` gates itself (no task.md,
        explicit prompts, or self-gen mode resolve to no document user), and
        ``allow_document_user`` keeps programmatic scene callers opted out.
        """
        spec = self.loop_strategy_spec
        built = build_loop_user(spec) if spec is not None else None
        if self.user is not None:
            if built is not None:
                raise ValueError(
                    "loop_strategy and an explicit user are mutually "
                    "exclusive — the strategy materializes its own user"
                )
            return
        document_user = (
            _task_document_user_runtime(
                self.task_path,
                prompts=self.prompts,
                skill_mode=self.skill_mode,
            )
            if self.allow_document_user
            else None
        )
        if spec is not None and built is not None:
            if document_user is not None:
                logger.warning(
                    "loop_strategy=%s overrides the user declared by %s/task.md",
                    spec.name,
                    self.task_path.name,
                )
            self.user, self.max_user_rounds = built
            # Fail at construction — before any sandbox is provisioned —
            # when the task declares more scene turns than the strategy's
            # round budget (the engine enforces the same bound at run time).
            declared_turns = sum(len(scene.turns) for scene in self.scenes)
            if declared_turns > self.max_user_rounds:
                raise ValueError(
                    f"loop_strategy {spec.name!r} allows max_user_rounds="
                    f"{self.max_user_rounds} but the task declares "
                    f"{declared_turns} scene turns — raise k or drop the "
                    "loop strategy"
                )
            return
        if spec is not None:
            # Explicit single-shot: a true one-prompt run, even when the
            # task document declares a user.
            if document_user is not None:
                logger.warning(
                    "loop_strategy=single-shot suppresses the user declared "
                    "by %s/task.md",
                    self.task_path.name,
                )
            return
        if document_user is not None:
            self.user, max_rounds = document_user
            if max_rounds is not None:
                self.max_user_rounds = max_rounds

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
        # User materialization — including the task-document user fallback —
        # is owned entirely by __post_init__._resolve_user, so this path and
        # direct construction cannot diverge (single-shot semantics fork).
        # The scenes built above are derived, not caller-authored, so the
        # document user stays eligible.
        kwargs.setdefault("allow_document_user", True)
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
    def loop_strategy_spec(self) -> LoopStrategySpec | None:
        """The parsed loop strategy — spec strings are parsed in __post_init__."""
        spec = self.loop_strategy
        return spec if isinstance(spec, LoopStrategySpec) else None

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
