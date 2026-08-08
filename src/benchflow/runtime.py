"""BenchFlow Runtime — the execution center.

``Runtime.execute()`` is the single execution path for both single-agent
and multi-agent runs. Everything else layers on top:

- ``bf.run(scene, env)`` → convenience sugar
- ``SDK.run(...)`` → backwards-compat shim
- ``Eval.run(...)`` → batch of Runtime.execute()

Architecture:
    Agent  → thin wrapper around registry entry + model + creds
    Environment → wraps Docker/Daytona sandbox, owns lifecycle
    Scene → declarative roles + turns, lowered to rollout Steps
    Runtime → env + rollout execution loop + verify
    RuntimeResult → trajectories + messages + rewards + snapshots
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchflow.agents.registry import AgentConfig, resolve_agent
from benchflow.skill_policy import SKILL_MODE_NO_SKILL

if TYPE_CHECKING:
    from benchflow.models import RolloutResult as RunResult
    from benchflow.rollout import RolloutConfig as TrialConfig

logger = logging.getLogger(__name__)


class Environment:
    """Wraps a Docker/Daytona sandbox environment, owns lifecycle.

    Usage::

        env = Environment.from_task("tasks/my-task", sandbox="daytona")
        await env.start()
        # ... run agents ...
        await env.stop()

    Or as a context manager::

        async with Environment.from_task("tasks/X", sandbox="daytona") as env:
            result = await runtime.execute()
    """

    def __init__(self, inner: Any, task_path: Path, sandbox: str) -> None:
        self._inner = inner
        self.task_path = task_path
        self.sandbox = sandbox
        self._started = False

    @classmethod
    def from_task(
        cls,
        task_path: str | Path,
        sandbox: str = "daytona",
        rollout_name: str | None = None,
        planes: Any | None = None,
    ) -> Environment:
        """Create an environment from a task directory."""
        from uuid import uuid4

        from benchflow.contracts import default_rollout_planes
        from benchflow.task import RolloutPaths, Task

        task_path = Path(task_path)
        task = Task(task_path)
        rollout_name = rollout_name or task_path.name
        rollout_paths = RolloutPaths(
            rollout_dir=Path.cwd()
            / "jobs"
            / "environment"
            / f"{rollout_name}__{uuid4().hex[:8]}"
        )
        rollout_paths.mkdir()
        plane_bundle = planes or default_rollout_planes()
        try:
            inner = plane_bundle.create_environment(
                sandbox,
                task=task,
                task_path=task_path,
                rollout_name=rollout_name,
                rollout_paths=rollout_paths,
                preserve_agent_network=False,
                environment_manifest=None,
            )
        except Exception:
            # create_environment failed (e.g. a missing optional sandbox SDK) —
            # don't leave the empty rollout dir we just created littering
            # jobs/environment/. Best-effort cleanup, then re-raise unchanged.
            import shutil

            shutil.rmtree(rollout_paths.rollout_dir, ignore_errors=True)
            raise
        return cls(inner=inner, task_path=task_path, sandbox=sandbox)

    @property
    def inner(self) -> Any:
        """The underlying harbor environment (Docker/Daytona). Use for Scene-based shared sandbox access."""
        return self._inner

    @property
    def task(self) -> Any:
        from benchflow.task import Task

        return Task(self.task_path)

    async def start(self, force_build: bool = False) -> None:
        await self._inner.start(force_build=force_build)
        self._started = True

    async def stop(self, delete: bool = True) -> None:
        if self._started:
            await self._inner.stop(delete=delete)
            self._started = False

    async def exec(self, cmd: str, **kwargs) -> Any:
        """Run a command in the sandbox.

        Pass ``service="<name>"`` to target an additional compose service
        (a vulhub-style target container) instead of the default agent
        container ``"main"`` — see #248.
        """
        return await self._inner.exec(cmd, **kwargs)

    async def exec_in_service(self, service: str, cmd: str, **kwargs) -> Any:
        """Run a command in a named compose service container (#248).

        Ergonomic wrapper for ``exec(cmd, service=service)``. Useful for
        injecting flags into, or verifying state of, a multi-container
        task's target container.
        """
        return await self._inner.exec(cmd, service=service, **kwargs)

    async def upload_file(self, src: str | Path, dst: str) -> None:
        await self._inner.upload_file(src, dst)

    async def upload_dir(
        self, src: str | Path, dst: str, service: str = "main"
    ) -> None:
        """Upload a directory into the sandbox.

        Pass ``service="<name>"`` to target a non-``main`` compose service
        (a vulhub-style target container) — see #248.
        """
        await self._inner.upload_dir(src, dst, service=service)

    async def download_file(self, src: str, dst: str | Path) -> None:
        await self._inner.download_file(src, dst)

    async def download_dir(
        self, src: str, dst: str | Path, service: str = "main"
    ) -> None:
        """Download a directory from the sandbox.

        Pass ``service="<name>"`` to fetch from a non-``main`` compose service
        (a vulhub-style target container) — see #248.
        """
        await self._inner.download_dir(src, dst, service=service)

    async def __aenter__(self) -> Environment:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    def __repr__(self) -> str:
        return f"Environment({self.task_path.name!r}, sandbox={self.sandbox!r})"


@dataclass
class Agent:
    """Thin wrapper around a registered agent + model + credentials."""

    name: str
    model: str
    env: dict[str, str] = field(default_factory=dict)

    @property
    def config(self) -> AgentConfig | None:
        try:
            return resolve_agent(self.name)
        except KeyError:
            return None

    @property
    def launch_cmd(self) -> str:
        config = self.config
        if config is None:
            return self.name
        return config.launch_override_cmd or config.launch_cmd

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, model={self.model!r})"


@dataclass
class RuntimeConfig:
    """Configuration for a Runtime execution."""

    sandbox_user: str | None = "agent"
    sandbox_setup_timeout: int = 120
    max_rounds: int = 10
    snapshot_policy: str = "none"
    reward_stream: bool = True
    timeout: int = 900
    jobs_dir: str | Path = "jobs"
    rollout_name: str | None = None
    skills_dir: str | Path | None = None
    skill_mode: str = SKILL_MODE_NO_SKILL
    context_root: str | Path | None = None
    base_image_override: str | None = None
    pre_agent_hooks: list | None = None
    sandbox_locked_paths: list[str] | None = None
    usage_tracking: Any = None


@dataclass
class RuntimeResult:
    """Canonical output from Runtime.execute().

    Artifact-oriented: exposes paths and structured summaries,
    not only in-memory objects.

    Guaranteed artifacts (when run completes):
        rollout_dir/result.json       — reward, timing, error, metadata
        rollout_dir/rewards.jsonl     — terminal + rubric reward events
        rollout_dir/trajectory/       — ACP trajectory JSONL
        rollout_dir/timing.json       — phase-level timing
        rollout_dir/config.json       — run configuration snapshot
        rollout_dir/prompts.json      — prompts sent to agent

    Optional artifacts:
        rollout_dir/snapshots/             — checkpoint refs (if snapshot_policy != "none")
    """

    task_name: str
    rollout_name: str
    reward: float | None
    rewards: dict | None
    n_tool_calls: int
    error: str | None
    verifier_error: str | None
    trajectory: list[dict]
    messages: list[dict] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)
    rollout_dir: Path | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def passed(self) -> bool:
        from benchflow._utils.scoring import classify_result_outcome

        return (
            classify_result_outcome(
                {
                    "rewards": self.rewards,
                    "error": self.error,
                    "verifier_error": self.verifier_error,
                }
            )
            == "passed"
        )

    @property
    def verified(self) -> bool:
        from benchflow._utils.scoring import classify_result_outcome

        return classify_result_outcome(
            {
                "rewards": self.rewards,
                "error": self.error,
                "verifier_error": self.verifier_error,
            }
        ) in {"passed", "failed"}


class Runtime:
    """The 0.3 execution center.

    Single execution path for both single-agent and multi-agent runs.
    Owns: environment lifecycle, agent setup, ACP session, verification,
    reward emission, snapshots, artifact writing.

    Usage::

        agent = Agent("gemini", model="gemini-3.1-flash-lite-preview")
        env = Environment.from_task("tasks/X", sandbox="daytona")
        runtime = Runtime(env, agent)
        result = await runtime.execute()
    """

    def __init__(
        self,
        env: Environment,
        agent: Agent,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config or RuntimeConfig()

    async def execute(self) -> RuntimeResult:
        """Run the full execution loop via Trial.

        Runtime is the stable user-facing surface. Trial owns the
        decomposed lifecycle phases underneath.

        Honours the caller-supplied :class:`Environment`: the live
        ``env.inner`` sandbox is reused instead of creating a second one
        (fixes #388). If the caller has not yet called ``env.start()``
        we start it now so ``env`` stays in a consistent state and the
        underlying sandbox is brought up exactly once.
        """
        from benchflow._types import Scene
        from benchflow.rollout import Rollout, RolloutConfig

        config = self.config
        trial_config = RolloutConfig(
            task_path=self.env.task_path,
            scenes=[
                Scene.single(
                    agent=self.agent.name,
                    model=self.agent.model,
                )
            ],
            environment=self.env.sandbox,
            sandbox_user=config.sandbox_user,
            sandbox_locked_paths=config.sandbox_locked_paths,
            sandbox_setup_timeout=config.sandbox_setup_timeout,
            jobs_dir=config.jobs_dir,
            rollout_name=config.rollout_name,
            timeout=config.timeout,
            context_root=config.context_root,
            base_image_override=config.base_image_override,
            pre_agent_hooks=config.pre_agent_hooks,
            agent=self.agent.name,
            model=self.agent.model,
            agent_env=self.agent.env,
            skills_dir=config.skills_dir,
            skill_mode=config.skill_mode,
            usage_tracking=config.usage_tracking,
        )

        rollout = await Rollout.create(trial_config)

        # If the caller has not started the Environment yet, do it now so
        # the Environment's _started flag stays accurate and the caller's
        # later env.stop() works. Then hand the live sandbox to Rollout —
        # this is what makes Runtime honour the input Environment instead
        # of silently building a second one. #388.
        if not self.env._started:
            await self.env.start()
        rollout.use_prebuilt_env(self.env.inner)

        run_result = await rollout.run()

        reward = (run_result.rewards or {}).get("reward")
        # The Rollout owns the on-disk artifact directory; lift it into the
        # RuntimeResult so callers can locate result.json / trajectory/ etc.
        # without scanning jobs_dir. Required because RuntimeResult's
        # docstring promises an artifact-oriented surface (#378).
        rollout_dir = getattr(rollout, "_rollout_dir", None)
        return RuntimeResult(
            task_name=run_result.task_name,
            rollout_name=run_result.rollout_name,
            reward=reward,
            rewards=run_result.rewards,
            n_tool_calls=run_result.n_tool_calls,
            error=run_result.error,
            verifier_error=run_result.verifier_error,
            trajectory=run_result.trajectory,
            rollout_dir=rollout_dir,
            started_at=run_result.started_at,
            finished_at=run_result.finished_at,
        )


async def run(
    subject: Agent | TrialConfig | str,
    env: Environment | str | None = None,
    config: RuntimeConfig | None = None,
    *,
    task_path: str | Path | None = None,
    model: str | None = None,
) -> RuntimeResult | RunResult:
    """Primary user-facing API — multiple calling conventions.

    Usage::

        import benchflow as bf

        # 1. TrialConfig (Scene-based, full control)
        result = await bf.run(TrialConfig(task_path=..., scenes=[...]))

        # 2. Agent + Environment (0.3 style)
        result = await bf.run(Agent("gemini", "flash"), Environment.from_task("tasks/X"))

        # 3. Agent name string (simplest)
        result = await bf.run("gemini", task_path="tasks/X")
    """
    from benchflow._types import Scene
    from benchflow.rollout import SKILL_MODE_SELF_GEN, Rollout, RolloutConfig

    if isinstance(subject, RolloutConfig):
        if subject.skill_mode == SKILL_MODE_SELF_GEN:
            from benchflow.self_gen import run_self_gen

            return await run_self_gen(subject)
        rollout = await Rollout.create(subject)
        return await rollout.run()

    if isinstance(subject, Agent):
        if not isinstance(env, Environment):
            raise TypeError(
                f"When passing an Agent, env must be an Environment, got {type(env).__name__}. "
                f"Use bf.run('agent-name', task_path=...) for the string shortcut."
            )
        runtime = Runtime(env, subject, config)
        return await runtime.execute()

    if isinstance(subject, str):
        if task_path is None:
            raise ValueError("task_path required when passing agent name as string")
        rc = config or RuntimeConfig()
        rollout_config = RolloutConfig(
            task_path=Path(task_path),
            scenes=[Scene.single(agent=subject, model=model)],
            environment=env if isinstance(env, str) else "docker",
            sandbox_user=rc.sandbox_user,
            sandbox_locked_paths=rc.sandbox_locked_paths,
            sandbox_setup_timeout=rc.sandbox_setup_timeout,
            jobs_dir=rc.jobs_dir,
            context_root=rc.context_root,
            base_image_override=rc.base_image_override,
            pre_agent_hooks=rc.pre_agent_hooks,
            skills_dir=rc.skills_dir,
            skill_mode=rc.skill_mode,
            agent=subject,
            model=model,
            usage_tracking=rc.usage_tracking,
        )
        rollout = await Rollout.create(rollout_config)
        return await rollout.run()

    raise TypeError(f"Unsupported subject type: {type(subject).__name__}")
