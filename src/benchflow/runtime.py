"""BenchFlow Runtime — the 0.3 execution center.

``Runtime.execute()`` is the single execution path for both single-agent
and multi-agent runs. Everything else layers on top:

- ``bf.run(scene, env)`` → convenience sugar
- ``SDK.run(...)`` → backwards-compat shim
- ``Eval.run(...)`` → batch of Runtime.execute()

Architecture:
    Agent  → thin wrapper around registry entry + model + creds
    Environment → wraps harbor Docker/Daytona env, owns lifecycle
    Scene → 1+ roles + transport + scheduler (from _scene.py)
    Runtime → env + scene + execute loop + verify
    RuntimeResult → trajectories + messages + rewards + snapshots
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.agents.registry import AGENT_LAUNCH, AGENTS, AgentConfig

logger = logging.getLogger(__name__)


class Environment:
    """Wraps a Harbor Docker/Daytona environment, owns lifecycle.

    Usage::

        env = Environment.from_task("tasks/my-task", backend="daytona")
        await env.start()
        # ... run agents ...
        await env.stop()

    Or as a context manager::

        async with Environment.from_task("tasks/X", backend="daytona") as env:
            result = await runtime.execute()
    """

    def __init__(self, inner: Any, task_path: Path, backend: str) -> None:
        self._inner = inner
        self.task_path = task_path
        self.backend = backend
        self._started = False

    @classmethod
    def from_task(
        cls,
        task_path: str | Path,
        backend: str = "daytona",
        trial_name: str | None = None,
    ) -> Environment:
        """Create an environment from a task directory."""
        from harbor.models.task.task import Task

        from benchflow._env_setup import _create_environment

        task_path = Path(task_path)
        task = Task(task_path)
        trial_name = trial_name or task_path.name
        inner = _create_environment(
            environment_type=backend,
            task=task,
            task_path=task_path,
            trial_name=trial_name,
            trial_paths=None,
        )
        return cls(inner=inner, task_path=task_path, backend=backend)

    @property
    def task(self) -> Any:
        from harbor.models.task.task import Task

        return Task(self.task_path)

    async def start(self, force_build: bool = False) -> None:
        await self._inner.start(force_build=force_build)
        self._started = True

    async def stop(self, delete: bool = True) -> None:
        if self._started:
            await self._inner.stop(delete=delete)
            self._started = False

    async def exec(self, cmd: str, **kwargs) -> Any:
        return await self._inner.exec(cmd, **kwargs)

    async def upload_file(self, src: str | Path, dst: str) -> None:
        await self._inner.upload_file(src, dst)

    async def upload_dir(self, src: str | Path, dst: str) -> None:
        await self._inner.upload_dir(src, dst)

    async def download_file(self, src: str, dst: str | Path) -> None:
        await self._inner.download_file(src, dst)

    async def __aenter__(self) -> Environment:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    def __repr__(self) -> str:
        return f"Environment({self.task_path.name!r}, backend={self.backend!r})"


@dataclass
class Agent:
    """Thin wrapper around a registered agent + model + credentials."""

    name: str
    model: str
    env: dict[str, str] = field(default_factory=dict)

    @property
    def config(self) -> AgentConfig | None:
        return AGENTS.get(self.name)

    @property
    def launch_cmd(self) -> str:
        return AGENT_LAUNCH.get(self.name, self.name)

    def __repr__(self) -> str:
        return f"Agent({self.name!r}, model={self.model!r})"


@dataclass
class RuntimeConfig:
    """Configuration for a Runtime execution."""

    sandbox_user: str | None = "agent"
    max_rounds: int = 10
    snapshot_policy: str = "none"
    reward_stream: bool = True
    timeout: int = 900
    jobs_dir: str | Path = "jobs"
    trial_name: str | None = None
    skills_dir: str | Path | None = None
    context_root: str | Path | None = None
    pre_agent_hooks: list | None = None
    sandbox_locked_paths: list[str] | None = None


@dataclass
class RuntimeResult:
    """Canonical output from Runtime.execute().

    Artifact-oriented: exposes paths and structured summaries,
    not only in-memory objects.

    Guaranteed artifacts (when run completes):
        trial_dir/result.json       — reward, timing, error, metadata
        trial_dir/rewards.jsonl     — terminal + rubric reward events
        trial_dir/trajectory/       — ACP trajectory JSONL
        trial_dir/timing.json       — phase-level timing
        trial_dir/config.json       — run configuration snapshot
        trial_dir/prompts.json      — prompts sent to agent

    Optional artifacts:
        trial_dir/scene_trajectory.jsonl — inter-agent messages (multi-agent)
        trial_dir/snapshots/             — checkpoint refs (if snapshot_policy != "none")
    """

    task_name: str
    trial_name: str
    reward: float | None
    rewards: dict | None
    n_tool_calls: int
    error: str | None
    verifier_error: str | None
    trajectory: list[dict]
    messages: list[dict] = field(default_factory=list)
    snapshots: list[str] = field(default_factory=list)
    trial_dir: Path | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def passed(self) -> bool:
        return self.reward is not None and self.reward > 0

    @property
    def verified(self) -> bool:
        return self.verifier_error is None and self.reward is not None

    def to_run_result(self) -> Any:
        """Convert to legacy RunResult for SDK.run() compat."""
        from benchflow.models import RunResult

        return RunResult(
            task_name=self.task_name,
            trial_name=self.trial_name,
            rewards=self.rewards,
            trajectory=self.trajectory,
            agent="",
            agent_name="",
            model="",
            n_tool_calls=self.n_tool_calls,
            n_prompts=0,
            error=self.error,
            verifier_error=self.verifier_error,
            partial_trajectory=False,
            trajectory_source=None,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )


class Runtime:
    """The 0.3 execution center.

    Single execution path for both single-agent and multi-agent runs.
    Owns: environment lifecycle, agent setup, ACP session, verification,
    reward emission, snapshots, artifact writing.

    Usage::

        agent = Agent("gemini", model="gemini-3.1-flash-lite-preview")
        env = Environment.from_task("tasks/X", backend="daytona")
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
        """Run the full execution loop: setup -> agent -> verify -> result.

        Delegates to SDK.run() internally for the execution mechanics.
        The Runtime API is the stable surface; internals will migrate
        from SDK to Runtime directly in subsequent PRs.
        """
        from benchflow.sdk import SDK

        config = self.config
        sdk = SDK()
        run_result = await sdk.run(
            task_path=self.env.task_path,
            agent=self.agent.name,
            model=self.agent.model,
            agent_env=self.agent.env,
            environment=self.env.backend,
            jobs_dir=str(config.jobs_dir),
            trial_name=config.trial_name,
            sandbox_user=config.sandbox_user,
            sandbox_locked_paths=config.sandbox_locked_paths,
            skills_dir=config.skills_dir,
            context_root=config.context_root,
            pre_agent_hooks=config.pre_agent_hooks,
        )

        reward = (run_result.rewards or {}).get("reward")
        return RuntimeResult(
            task_name=run_result.task_name,
            trial_name=run_result.trial_name,
            reward=reward,
            rewards=run_result.rewards,
            n_tool_calls=run_result.n_tool_calls,
            error=run_result.error,
            verifier_error=run_result.verifier_error,
            trajectory=run_result.trajectory,
            started_at=run_result.started_at,
            finished_at=run_result.finished_at,
        )


async def run(
    agent: Agent,
    env: Environment,
    config: RuntimeConfig | None = None,
) -> RuntimeResult:
    """Convenience function — the primary user-facing API.

    Usage::

        import benchflow as bf
        result = await bf.run(agent, env)
        print(result.reward)
    """
    runtime = Runtime(env, agent, config)
    return await runtime.execute()
