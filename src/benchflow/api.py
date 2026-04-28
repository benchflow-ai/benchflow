# Semver-stable verb contract. Signature changes require contract review (see CLAUDE.md).
"""BenchFlow public ergonomic facade — the verb-tier contract surface.

This file is the verb-side counterpart to ``benchflow.contracts`` (nouns).
Together they are the two semver-stable public-API files: every other module
under ``benchflow`` is implementation, free to churn between minors.

Verbs (signature-stable):
    ``bf.run(...)``       — single-trial entry point; thin wrapper over Trial.
    ``bf.run_batch(...)`` — many-trial entry point (added in v0.4); wraps Job.

Thin classes:
    ``Agent``        — registry handle + model + credentials.
    ``Environment``  — harbor Docker/Daytona wrapper with lifecycle.
    ``RuntimeResult``— legacy result type (kept for SDK back-compat shim).

Edits to this file should be reviewed against the public-API snapshot test
(:mod:`tests.test_public_api_snapshot`) — drift fails CI unless the
snapshot is updated in the same PR.
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
        from benchflow.task import Task

        from benchflow.sandbox.build import _create_environment

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
        """Convert to legacy TrialResult for SDK.run() compat."""
        from benchflow.results import TrialResult

        return TrialResult(
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


async def run(
    subject: "Agent | TrialConfig | str",
    env: "Environment | str | None" = None,
    *,
    task_path: "str | Path | None" = None,
    model: str | None = None,
) -> "TrialResult":
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
    from benchflow.trial import Scene, Trial, TrialConfig

    if isinstance(subject, TrialConfig):
        trial = await Trial.create(subject)
        return await trial.run()

    if isinstance(subject, Agent):
        if not isinstance(env, Environment):
            raise TypeError(
                f"When passing an Agent, env must be an Environment, got {type(env).__name__}. "
                f"Use bf.run('agent-name', task_path=...) for the string shortcut."
            )
        trial_config = TrialConfig(
            task_path=env.task_path,
            scenes=[Scene.single(agent=subject.name, model=subject.model)],
            environment=env.backend,
            agent=subject.name,
            model=subject.model,
            agent_env=subject.env,
        )
        trial = await Trial.create(trial_config)
        return await trial.run()

    if isinstance(subject, str):
        if task_path is None:
            raise ValueError("task_path required when passing agent name as string")
        trial_config = TrialConfig(
            task_path=Path(task_path),
            scenes=[Scene.single(agent=subject, model=model)],
            environment=env if isinstance(env, str) else "docker",
            agent=subject,
            model=model,
        )
        trial = await Trial.create(trial_config)
        return await trial.run()

    raise TypeError(f"Unsupported subject type: {type(subject).__name__}")


async def run_batch(
    tasks: "str | Path",
    agent: str = "claude-agent-acp",
    model: str | None = None,
    *,
    jobs_dir: "str | Path | None" = None,
    concurrency: int = 4,
    retries: int = 0,
    environment: str = "docker",
    prompts: list[str | None] | None = None,
    agent_env: dict[str, str] | None = None,
    skills_dir: str | None = None,
) -> Any:
    """Run an agent against a directory of tasks — many trials, one JobResult.

    Thin wrapper over :class:`benchflow.job.Job`. Returns a
    :class:`benchflow.JobResult` with aggregate counts (passed / failed /
    errored / verifier_errored) plus the score.

    Usage::

        import benchflow as bf

        result = await bf.run_batch("benchmarks/terminal-bench/tasks",
                                    agent="claude-agent-acp",
                                    concurrency=8)
        print(result.score)

    Per-trial detail (trajectories, n_tool_calls) is written to
    ``jobs_dir/<trial>/result.json`` by the job runner — JobResult itself
    deliberately omits per-trial trajectories to stay aggregate-only.
    """
    from benchflow.contracts.job_config import JobConfig, RetryConfig
    from benchflow.job import Job

    tasks_dir = Path(tasks)
    if jobs_dir is None:
        jobs_dir = Path("jobs") / datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

    config = JobConfig(
        agent=agent,
        model=model,
        environment=environment,
        concurrency=concurrency,
        retry=RetryConfig(max_retries=retries),
        prompts=prompts,
        agent_env=agent_env or {},
        skills_dir=skills_dir,
    )
    job = Job(tasks_dir=tasks_dir, jobs_dir=jobs_dir, config=config)
    return await job.run()
