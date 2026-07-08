"""Reusable single-task runtime primitive for training loops.

This module exposes the sandbox + verifier subset of a :class:`Rollout`
without launching an ACP agent. It is intentionally small: online training
loops can bring their own policy/model loop, execute bash-like actions in the
BenchFlow task sandbox, then call the normal verifier and artifact writers.
"""

from __future__ import annotations

import contextlib
import shlex
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from benchflow.environment.manifest import EnvironmentManifest
from benchflow.models import RolloutResult
from benchflow.rollout._config import RolloutConfig
from benchflow.skill_policy import SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL


@dataclass
class TaskRuntimeConfig:
    """Configuration for a BenchFlow task runtime session."""

    task_path: str | Path
    environment: str = "docker"
    sandbox_user: str | None = "agent"
    sandbox_locked_paths: list[str] | None = None
    sandbox_setup_timeout: int = 120
    job_name: str | None = None
    rollout_name: str | None = None
    jobs_dir: str | Path = "jobs"
    context_root: str | Path | None = None
    base_image_override: str | None = None
    pre_agent_hooks: list | None = None
    environment_manifest: EnvironmentManifest | None = None
    config_override: dict | None = None
    skills_dir: str | Path | None = None
    skill_mode: str = SKILL_MODE_NO_SKILL
    runtime_label: str = "task-runtime"
    planes: Any | None = None

    def __post_init__(self) -> None:
        self.task_path = Path(self.task_path)
        if self.context_root is not None:
            self.context_root = Path(self.context_root)
        if self.skills_dir is not None:
            self.skills_dir = Path(self.skills_dir)
        if self.skill_mode not in {SKILL_MODE_NO_SKILL, SKILL_MODE_WITH_SKILL}:
            raise ValueError("TaskRuntimeConfig supports no-skill and with-skill only")

    def to_rollout_config(self) -> RolloutConfig:
        """Lower to the Rollout configuration used to own lifecycle/artifacts."""

        return RolloutConfig(
            task_path=Path(self.task_path),
            environment=self.environment,
            sandbox_user=self.sandbox_user,
            sandbox_locked_paths=self.sandbox_locked_paths,
            sandbox_setup_timeout=self.sandbox_setup_timeout,
            skip_agent_install=True,
            job_name=self.job_name,
            rollout_name=self.rollout_name,
            jobs_dir=self.jobs_dir,
            context_root=self.context_root,
            base_image_override=self.base_image_override,
            pre_agent_hooks=self.pre_agent_hooks,
            environment_manifest=self.environment_manifest,
            config_override=self.config_override,
            agent=self.runtime_label,
            model=None,
            prompts=[],
            skills_dir=self.skills_dir,
            skill_mode=self.skill_mode,
            allow_document_user=False,
            planes=self.planes,
        )


@dataclass(frozen=True)
class BashToolResult:
    """Result from a runtime bash tool call."""

    command: str
    return_code: int
    stdout: str
    stderr: str
    elapsed_sec: float


@dataclass(frozen=True)
class TaskRuntimeResult:
    """Verifier result plus the normal BenchFlow artifact location."""

    task_name: str
    rollout_name: str
    reward: float | None
    rewards: dict | None
    verifier_error: str | None
    error: str | None
    rollout_dir: Path
    result: RolloutResult


class TaskRuntime:
    """Run one BenchFlow-compatible task as a reusable sandbox primitive.

    The caller drives the policy/training loop externally through
    :meth:`bash`. :meth:`verify` then runs the same verifier and result writers
    as a regular rollout, preserving artifact paths under ``jobs_dir``.
    """

    def __init__(self, config: TaskRuntimeConfig) -> None:
        self.config = config
        self._rollout: Any | None = None
        self._started = False
        self._verified = False

    @classmethod
    async def create(cls, config: TaskRuntimeConfig) -> TaskRuntime:
        """Create and start a task runtime session."""

        runtime = cls(config)
        await runtime.start()
        return runtime

    @property
    def rollout(self) -> Any:
        if self._rollout is None:
            raise RuntimeError("TaskRuntime.start() must run first")
        return self._rollout

    @property
    def env(self) -> Any:
        return self.rollout.env

    @property
    def workspace(self) -> str:
        rollout = self.rollout
        return getattr(rollout, "_agent_cwd", None) or "/app"

    @property
    def rollout_dir(self) -> Path:
        rollout_dir = getattr(self.rollout, "_rollout_dir", None)
        if not isinstance(rollout_dir, Path):
            raise RuntimeError("TaskRuntime.start() did not initialize artifacts")
        return rollout_dir

    async def start(self) -> None:
        """Set up and start the sandbox without launching an ACP agent."""

        if self._started:
            return
        from benchflow.rollout import Rollout

        rollout = await Rollout.create(self.config.to_rollout_config())
        try:
            await rollout.setup()
            await rollout.start()
            await rollout.install_agent()
        except BaseException:
            with contextlib.suppress(Exception):
                await rollout.cleanup()
            raise
        self._rollout = rollout
        self._started = True

    async def bash(
        self,
        command: str,
        *,
        timeout_sec: int = 30,
        user: str | None = None,
    ) -> BashToolResult:
        """Execute a bash-like command in the task workspace.

        The command is run through ``bash -lc`` after ``cd`` into the resolved
        task workspace. This mirrors the shell affordance most training loops
        need while keeping provider-specific tool protocols out of BenchFlow.
        """

        if not self._started:
            raise RuntimeError("TaskRuntime.start() must run before bash()")
        if not hasattr(self.env, "exec"):
            raise RuntimeError("Active sandbox does not expose exec()")
        workspace = shlex.quote(self.workspace)
        script = f"cd {workspace} && {command}"
        shell_command = f"bash -lc {shlex.quote(script)}"
        exec_user = user if user is not None else self.config.sandbox_user

        t0 = monotonic()
        result = await self.env.exec(
            shell_command,
            user=exec_user or "root",
            timeout_sec=timeout_sec,
        )
        elapsed = monotonic() - t0
        tool_result = BashToolResult(
            command=command,
            return_code=int(
                getattr(result, "return_code", getattr(result, "exit_code", 0))
            ),
            stdout=str(getattr(result, "stdout", "") or ""),
            stderr=str(getattr(result, "stderr", "") or ""),
            elapsed_sec=round(elapsed, 3),
        )
        self._record_bash_event(tool_result, timeout_sec=timeout_sec, user=exec_user)
        return tool_result

    def _record_bash_event(
        self,
        result: BashToolResult,
        *,
        timeout_sec: int,
        user: str | None,
    ) -> None:
        self.rollout.record_external_tool_call(
            tool_name="bash",
            event={
                "command": result.command,
                "return_code": result.return_code,
                "status": "completed" if result.return_code == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timeout_sec": timeout_sec,
                "user": user,
                "elapsed_sec": result.elapsed_sec,
            },
        )

    async def verify(self) -> TaskRuntimeResult:
        """Run the task verifier, write normal rollout artifacts, and return reward."""

        if self._verified:
            raise RuntimeError("TaskRuntime.verify() can only be called once")
        rewards = await self.rollout.verify()
        self._verified = True
        result = self.rollout.result
        if result is None:
            raise RuntimeError("Rollout did not produce a verified result")
        reward = (rewards or {}).get("reward") if isinstance(rewards, dict) else None
        return TaskRuntimeResult(
            task_name=result.task_name,
            rollout_name=result.rollout_name,
            reward=reward,
            rewards=rewards,
            verifier_error=result.verifier_error,
            error=result.error,
            rollout_dir=self.rollout_dir,
            result=result,
        )

    async def close(self) -> None:
        """Clean up the sandbox lifecycle owned by this runtime."""

        if self._rollout is None:
            return
        await self._rollout.cleanup()
        self._started = False

    async def __aenter__(self) -> TaskRuntime:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


__all__ = [
    "BashToolResult",
    "TaskRuntime",
    "TaskRuntimeConfig",
    "TaskRuntimeResult",
]
