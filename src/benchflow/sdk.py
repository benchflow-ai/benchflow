"""Backward-compat shim — the SDK class delegates to Rollout.

New code should use ``bf.run()`` or ``Rollout`` directly.
``from benchflow.sdk import SDK`` keeps working for existing callers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow._types import Scene
from benchflow.contracts import default_rollout_planes
from benchflow.diagnostics import VerifierTimeoutDiagnostic
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.models import RolloutResult, TrajectorySource
from benchflow.rollout import (
    _build_rollout_result,
    _init_rollout,
    _resolve_prompts,
    _run_oracle,
    _start_env_and_upload,
    _verify_rollout,
    _write_config,
    _write_rewards_jsonl,
)

logger = logging.getLogger(__name__)

# Re-export so ``from benchflow.sdk import _write_rewards_jsonl`` keeps working.
__all__ = ["SDK", "_write_rewards_jsonl"]

# Backward-compat alias
RunResult = RolloutResult


class SDK:
    """Backward-compat shim — delegates to :mod:`benchflow.rollout`.

    Usage::

        sdk = SDK()
        result = await sdk.run(task_path=..., agent=...)
    """

    @staticmethod
    def _init_trial(
        task_path: Path,
        job_name: str | None,
        rollout_name: str | None,
        jobs_dir: str | Path,
    ) -> tuple[Any, Path, Any, datetime, str, str]:
        return _init_rollout(task_path, job_name, rollout_name, jobs_dir)

    @staticmethod
    def _write_config(
        rollout_dir: Path,
        **kwargs: Any,
    ) -> None:
        _write_config(rollout_dir, **kwargs)

    @staticmethod
    def _resolve_prompts(
        task_path: Path,
        prompts: list[str | None] | None,
        skills_dir: str | Path | None = None,
        skill_nudge: str = "",
        agent: str | None = None,
    ) -> list[str]:
        return _resolve_prompts(
            task_path,
            prompts,
            skills_dir=skills_dir,
            skill_nudge=skill_nudge,
            agent=agent,
        )

    @staticmethod
    def _build_result(
        rollout_dir: Path,
        *,
        task_name: str,
        rollout_name: str,
        agent: str,
        agent_name: str,
        model: str | None,
        n_tool_calls: int,
        prompts: list[str],
        error: str | None,
        verifier_error: str | None,
        trajectory: list[dict],
        partial_trajectory: bool,
        trajectory_source: TrajectorySource | None = None,
        rewards: dict | None,
        started_at: datetime,
        timing: dict[str, float],
        scenes: list[Scene] | None = None,
        source_provenance: dict[str, Any] | None = None,
    ) -> RolloutResult:
        return _build_rollout_result(
            rollout_dir,
            task_name=task_name,
            rollout_name=rollout_name,
            agent=agent,
            agent_name=agent_name,
            model=model,
            n_tool_calls=n_tool_calls,
            prompts=prompts,
            error=error,
            verifier_error=verifier_error,
            trajectory=trajectory,
            partial_trajectory=partial_trajectory,
            trajectory_source=trajectory_source,
            rewards=rewards,
            started_at=started_at,
            timing=timing,
            scenes=scenes,
            source_provenance=source_provenance,
        )

    async def _start_env_and_upload(
        self, env: Any, task_path: Path, timing: dict
    ) -> None:
        await _start_env_and_upload(env, task_path, timing)

    async def _run_oracle(
        self,
        env: Any,
        task_path: Path,
        timeout: int,
        sandbox_user: str | None = None,
    ) -> tuple[list[dict], str]:
        return await _run_oracle(env, task_path, timeout, sandbox_user=sandbox_user)

    async def _verify(
        self,
        env: Any,
        task: Any,
        rollout_paths: Any,
        timing: dict,
        sandbox_user: str | None = None,
        workspace: str | None = None,
    ) -> tuple[dict | None, str | None, VerifierTimeoutDiagnostic | None]:
        return await _verify_rollout(
            env,
            task,
            rollout_paths,
            timing,
            default_rollout_planes(),
            sandbox_user=sandbox_user,
            workspace=workspace,
        )

    async def run(
        self,
        task_path: str | Path,
        agent: str = "claude-agent-acp",
        prompts: list[str | None] | None = None,
        *,
        model: str | None = None,
        agent_env: dict[str, str] | None = None,
        job_name: str | None = None,
        rollout_name: str | None = None,
        jobs_dir: str | Path = "jobs",
        concurrency: int = 1,
        agent_idle_timeout: int | None = 600,
        environment: str = "docker",
        environment_manifest: EnvironmentManifest | None = None,
        skills_dir: str | Path | None = None,
        sandbox_user: str | None = "agent",
        sandbox_locked_paths: list[str] | None = None,
        sandbox_setup_timeout: int = 120,
        pre_agent_hooks: list | None = None,
        context_root: str | Path | None = None,
        skill_mode: str = "default",
        skill_creator_dir: str | Path | None = None,
        self_gen_no_internet: bool = False,
        source_provenance: dict[str, Any] | None = None,
        usage_tracking: Any = None,
    ) -> RolloutResult:
        """Run a task — delegates to :func:`benchflow.run`."""
        from benchflow.rollout import RolloutConfig
        from benchflow.runtime import run

        config = RolloutConfig(
            task_path=Path(task_path),
            agent=agent,
            prompts=prompts,
            model=model,
            agent_env=agent_env,
            job_name=job_name,
            rollout_name=rollout_name,
            jobs_dir=jobs_dir,
            concurrency=concurrency,
            agent_idle_timeout=agent_idle_timeout,
            environment=environment,
            environment_manifest=environment_manifest,
            skills_dir=skills_dir,
            sandbox_user=sandbox_user,
            sandbox_locked_paths=sandbox_locked_paths,
            sandbox_setup_timeout=sandbox_setup_timeout,
            pre_agent_hooks=pre_agent_hooks,
            context_root=context_root,
            skill_mode=skill_mode,
            skill_creator_dir=skill_creator_dir,
            self_gen_no_internet=self_gen_no_internet,
            source_provenance=source_provenance,
            usage_tracking=usage_tracking,
        )
        return await run(config)  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
