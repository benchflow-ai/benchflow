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

from benchflow.agents.registry import AGENTS, AGENT_LAUNCH, AgentConfig

logger = logging.getLogger(__name__)


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
