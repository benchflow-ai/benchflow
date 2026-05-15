"""Rollout result model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

TrajectorySource = Literal["acp", "scraped", "partial_acp"]
"""Provenance label for captured trajectory events."""


class RolloutResult:
    """Outcome of one rollout attempt."""

    def __init__(
        self,
        task_name: str,
        trial_name: str = "",
        rewards: dict[str, float | int] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
        agent: str = "",
        agent_name: str = "",
        model: str = "",
        n_tool_calls: int = 0,
        n_prompts: int = 0,
        error: str | None = None,
        verifier_error: str | None = None,
        partial_trajectory: bool = False,
        trajectory_source: TrajectorySource | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ):
        self.task_name = task_name
        # ``trial_name`` is retained in artifacts during migration; rollout_name
        # is exposed as the rollout-native spelling.
        self.trial_name = trial_name
        self.rewards = rewards
        self.trajectory = trajectory or []
        self.agent = agent
        self.agent_name = agent_name
        self.model = model
        self.n_tool_calls = n_tool_calls
        self.n_prompts = n_prompts
        self.error = error
        self.verifier_error = verifier_error
        self.partial_trajectory = partial_trajectory
        self.trajectory_source = trajectory_source
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def rollout_name(self) -> str:
        """Rollout-native name for the current artifact ``trial_name``."""

        return self.trial_name

    @property
    def success(self) -> bool:
        """True when the rollout completed without agent or verifier error."""

        return self.error is None and self.verifier_error is None

    def __repr__(self) -> str:
        status = "OK" if self.success else f"ERROR: {self.error or self.verifier_error}"
        return (
            f"{self.__class__.__name__}(task={self.task_name}, {status}, "
            f"rewards={self.rewards}, "
            f"trajectory={len(self.trajectory)} events)"
        )
