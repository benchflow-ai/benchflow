"""Core reward protocol and result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from benchflow.rewards.events import RewardEvent


@runtime_checkable
class RewardFunc(Protocol):
    """Single scoring dimension."""

    async def score(self, rollout_dir: Path) -> float: ...


@dataclass
class VerifyResult:
    """Aggregated result from a Rubric evaluation.

    Attributes:
        reward: Weighted aggregate score across all reward functions.
        items:  Per-function scores keyed by class name.
        events: Reward events collected during scoring.
        error:  Error message if scoring failed, else None.
    """

    reward: float
    items: dict[str, float] = field(default_factory=dict)
    events: list[RewardEvent] = field(default_factory=list)
    error: str | None = None
