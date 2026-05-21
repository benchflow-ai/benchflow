"""Core reward protocol and result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from benchflow.rewards.events import Granularity, RewardEvent, Space


@runtime_checkable
class RewardFunc(Protocol):
    """Single scoring dimension."""

    async def score(self, rollout_dir: Path) -> float: ...


@dataclass
class VerifyResult:
    """Aggregated result from a Rubric or node evaluation.

    The architecture's Reward contract result (``docs/architecture.md``,
    "The four contracts"): ``{reward, items, events, space, granularity}`` —
    ``space`` and ``granularity`` are the ``(space, granularity)`` tag the
    architecture mandates on every reward record.

    Attributes:
        reward:      Weighted aggregate score across all reward functions.
        items:       Per-function scores keyed by source name.
        events:      Reward events collected during scoring.
        error:       Error message if scoring failed, else None.
        space:       Evaluation space of the headline ``reward`` — "output"
                     (did it finish the job?), "action", "reasoning",
                     "memory", or "latent". Defaults to "output".
        granularity: "terminal" (whole trajectory) or "step" (one edge).
                     Defaults to "terminal".
    """

    reward: float
    items: dict[str, float] = field(default_factory=dict)
    events: list[RewardEvent] = field(default_factory=list)
    error: str | None = None
    space: Space = "output"
    granularity: Granularity = "terminal"
