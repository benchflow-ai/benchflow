"""Composable reward functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from benchflow.rewards.events import RewardEvent

RewardValue = float | RewardEvent | list[RewardEvent]


@dataclass(frozen=True)
class RewardContext:
    """Inputs available to reward functions."""

    rollout_dir: Path
    task_dir: Path | None = None
    trajectory: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class RewardFunc(Protocol):
    """Scoring interface for terminal, process, or dense rewards."""

    async def score(self, ctx: RewardContext) -> RewardValue: ...


@dataclass
class Rubric:
    """Weighted collection of reward functions."""

    reward_funcs: list[RewardFunc]
    weights: list[float] | None = None

    async def score(self, ctx: RewardContext) -> list[RewardEvent]:
        if self.weights is not None and len(self.weights) != len(self.reward_funcs):
            raise ValueError("weights length must match reward_funcs length")

        events: list[RewardEvent] = []
        for i, reward_func in enumerate(self.reward_funcs):
            value = await reward_func.score(ctx)
            weight = self.weights[i] if self.weights is not None else 1.0
            for event in _coerce_events(value, source=reward_func.__class__.__name__):
                meta = {**event.meta}
                if weight != 1.0:
                    meta["raw_value"] = event.value
                    meta["weight"] = weight
                events.append(
                    RewardEvent(
                        ts=event.ts,
                        type=event.type,
                        source=event.source,
                        value=event.value * weight,
                        tag=event.tag,
                        step_index=event.step_index,
                        meta=meta,
                    )
                )
        return events


def _coerce_events(value: RewardValue, *, source: str) -> list[RewardEvent]:
    if isinstance(value, RewardEvent):
        return [value]
    if isinstance(value, list):
        return value
    return [RewardEvent(type="terminal", source=source, value=float(value))]
