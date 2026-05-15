"""Composable Rubric — weighted collection of RewardFuncs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from benchflow.rewards.events import RewardEvent
from benchflow.rewards.protocol import RewardFunc, VerifyResult


@dataclass
class Rubric:
    """Weighted collection of reward functions.

    When ``weights`` is None every function is weighted equally.
    """

    reward_funcs: list[RewardFunc]
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        if self.weights is not None and len(self.weights) != len(self.reward_funcs):
            raise ValueError(
                f"weights length ({len(self.weights)}) != "
                f"reward_funcs length ({len(self.reward_funcs)})"
            )

    async def score(self, rollout_dir: Path) -> VerifyResult:
        """Run all reward functions and return weighted result."""
        n = len(self.reward_funcs)
        if n == 0:
            return VerifyResult(reward=0.0)

        weights = self.weights if self.weights is not None else [1.0 / n] * n

        items: dict[str, float] = {}
        events: list[RewardEvent] = []
        errors: list[str] = []

        results = await asyncio.gather(
            *(func.score(rollout_dir) for func in self.reward_funcs),
            return_exceptions=True,
        )

        weighted_sum = 0.0
        name_counts: dict[str, int] = {}
        for func, weight, result in zip(
            self.reward_funcs, weights, results, strict=True
        ):
            base_name = type(func).__name__
            count = name_counts.get(base_name, 0)
            name_counts[base_name] = count + 1
            name = f"{base_name}_{count}" if count > 0 else base_name
            if isinstance(result, BaseException):
                errors.append(f"{name}: {result}")
                items[name] = 0.0
                continue
            score = float(result)
            items[name] = score
            weighted_sum += weight * score
            events.append(
                RewardEvent(
                    type="terminal",
                    reward=score,
                    source=name,
                )
            )

        return VerifyResult(
            reward=weighted_sum,
            items=items,
            events=events,
            error="; ".join(errors) if errors else None,
        )
