"""Validation helpers for verifier-produced reward maps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

RewardValue = float | int
RewardMap = dict[str, Any]


def is_valid_reward_number(value: Any) -> bool:
    """Return True for finite scalar rewards in BenchFlow's [0, 1] range."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_reward_map(
    rewards: Mapping[str, Any] | None, *, source: str = "verifier"
) -> RewardMap:
    """Validate and normalize a verifier reward mapping."""
    if rewards is None:
        raise ValueError(f"{source} returned no rewards")

    reward = rewards.get("reward")
    if not is_valid_reward_number(reward):
        raise ValueError(
            f"{source} returned rewards without numeric 'reward' between 0.0 and 1.0"
        )

    parsed: RewardMap = {"reward": reward}
    for key, value in rewards.items():
        if key == "reward":
            continue
        if key == "rubric":
            parsed[str(key)] = _validate_rubric(value, source=source)
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            parsed[str(key)] = value
            continue
        if not is_valid_reward_number(value):
            raise ValueError(
                f"{source} returned rewards with invalid reward value for {str(key)!r}"
            )
        parsed[str(key)] = value
    return parsed


def _validate_rubric(value: Any, *, source: str) -> list[dict[str, Any]]:
    """Validate structured rubric/process reward details without flattening them."""
    if not isinstance(value, list):
        raise ValueError(f"{source} returned rewards with invalid value for 'rubric'")

    parsed: list[dict[str, Any]] = []
    for i, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"{source} returned rewards with invalid rubric item at index {i}"
            )
        rubric_item: dict[str, Any] = {str(k): v for k, v in item.items()}
        score = rubric_item.get("score")
        if not is_valid_reward_number(score):
            raise ValueError(
                f"{source} returned rewards with invalid rubric score at index {i}"
            )
        parsed.append(rubric_item)
    return parsed
