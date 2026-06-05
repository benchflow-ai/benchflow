"""Validation helpers for verifier-produced reward maps."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RewardValue = float | int
RewardMap = dict[str, Any]

# Top-level reward JSON keys that are not scalar metrics in [0, 1].
RESERVED_REWARD_KEYS = frozenset(
    {
        "reward",
        "rubric",
        "items",
        "evidence",
        "artifacts",
        "metadata",
        "reason",
        "reasons",
        "errors",
        "metrics",
        "regressions",
        "participants",
        "winner",
        "raw",
        "debug",
        "aggregate_policy",
    }
)


def is_valid_reward_number(value: Any) -> bool:
    """Return True for finite scalar rewards in BenchFlow's [0, 1] range."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


class RewardFileParseError(ValueError):
    """Raised when verifier reward files cannot be parsed or disagree."""


def parse_verifier_reward_files(
    *,
    reward_text_path: Path,
    reward_json_path: Path,
    source: str = "verifier",
) -> RewardMap:
    """Parse verifier reward outputs with JSON-first precedence."""
    has_json = reward_json_path.exists()
    has_text = reward_text_path.exists()

    if has_json and has_text:
        json_rewards = _parse_reward_json_file(reward_json_path, source=source)
        text_rewards = _parse_reward_text_file(reward_text_path, source=source)
        json_scalar = float(json_rewards["reward"])
        text_scalar = float(text_rewards["reward"])
        if json_scalar != text_scalar:
            raise RewardFileParseError(
                "reward.json aggregate "
                f"{json_scalar} disagrees with reward.txt scalar {text_scalar}"
            )
        return json_rewards

    if has_json:
        return _parse_reward_json_file(reward_json_path, source=source)
    if has_text:
        return _parse_reward_text_file(reward_text_path, source=source)

    raise RewardFileParseError(
        f"No reward file found at {reward_text_path} or {reward_json_path}"
    )


def _parse_reward_text_file(path: Path, *, source: str) -> RewardMap:
    if path.stat().st_size == 0:
        raise RewardFileParseError(f"Reward file is empty at {path}")
    text = path.read_text().strip()
    if not text:
        raise RewardFileParseError(f"Reward file is empty at {path}")
    try:
        reward = float(text.splitlines()[0].strip())
    except (ValueError, TypeError, IndexError) as exc:
        raise RewardFileParseError(f"Failed to parse rewards from text file {path}") from exc
    if not is_valid_reward_number(reward):
        raise RewardFileParseError(
            f"Reward text file {path} must contain a finite numeric reward "
            "between 0.0 and 1.0"
        )
    return {"reward": reward}


def _parse_reward_json_file(path: Path, *, source: str) -> RewardMap:
    if path.stat().st_size == 0:
        raise RewardFileParseError(f"Reward file is empty at {path}")
    try:
        rewards = json.loads(path.read_text())
    except (ValueError, TypeError) as exc:
        raise RewardFileParseError(f"Failed to parse rewards from JSON file {path}") from exc

    if not isinstance(rewards, dict):
        raise RewardFileParseError(
            f"Reward JSON file {path} must contain an object with numeric rewards"
        )

    try:
        return validate_reward_map(rewards, source=source)
    except ValueError as exc:
        raise RewardFileParseError(f"Reward JSON file {path} {exc}") from exc


def _resolve_canonical_reward(
    rewards: Mapping[str, Any],
    metric_keys: list[str],
    *,
    source: str,
) -> float:
    """Resolve the scalar aggregate from explicit or multi-metric reward maps."""
    if "reward" in rewards:
        explicit = rewards.get("reward")
        if not is_valid_reward_number(explicit):
            raise ValueError(
                f"{source} returned rewards with invalid reward value for 'reward'"
            )
        assert isinstance(explicit, int | float)
        return float(explicit)

    if not metric_keys:
        raise ValueError(
            f"{source} returned rewards missing numeric 'reward' between 0.0 and 1.0"
        )

    aggregate_policy = rewards.get("aggregate_policy")
    if isinstance(aggregate_policy, Mapping):
        field = aggregate_policy.get("field")
        if field is not None:
            field_name = str(field)
            selected = rewards.get(field_name)
            if not is_valid_reward_number(selected):
                raise ValueError(
                    f"{source} returned rewards with aggregate_policy.field "
                    f"{field_name!r} that is not a numeric reward between 0.0 and 1.0"
                )
            assert isinstance(selected, int | float)
            return float(selected)

    values = [float(rewards[key]) for key in metric_keys]
    return sum(values) / len(values)


def validate_reward_map(
    rewards: Mapping[str, Any] | None, *, source: str = "verifier"
) -> RewardMap:
    """Validate and normalize a verifier reward mapping."""
    if rewards is None:
        raise ValueError(f"{source} returned no rewards")

    parsed: RewardMap = {}
    metric_keys: list[str] = []

    for key, value in rewards.items():
        key_str = str(key)
        if key_str == "rubric":
            parsed[key_str] = _validate_rubric(value, source=source)
            continue
        if key_str in RESERVED_REWARD_KEYS:
            if key_str != "reward":
                parsed[key_str] = value
            continue
        if not is_valid_reward_number(value):
            raise ValueError(
                f"{source} returned rewards with invalid reward value for {key_str!r}"
            )
        parsed[key_str] = value
        metric_keys.append(key_str)

    parsed["reward"] = _resolve_canonical_reward(
        rewards, metric_keys, source=source
    )
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
