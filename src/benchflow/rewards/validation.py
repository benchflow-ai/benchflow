"""Validation helpers for verifier-produced reward maps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

RewardValue = float | int
RewardMap = dict[str, Any]

_VALID_SPACES = {"output", "action", "reasoning", "memory", "latent"}
_VALID_GRANULARITIES = {"terminal", "step"}
_STRUCTURED_REWARD_KEYS = {
    "aggregate",
    "artifacts",
    "debug",
    "details",
    "details_path",
    "errors",
    "evidence",
    "items",
    "metadata",
    "participants",
    "raw",
    "reason",
    "reasons",
    "regressions",
    "winner",
}


def is_valid_reward_number(value: Any) -> bool:
    """Return True for finite scalar rewards in BenchFlow's [0, 1] range."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_reward_map(
    rewards: Mapping[str, Any] | None,
    *,
    source: str = "verifier",
    aggregate_policy: Mapping[str, Any] | None = None,
) -> RewardMap:
    """Validate and normalize a verifier reward mapping.

    The scalar ``reward`` remains the canonical training/eval score when
    present. Richer verifier packages may also return structured details such
    as ``metrics``, ``aggregate``, ``rubric``, or evidence payloads; those are
    preserved rather than flattened into scalar-only maps.
    """
    if rewards is None:
        raise ValueError(f"{source} returned no rewards")

    reward = rewards.get("reward")
    if "reward" in rewards and not is_valid_reward_number(reward):
        raise ValueError(
            f"{source} returned rewards without numeric 'reward': "
            "missing numeric 'reward' between 0.0 and 1.0"
        )

    parsed: RewardMap = {}
    has_numeric_metric = False
    for key, value in rewards.items():
        key = str(key)
        if key == "reward":
            parsed[key] = value
            continue
        if key == "rubric":
            parsed[key] = _validate_rubric(value, source=source)
            continue
        if key == "metrics":
            parsed[key] = _validate_metrics(value, source=source)
            has_numeric_metric = bool(parsed[key])
            continue
        if key == "space":
            parsed[key] = _validate_space(value, source=source)
            continue
        if key == "granularity":
            parsed[key] = _validate_granularity(value, source=source)
            continue
        if key == "aggregate":
            parsed[key] = _validate_aggregate(value, source=source)
            continue
        if key in _STRUCTURED_REWARD_KEYS:
            parsed[key] = value
            continue
        if not is_valid_reward_number(value):
            raise ValueError(
                f"{source} returned rewards with invalid reward value for {key!r}"
            )
        parsed[key] = value
        has_numeric_metric = True

    has_declared_aggregate = "aggregate" in parsed or bool(aggregate_policy)
    if "reward" not in parsed and (
        not has_declared_aggregate or not has_numeric_metric
    ):
        raise ValueError(
            f"{source} returned rewards without numeric 'reward': "
            "missing numeric 'reward' or aggregate policy for multi-metric rewards"
        )
    return parsed


def apply_aggregate_policy(
    rewards: Mapping[str, Any],
    *,
    aggregate_policy: Mapping[str, Any] | None = None,
    source: str = "verifier",
    strict: bool = False,
) -> RewardMap:
    """Compute canonical ``reward`` from metrics and a declared aggregate policy."""

    parsed = dict(rewards)
    if "reward" in parsed and not strict:
        return parsed

    metrics = _aggregate_metrics(parsed)
    if not metrics:
        raise ValueError(f"{source} has no metrics to aggregate into reward")

    reward_aggregate = parsed.get("aggregate")
    reward_policy = (
        dict(reward_aggregate) if isinstance(reward_aggregate, Mapping) else {}
    )
    document_policy = dict(aggregate_policy or {})

    field = document_policy.get("field") or reward_policy.get("primary") or "reward"
    if field != "reward":
        raise ValueError(
            f"{source} aggregate policy field must be 'reward' for runtime scoring"
        )

    method = (
        document_policy.get("method")
        or reward_policy.get("method")
        or document_policy.get("fallback")
    )
    if not isinstance(method, str) or not method:
        raise ValueError(f"{source} aggregate policy is missing method")

    expected_metrics = _aggregate_expected_metrics(document_policy, reward_policy)
    if expected_metrics is not None and set(metrics) != expected_metrics:
        missing = expected_metrics - set(metrics)
        extra = set(metrics) - expected_metrics
        parts: list[str] = []
        if missing:
            parts.append("missing metrics: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("extra metrics: " + ", ".join(sorted(extra)))
        raise ValueError(
            f"{source} metrics must match declared criteria exactly"
            + (": " + "; ".join(parts) if parts else "")
        )

    weights = document_policy.get("weights")
    if weights is None:
        weights = reward_policy.get("weights")
    threshold = document_policy.get("threshold")
    if threshold is None:
        threshold = reward_policy.get("threshold")
    reward = _compute_aggregate_reward(
        metrics,
        method=method,
        weights=weights,
        threshold=threshold,
        source=source,
    )
    if not is_valid_reward_number(reward):
        raise ValueError(
            f"{source} aggregate policy produced invalid reward {reward!r}"
        )
    if "reward" in parsed and strict:
        existing = parsed["reward"]
        if not is_valid_reward_number(existing):
            raise ValueError(
                f"{source} returned rewards without numeric 'reward': "
                "missing numeric 'reward' between 0.0 and 1.0"
            )
        if not math.isclose(float(existing), reward, abs_tol=1e-9):
            raise ValueError(
                f"{source} reward={existing} does not match criteria aggregate {reward}"
            )
    return {**parsed, "reward": reward}


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


def _validate_metrics(value: Any, *, source: str) -> dict[str, RewardValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} returned rewards with invalid value for 'metrics'")

    parsed: dict[str, RewardValue] = {}
    for key, metric in value.items():
        if not is_valid_reward_number(metric):
            raise ValueError(
                f"{source} returned rewards with invalid metric value for {str(key)!r}"
            )
        parsed[str(key)] = metric
    return parsed


def _validate_aggregate(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{source} returned rewards with invalid value for 'aggregate'"
        )
    parsed = {str(key): item for key, item in value.items()}
    method = parsed.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(
            f"{source} returned rewards with aggregate missing string 'method'"
        )
    weights = parsed.get("weights")
    if weights is not None:
        if not isinstance(weights, Mapping):
            raise ValueError(
                f"{source} returned rewards with invalid aggregate weights"
            )
        parsed["weights"] = {
            str(key): _validate_weight(weight, source=source, key=str(key))
            for key, weight in weights.items()
        }
    primary = parsed.get("primary")
    if primary is not None and not isinstance(primary, str):
        raise ValueError(f"{source} returned rewards with invalid aggregate primary")
    return parsed


def _validate_weight(value: Any, *, source: str, key: str) -> float | int:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(
            f"{source} returned rewards with invalid aggregate weight for {key!r}"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"{source} returned rewards with invalid aggregate weight for {key!r}"
        )
    return value


def _aggregate_metrics(rewards: Mapping[str, Any]) -> dict[str, RewardValue]:
    raw_metrics = rewards.get("metrics")
    if isinstance(raw_metrics, Mapping):
        return {
            str(key): metric
            for key, metric in raw_metrics.items()
            if is_valid_reward_number(metric)
        }
    metrics: dict[str, RewardValue] = {}
    for key, value in rewards.items():
        if key in {"aggregate", "reward"} | _STRUCTURED_REWARD_KEYS:
            continue
        if is_valid_reward_number(value):
            metrics[str(key)] = value
    return metrics


def _aggregate_expected_metrics(
    document_policy: Mapping[str, Any],
    reward_policy: Mapping[str, Any],
) -> set[str] | None:
    raw = document_policy.get("criteria")
    if raw is None:
        raw = reward_policy.get("criteria")
    if raw is None:
        return None
    if not isinstance(raw, list | tuple):
        raise ValueError("aggregate policy criteria must be a list")
    expected: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ValueError("aggregate policy criteria must be non-empty strings")
        expected.add(item)
    return expected


def _compute_aggregate_reward(
    metrics: Mapping[str, RewardValue],
    *,
    method: str,
    weights: Any,
    threshold: Any = None,
    source: str,
) -> float:
    normalized_method = method.replace("-", "_")
    if normalized_method == "mean":
        return sum(float(value) for value in metrics.values()) / len(metrics)

    if normalized_method == "all_pass":
        return 1.0 if all(float(value) >= 0.5 for value in metrics.values()) else 0.0

    if normalized_method == "any_pass":
        return 1.0 if any(float(value) >= 0.5 for value in metrics.values()) else 0.0

    if normalized_method not in {"weighted_mean", "weighted_sum", "threshold"}:
        raise ValueError(f"{source} aggregate policy method is unsupported: {method}")
    if normalized_method == "weighted_sum" and weights is None:
        raise ValueError(
            f"{source} aggregate policy method weighted_sum requires weights"
        )

    parsed_weights = _aggregate_weights(metrics, weights=weights, source=source)
    weighted_sum = sum(float(metrics[key]) * parsed_weights[key] for key in metrics)
    if normalized_method == "weighted_sum":
        return weighted_sum

    weight_total = sum(parsed_weights.values())
    if weight_total <= 0:
        raise ValueError(f"{source} aggregate policy weights must sum above zero")
    weighted_mean = weighted_sum / weight_total
    if normalized_method == "threshold":
        parsed_threshold = _aggregate_threshold(threshold, source=source)
        return 1.0 if weighted_mean >= parsed_threshold else 0.0
    return weighted_mean


def _aggregate_weights(
    metrics: Mapping[str, RewardValue],
    *,
    weights: Any,
    source: str,
) -> dict[str, float]:
    if weights is None:
        return {key: 1.0 for key in metrics}
    if not isinstance(weights, Mapping):
        raise ValueError(f"{source} aggregate policy weights must be a mapping")
    extra_weights = {str(key) for key in weights} - set(metrics)
    if extra_weights:
        raise ValueError(
            f"{source} aggregate policy has weights for unknown metrics: "
            + ", ".join(sorted(extra_weights))
        )
    parsed: dict[str, float] = {}
    for key in metrics:
        raw_weight = weights.get(key)
        if raw_weight is None:
            raise ValueError(
                f"{source} aggregate policy is missing weight for metric {key!r}"
            )
        if not isinstance(raw_weight, int | float) or isinstance(raw_weight, bool):
            raise ValueError(
                f"{source} aggregate policy has invalid weight for metric {key!r}"
            )
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"{source} aggregate policy has invalid weight for metric {key!r}"
            )
        parsed[key] = weight
    return parsed


def _aggregate_threshold(value: Any, *, source: str) -> float:
    if value is None:
        return 0.7
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{source} aggregate policy threshold must be numeric")
    threshold = float(value)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"{source} aggregate policy threshold must be between 0.0 and 1.0"
        )
    return threshold


def _validate_space(value: Any, *, source: str) -> str:
    if value not in _VALID_SPACES:
        raise ValueError(f"{source} returned rewards with invalid value for 'space'")
    return str(value)


def _validate_granularity(value: Any, *, source: str) -> str:
    if value not in _VALID_GRANULARITIES:
        raise ValueError(
            f"{source} returned rewards with invalid value for 'granularity'"
        )
    return str(value)
