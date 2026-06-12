"""Validation helpers for verifier-produced reward maps."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable, Mapping
from typing import Any

RewardValue = float | int
RewardMap = dict[str, Any]

# Operator toggle for the lenient reward-map path. Default (unset/empty/``0``)
# keeps the strict contract; any truthy value opts the classic
# ``test.sh → reward.json`` flow into lenient parsing. Lives here next to the
# validator so the toggle is documented alongside the behaviour it controls.
_REWARD_LENIENT_ENV = "BENCHFLOW_REWARD_LENIENT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Legacy aliases a lenient reward map may carry instead of a scalar ``reward``.
# Checked in order; the first numeric match becomes the canonical ``reward``.
_LENIENT_REWARD_ALIASES = ("score", "rewards")

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


def reward_lenient_from_env() -> bool:
    """Return True when ``BENCHFLOW_REWARD_LENIENT`` opts into lenient parsing.

    Lenient mode keeps the classic ``test.sh → reward.json`` flow working when a
    verifier emits a rich reward map carrying extra bookkeeping keys (e.g. the
    Harbor-era ``{"reward": 1.0, "done": true, ...}``) or non-numeric metrics:
    unrecognized/non-numeric top-level keys and non-numeric metric entries are
    dropped with a single warning instead of failing the whole run. A usable
    scalar ``reward`` is still required (or derived from a ``score``/``rewards``
    alias / declared aggregate policy). Unset, empty, or ``0`` keeps the default
    strict contract — there is no behaviour change unless the operator opts in.
    """
    return os.environ.get(_REWARD_LENIENT_ENV, "").strip().lower() in _TRUTHY


def _lenient_reward_alias(rewards: Mapping[str, Any]) -> str | None:
    """Return the first legacy alias key holding a usable scalar reward."""
    for alias in _LENIENT_REWARD_ALIASES:
        if alias in rewards and is_valid_reward_number(rewards[alias]):
            return alias
    return None


def _apply_structured(
    parsed: RewardMap,
    key: str,
    value: Any,
    validate: Callable[..., Any],
    *,
    source: str,
    lenient: bool,
    dropped: list[str],
) -> None:
    """Validate a recognized structured key, dropping it in lenient mode.

    Strict mode re-raises the validator's ``ValueError``; lenient mode records
    the malformed key so it can be reported in the single aggregated warning.
    """
    try:
        parsed[key] = validate(value, source=source)
    except ValueError:
        # The assignment never ran, so ``parsed`` is untouched; just record it.
        if not lenient:
            raise
        dropped.append(key)


def validate_reward_map(
    rewards: Mapping[str, Any] | None,
    *,
    source: str = "verifier",
    aggregate_policy: Mapping[str, Any] | None = None,
    lenient: bool = False,
) -> RewardMap:
    """Validate and normalize a verifier reward mapping.

    The scalar ``reward`` remains the canonical training/eval score when
    present. Richer verifier packages may also return structured details such
    as ``metrics``, ``aggregate``, ``rubric``, or evidence payloads; those are
    preserved rather than flattened into scalar-only maps.

    With ``lenient=False`` (the default) the strict contract applies: any
    unrecognized non-numeric top-level key, or a non-numeric metric, raises and
    fails the run.

    With ``lenient=True`` the classic ``test.sh → reward.json`` flow keeps
    working when a verifier emits extra bookkeeping keys (e.g. ``done``) or
    non-numeric metrics. Such keys — and any malformed recognized-structured
    key — are dropped, the non-numeric metric *entries* are pruned from
    ``metrics``, and a single :func:`warnings.warn` lists everything dropped
    instead of raising. A usable scalar ``reward`` is still required: it is
    taken from ``reward`` when valid, otherwise derived from a numeric
    ``score``/``rewards`` alias, otherwise from numeric metrics plus a declared
    aggregate policy. The operator opts in via ``BENCHFLOW_REWARD_LENIENT=1``
    (see :func:`reward_lenient_from_env`).
    """
    if rewards is None:
        raise ValueError(f"{source} returned no rewards")

    working: Mapping[str, Any] = rewards
    dropped: list[str] = []

    if lenient:
        # Operate on a copy so we can re-home a legacy alias onto ``reward``
        # and drop an unusable scalar without mutating the caller's mapping.
        mutable = dict(rewards)
        if "reward" in mutable and not is_valid_reward_number(mutable["reward"]):
            dropped.append("reward")
            del mutable["reward"]
        if "reward" not in mutable:
            alias = _lenient_reward_alias(mutable)
            if alias is not None:
                mutable["reward"] = mutable.pop(alias)
        working = mutable

    reward = working.get("reward")
    if "reward" in working and not is_valid_reward_number(reward):
        # Unreachable in lenient mode (an unusable ``reward`` is dropped above).
        raise ValueError(
            f"{source} returned rewards without numeric 'reward': "
            "missing numeric 'reward' between 0.0 and 1.0"
        )

    parsed: RewardMap = {}
    has_numeric_metric = False
    for key, value in working.items():
        key = str(key)
        if key == "reward":
            parsed[key] = value
            continue
        if key == "rubric":
            _apply_structured(
                parsed,
                key,
                value,
                _validate_rubric,
                source=source,
                lenient=lenient,
                dropped=dropped,
            )
            continue
        if key == "metrics":
            if lenient:
                metrics = _lenient_metrics(value, dropped=dropped)
                if metrics is not None:
                    parsed[key] = metrics
                    has_numeric_metric = has_numeric_metric or bool(metrics)
                continue
            parsed[key] = _validate_metrics(value, source=source)
            has_numeric_metric = bool(parsed[key])
            continue
        if key == "space":
            _apply_structured(
                parsed,
                key,
                value,
                _validate_space,
                source=source,
                lenient=lenient,
                dropped=dropped,
            )
            continue
        if key == "granularity":
            _apply_structured(
                parsed,
                key,
                value,
                _validate_granularity,
                source=source,
                lenient=lenient,
                dropped=dropped,
            )
            continue
        if key == "aggregate":
            _apply_structured(
                parsed,
                key,
                value,
                _validate_aggregate,
                source=source,
                lenient=lenient,
                dropped=dropped,
            )
            continue
        if key in _STRUCTURED_REWARD_KEYS:
            parsed[key] = value
            continue
        if not is_valid_reward_number(value):
            if lenient:
                dropped.append(key)
                continue
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

    if lenient and dropped:
        warnings.warn(
            f"{source} reward map parsed in lenient mode; dropped "
            f"unrecognized/non-numeric keys: {', '.join(dropped)}",
            stacklevel=2,
        )
    return parsed


def _lenient_metrics(
    value: Any,
    *,
    dropped: list[str],
) -> dict[str, RewardValue] | None:
    """Prune a ``metrics`` mapping to numeric entries for lenient parsing.

    Returns the kept numeric metrics (possibly empty) and records each pruned
    entry as ``metrics.<key>``. A non-mapping ``metrics`` value is dropped
    wholesale (recorded as ``metrics``) and ``None`` is returned to signal the
    key should not appear in the parsed map.
    """
    if not isinstance(value, Mapping):
        dropped.append("metrics")
        return None
    kept: dict[str, RewardValue] = {}
    for key, metric in value.items():
        if is_valid_reward_number(metric):
            kept[str(key)] = metric
        else:
            dropped.append(f"metrics.{key}")
    return kept


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
