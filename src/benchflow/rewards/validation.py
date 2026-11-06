"""Validation helpers for verifier-produced reward maps."""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable, Mapping
from typing import Any

RewardValue = float | int
RewardMap = dict[str, Any]
RewardRange = tuple[float, float]

# BenchFlow's canonical reward contract. A task may WIDEN it (never narrow or
# shift it) by declaring ``[verifier] reward_range`` — see
# :func:`validate_declared_reward_range`.
DEFAULT_REWARD_RANGE: RewardRange = (0.0, 1.0)

# How a trial's ``verifier_error`` starts when its verifier wrote a reward
# outside the task's range (#1178); ``classify_verifier_error`` keys
# the ``reward_out_of_range`` category on it.
REWARD_OUT_OF_RANGE = "reward out of range:"

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


class RewardOutOfRange(ValueError):
    """A verifier wrote a finite reward number outside the task's range (#1178).

    The trial is not scored, and the raw number is kept for the record
    (``key``, ``value`` and the ``reward_range`` it was checked against), so
    it can be read and regraded rather than lost in a parse error. BenchFlow
    never clamps it: a clamped reward would be a score the verifier did not
    give. Rust's ``bench`` 0.8.2 refuses the same numbers (its R302).
    """

    def __init__(
        self,
        key: str,
        value: float | int,
        reward_range: RewardRange | None,
        *,
        source: str = "verifier",
    ) -> None:
        self.key = key
        self.value = value
        self.reward_range: RewardRange = reward_range or DEFAULT_REWARD_RANGE
        self.declared = reward_range is not None
        self.source = source
        super().__init__(self.describe())

    def describe(self, where: str | None = None) -> str:
        """``<where>: <key> = <value> is outside [lo, hi]`` (``where`` defaults
        to the source the validator was given)."""
        lo, hi = self.reward_range
        whose = " (the task's reward_range)" if self.declared else ""
        return (
            f"{where or self.source}: {self.key} = {self.value!r} "
            f"is outside [{lo}, {hi}]{whose}"
        )


def is_out_of_range_reward(
    value: Any,
    *,
    reward_range: RewardRange | None = None,
) -> bool:
    """True for a finite number (not a bool) outside the task's reward range.

    Such a number is a reward the verifier meant but the task's contract does
    not allow, unlike a non-number or NaN, which is no reward at all.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    return math.isfinite(number) and not is_valid_reward_number(
        value, reward_range=reward_range
    )


def _non_finite_path(value: Any, path: str) -> str | None:
    """Where inside ``value`` the first non-finite number sits, or None.

    A kept field (an extra field beside the reward, or a structured key BenchFlow
    passes through) is written to the trial's JSON records as it is, and those
    writes refuse NaN and the infinities (``allow_nan=False``). Python's JSON
    decoder accepts ``NaN``, ``Infinity`` and an overflowing ``1e400``, so such a
    number has to be refused here, as a verifier problem, not when the result
    is published. An integer too big for a float counts too, as
    the result exporters write it as they write an infinity.
    """
    stack: list[tuple[Any, str]] = [(value, path)]
    while stack:
        item, where = stack.pop()
        if isinstance(item, bool):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                return where
        elif isinstance(item, int):
            try:
                float(item)
            except OverflowError:
                return where
        elif isinstance(item, Mapping):
            stack.extend((v, f"{where}.{k}") for k, v in reversed(item.items()))
        elif isinstance(item, list | tuple):
            stack.extend(
                (v, f"{where}[{i}]") for i, v in reversed(list(enumerate(item)))
            )
    return None


def is_valid_reward_number(
    value: Any,
    *,
    reward_range: RewardRange | None = None,
) -> bool:
    """Return True for finite scalar rewards within the task's reward range.

    The default range is BenchFlow's canonical [0, 1]; a task that declares
    ``[verifier] reward_range`` (validated at config parse by
    :func:`validate_declared_reward_range`) widens the accepted bounds.
    """
    lo, hi = reward_range or DEFAULT_REWARD_RANGE
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and lo <= float(value) <= hi
    )


def validate_declared_reward_range(value: Any) -> RewardRange:
    """Validate a task-declared ``[verifier] reward_range`` at config parse.

    A declared range may only WIDEN the canonical [0, 1] contract: it must be
    a ``[lo, hi]`` pair of finite numbers with ``lo < hi``, ``lo <= 0.0`` and
    ``hi >= 1.0``. Widen-only keeps the canonical anchors meaningful for every
    task — 0 ("did nothing") and 1 ("solved") stay valid rewards, so no-op and
    oracle baselines keep scoring and a range can never silently shift or
    narrow the contract. Safety-floor benchmarks, whose deliberate-violation
    floor sits *below* doing nothing, declare ``reward_range = [-1.0, 1.0]``.
    """
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int | float)
            for item in value
        )
    ):
        raise ValueError("reward_range must be a [lo, hi] pair of numbers")
    lo, hi = float(value[0]), float(value[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise ValueError("reward_range must be finite with lo < hi")
    if lo > 0.0 or hi < 1.0:
        raise ValueError(
            "reward_range may only widen the canonical [0.0, 1.0] contract "
            "(lo <= 0.0 and hi >= 1.0)"
        )
    return (lo, hi)


def reward_range_phrase(reward_range: RewardRange | None) -> str:
    """Render the accepted bounds for error messages (default: canonical)."""
    lo, hi = reward_range or DEFAULT_REWARD_RANGE
    return f"between {lo} and {hi}"


def declared_reward_range(task: Any) -> RewardRange | None:
    """Return ``task.config.verifier.reward_range`` as a tuple, or None.

    The config layer (``VerifierConfig.reward_range``) already validated a
    real task's declaration via :func:`validate_declared_reward_range`; this
    accessor only recognizes that validated 2-sequence shape, so Any-typed
    task fakes (e.g. MagicMock in tests) and legacy task objects without the
    field read as "undeclared" → the canonical [0, 1] contract.
    """
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    raw = getattr(verifier, "reward_range", None)
    if isinstance(raw, list | tuple) and len(raw) == 2:
        return (float(raw[0]), float(raw[1]))
    return None


def reward_lenient_from_env() -> bool:
    """Return True when ``BENCHFLOW_REWARD_LENIENT`` opts into lenient parsing.

    Lenient mode keeps the classic ``test.sh → reward.json`` flow working when a
    verifier emits a rich reward map carrying extra bookkeeping keys (e.g. the
    older ``{"reward": 1.0, "done": true, ...}``) or non-numeric metrics:
    unrecognized/non-numeric top-level keys and non-numeric metric entries are
    dropped with a single warning instead of failing the whole run. A usable
    scalar ``reward`` is still required (or derived from a ``score``/``rewards``
    alias / declared aggregate policy). Unset, empty, or ``0`` keeps the default
    strict contract — there is no behaviour change unless the operator opts in.
    """
    return os.environ.get(_REWARD_LENIENT_ENV, "").strip().lower() in _TRUTHY


def _lenient_reward_alias(
    rewards: Mapping[str, Any],
    *,
    reward_range: RewardRange | None,
) -> str | None:
    """Return the first legacy alias key holding a usable scalar reward."""
    for alias in _LENIENT_REWARD_ALIASES:
        if alias in rewards and is_valid_reward_number(
            rewards[alias], reward_range=reward_range
        ):
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
        # A recognized structure may retain metadata beside its validated
        # fields. Check all of it before publication, not only its scores
        # (CONF-V4-1); lenient mode drops the same whole key as other errors.
        bad = _non_finite_path(value, key)
        if bad is not None:
            raise ValueError(
                f"{source} returned rewards with a non-finite number "
                f"at {bad}: kept fields must be finite"
            )
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
    reward_range: RewardRange | None = None,
) -> RewardMap:
    """Validate and normalize a verifier reward mapping.

    The scalar ``reward`` remains the canonical training/eval score when
    present. Richer verifier packages may also return structured details such
    as ``metrics``, ``aggregate``, ``rubric``, or evidence payloads; those are
    preserved rather than flattened into scalar-only maps.

    Beside a numeric ``reward``, a top-level key that holds no number (a
    bool, a string, null, a list or an object: ``"feature1_passed": true``,
    ``"answer": "Empowering"``) is kept as written, in both modes (#1178):
    it is the verifier's own record of the trial, and throwing the whole
    reward away for it turned correct runs into verifier errors. Without a
    numeric ``reward`` (a flat map of metrics) such a key still fails the
    strict contract, as in Rust's ``bench``.

    With ``lenient=False`` (the default) the strict contract applies
    otherwise: a non-numeric metric, a malformed structured key, or (without
    a ``reward``) an unrecognized non-numeric top-level key raises and fails
    the run.

    With ``lenient=True`` the classic ``test.sh → reward.json`` flow keeps
    working when a verifier emits non-numeric metrics or malformed structured
    keys. Those are dropped, the non-numeric metric *entries* are pruned from
    ``metrics``, and a single :func:`warnings.warn` lists everything dropped
    instead of raising. A usable scalar ``reward`` is still required: it is
    taken from ``reward`` when valid, otherwise derived from a numeric
    ``score``/``rewards`` alias, otherwise from numeric metrics plus a declared
    aggregate policy. The operator opts in via ``BENCHFLOW_REWARD_LENIENT=1``
    (see :func:`reward_lenient_from_env`).

    ``reward_range`` is the task's declared reward contract (see
    :func:`validate_declared_reward_range`); ``None`` keeps the canonical
    strict [0, 1]. The range applies to every reward-valued number in the map
    — the scalar ``reward``, top-level numeric metrics, and ``metrics``
    entries — but NOT to ``rubric`` item scores, which are normalized
    judge-criterion scores and always stay in [0, 1]. A finite number outside
    it raises :class:`RewardOutOfRange` (a ``ValueError``) naming the key and
    the raw value: the trial is unscored, never clamped. Lenient
    mode raises it for the scalar ``reward`` too, rather than dropping the
    score and reading a ``score`` alias in its place; it still drops an
    out-of-range metric.
    """
    if rewards is None:
        raise ValueError(f"{source} returned no rewards")

    if is_out_of_range_reward(rewards.get("reward"), reward_range=reward_range):
        raise RewardOutOfRange("reward", rewards["reward"], reward_range, source=source)

    working: Mapping[str, Any] = rewards
    dropped: list[str] = []

    if lenient:
        # Operate on a copy so we can re-home a legacy alias onto ``reward``
        # and drop an unusable scalar without mutating the caller's mapping.
        mutable = dict(rewards)
        explicit = mutable.get("reward")
        if (
            isinstance(explicit, int | float)
            and not isinstance(explicit, bool)
            and _non_finite_path(explicit, "reward")
        ):
            # NaN or an infinity is a number the verifier wrote, not a missing
            # one: no ``score`` alias stands in for it.
            raise ValueError(
                f"{source} returned rewards without numeric 'reward': "
                f"reward = {explicit!r} is not a finite number"
            )
        if "reward" in mutable and not is_valid_reward_number(
            mutable["reward"], reward_range=reward_range
        ):
            dropped.append("reward")
            del mutable["reward"]
        if "reward" not in mutable:
            alias = _lenient_reward_alias(mutable, reward_range=reward_range)
            if alias is not None:
                mutable["reward"] = mutable.pop(alias)
        working = mutable

    reward = working.get("reward")
    if "reward" in working and not is_valid_reward_number(
        reward, reward_range=reward_range
    ):
        # Unreachable in lenient mode (an unusable ``reward`` is dropped above).
        raise ValueError(
            f"{source} returned rewards without numeric 'reward': "
            f"missing numeric 'reward' {reward_range_phrase(reward_range)}"
        )

    has_reward = "reward" in working
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
                metrics = _lenient_metrics(
                    value, dropped=dropped, reward_range=reward_range
                )
                if metrics is not None:
                    parsed[key] = metrics
                    has_numeric_metric = has_numeric_metric or bool(metrics)
                continue
            parsed[key] = _validate_metrics(
                value, source=source, reward_range=reward_range
            )
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
        if key in _STRUCTURED_REWARD_KEYS or (
            has_reward
            and (not isinstance(value, int | float) or isinstance(value, bool))
        ):
            # A passed-through structured key, or an extra field beside the
            # reward (#1178): kept as written, never scored, so it must hold
            # nothing the trial's JSON records cannot write.
            bad = _non_finite_path(value, key)
            if bad is not None:
                if lenient:
                    dropped.append(key)
                    continue
                raise ValueError(
                    f"{source} returned rewards with a non-finite number "
                    f"at {bad}: kept fields must be finite"
                )
            parsed[key] = value
            continue
        if not is_valid_reward_number(value, reward_range=reward_range):
            if lenient:
                dropped.append(key)
                continue
            if is_out_of_range_reward(value, reward_range=reward_range):
                raise RewardOutOfRange(key, value, reward_range, source=source)
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
    reward_range: RewardRange | None,
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
        if is_valid_reward_number(metric, reward_range=reward_range):
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
    reward_range: RewardRange | None = None,
) -> RewardMap:
    """Compute canonical ``reward`` from metrics and a declared aggregate policy.

    ``reward_range`` widens which metric values are aggregable and which
    computed rewards are acceptable (default: canonical [0, 1]); see
    :func:`validate_reward_map`.
    """

    parsed = dict(rewards)
    if "reward" in parsed and not strict:
        return parsed

    metrics = _aggregate_metrics(parsed, reward_range=reward_range)
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
    if not is_valid_reward_number(reward, reward_range=reward_range):
        raise ValueError(
            f"{source} aggregate policy produced invalid reward {reward!r}"
        )
    if "reward" in parsed and strict:
        existing = parsed["reward"]
        if not is_valid_reward_number(existing, reward_range=reward_range):
            raise ValueError(
                f"{source} returned rewards without numeric 'reward': "
                f"missing numeric 'reward' {reward_range_phrase(reward_range)}"
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


def _validate_metrics(
    value: Any,
    *,
    source: str,
    reward_range: RewardRange | None = None,
) -> dict[str, RewardValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} returned rewards with invalid value for 'metrics'")

    parsed: dict[str, RewardValue] = {}
    for key, metric in value.items():
        if is_out_of_range_reward(metric, reward_range=reward_range):
            raise RewardOutOfRange(
                f"metrics.{key}", metric, reward_range, source=source
            )
        if not is_valid_reward_number(metric, reward_range=reward_range):
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


def _aggregate_metrics(
    rewards: Mapping[str, Any],
    *,
    reward_range: RewardRange | None = None,
) -> dict[str, RewardValue]:
    raw_metrics = rewards.get("metrics")
    if isinstance(raw_metrics, Mapping):
        return {
            str(key): metric
            for key, metric in raw_metrics.items()
            if is_valid_reward_number(metric, reward_range=reward_range)
        }
    metrics: dict[str, RewardValue] = {}
    for key, value in rewards.items():
        if key in {"aggregate", "reward"} | _STRUCTURED_REWARD_KEYS:
            continue
        if is_valid_reward_number(value, reward_range=reward_range):
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
