"""Scores, bootstrap intervals and the noise gate for ``bench hillclimb``.

A split's input is ``{task: [value per scored trial]}``: rewards for the score
objective, USD per trial for the cost objective. Infrastructure errors are
not in the lists; they are counted apart.

- **Score**: the mean over tasks of each task's mean over its trials, so a
  task weighs the same however many of its trials were scored.
- **Standard error and interval**: a two-stage bootstrap. Each replicate
  draws the tasks with replacement, then each drawn task's trials with
  replacement, and takes the score. The standard error is the spread of the
  replicates; the interval is their 2.5th and 97.5th percentiles.
- **Delta between two surfaces**: the same bootstrap, paired. A replicate
  draws tasks once and resamples each side's trials of those tasks, so
  differences between tasks cancel and only what the change did remains.
- **Rerun noise**: how far the score moves between two runs of the *same*
  surface on the same tasks, by chance alone. A replicate draws tasks, then
  two independent resamples of each task's trials, and takes the difference.
  With few trials the resampled spread underestimates the true one by a
  factor ``(n - 1) / n``, so each task's trials are first spread out from
  their mean by ``sqrt(n / (n - 1))`` (the adjusted-residual bootstrap).

The noise gate compares ``--min-gain`` with the 95% band of the rerun noise,
``1.96 * rerun SE``: a patch with no effect clears a ``min_gain`` at least that
large on train less than 2.5% of the time by chance.

Everything is seeded and pure Python.

>>> est = bootstrap_score({"a": [1.0, 1.0], "b": [0.0, 0.0]}, samples=200, seed=0)
>>> est.value, est.tasks
(0.5, 2)
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

Z95 = 1.959963984540054
DEFAULT_SAMPLES = 2000

Values = Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high}


@dataclass(frozen=True)
class Estimate:
    """A score (or cost) with its bootstrap standard error and 95% interval."""

    value: float | None
    se: float | None
    ci: Interval | None
    tasks: int
    trials: int


@dataclass(frozen=True)
class Delta:
    """B minus A over the tasks both sides scored, with a paired interval."""

    value: float | None
    se: float | None
    ci: Interval | None
    paired_tasks: int
    samples: int


@dataclass(frozen=True)
class Noise:
    """How far a rerun of the same surface moves the score by chance."""

    se: float | None
    band95: float | None
    tasks: int
    min_trials: int
    reason: str | None = None


def _clean(values: Values) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for task, xs in values.items():
        kept = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
        if kept:
            out[task] = kept
    return out


def _mean(xs: Sequence[float]) -> float:
    return math.fsum(xs) / len(xs)


def score(values: Values) -> float | None:
    """Mean over tasks of the task means; None when no task has a value."""
    per_task = _clean(values)
    if not per_task:
        return None
    return _mean([_mean(xs) for xs in per_task.values()])


def _percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of already sorted values (0 <= q <= 1)."""
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def _summarize(reps: list[float]) -> tuple[float, Interval]:
    reps.sort()
    se = statistics.pstdev(reps) if len(reps) > 1 else 0.0
    return se, Interval(_percentile(reps, 0.025), _percentile(reps, 0.975))


def _resample_mean(rng: random.Random, xs: Sequence[float]) -> float:
    if len(xs) == 1:
        return xs[0]
    return math.fsum(rng.choices(xs, k=len(xs))) / len(xs)


def bootstrap_score(
    values: Values, *, samples: int = DEFAULT_SAMPLES, seed: int = 0
) -> Estimate:
    """The split's score with a two-stage (tasks, then trials) bootstrap."""
    per_task = _clean(values)
    tasks = sorted(per_task)
    trials = sum(len(v) for v in per_task.values())
    value = score(per_task)
    if value is None:
        return Estimate(None, None, None, 0, 0)
    if samples <= 0:
        return Estimate(value, None, None, len(tasks), trials)
    rng = random.Random(seed)
    lists = [per_task[t] for t in tasks]
    n = len(lists)
    reps = [
        math.fsum(_resample_mean(rng, lists[rng.randrange(n)]) for _ in range(n)) / n
        for _ in range(samples)
    ]
    se, ci = _summarize(reps)
    return Estimate(value, se, ci, len(tasks), trials)


def paired_delta(
    a: Values, b: Values, *, samples: int = DEFAULT_SAMPLES, seed: int = 0
) -> Delta:
    """``score(b) - score(a)`` over the tasks both scored, with a paired interval."""
    ca, cb = _clean(a), _clean(b)
    tasks = sorted(set(ca) & set(cb))
    if not tasks:
        return Delta(None, None, None, 0, samples)
    value = _mean([_mean(cb[t]) - _mean(ca[t]) for t in tasks])
    if samples <= 0:
        return Delta(value, None, None, len(tasks), 0)
    rng = random.Random(seed)
    pairs = [(ca[t], cb[t]) for t in tasks]
    n = len(pairs)
    reps = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            xa, xb = pairs[rng.randrange(n)]
            total += _resample_mean(rng, xb) - _resample_mean(rng, xa)
        reps.append(total / n)
    se, ci = _summarize(reps)
    return Delta(value, se, ci, len(tasks), samples)


def rerun_noise(
    values: Values, *, samples: int = DEFAULT_SAMPLES, seed: int = 0
) -> Noise:
    """The standard error of the score difference between two runs of the same
    surface on these tasks, from a two-stage bootstrap of a null difference.

    Needs at least two scored trials on every task: with one trial per task
    the trial-to-trial spread cannot be seen at all.
    """
    per_task = _clean(values)
    if not per_task:
        return Noise(None, None, 0, 0, "no task has a scored trial")
    min_trials = min(len(v) for v in per_task.values())
    if min_trials < 2:
        short = sorted(t for t, v in per_task.items() if len(v) < 2)
        return Noise(
            None,
            None,
            len(per_task),
            min_trials,
            f"{len(short)} task(s) have fewer than 2 scored trials "
            f"(e.g. {short[0]}), so trial-to-trial noise cannot be measured",
        )
    adjusted = []
    for xs in per_task.values():
        m = _mean(xs)
        k = math.sqrt(len(xs) / (len(xs) - 1))
        adjusted.append([m + (x - m) * k for x in xs])
    if samples <= 0:
        return Noise(None, None, len(adjusted), min_trials, "bootstrap disabled")
    rng = random.Random(seed)
    n = len(adjusted)
    reps = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            xs = adjusted[rng.randrange(n)]
            total += _resample_mean(rng, xs) - _resample_mean(rng, xs)
        reps.append(total / n)
    se = statistics.pstdev(reps)
    return Noise(se, Z95 * se, n, min_trials)


# ---------------------------------------------------------------------------
# The noise gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateSplit:
    split: str
    value: float | None
    se: float | None
    noise: Noise
    threshold: float
    ok: bool
    reason: str | None = None


@dataclass(frozen=True)
class Suggestion:
    trials: int | None = None
    train_tasks: int | None = None
    test_tasks: int | None = None
    min_gain: float | None = None


@dataclass(frozen=True)
class Gate:
    passed: bool
    objective: Literal["score", "cost"]
    min_gain: float
    train: GateSplit
    test: GateSplit
    message: str
    suggestion: Suggestion = field(default_factory=Suggestion)


def _gate_split(
    split: str,
    values: Values,
    *,
    min_gain: float,
    objective: str,
    samples: int,
    seed: int,
) -> GateSplit:
    est = bootstrap_score(values, samples=samples, seed=seed)
    noise = rerun_noise(values, samples=samples, seed=seed + 1)
    if noise.band95 is None:
        return GateSplit(split, est.value, est.se, noise, math.inf, False, noise.reason)
    band = noise.band95
    if objective == "cost":
        # min_gain is a fraction of the cost: compare like with like.
        if not est.value:
            return GateSplit(
                split,
                est.value,
                est.se,
                noise,
                math.inf,
                False,
                "no cost was recorded (the cost objective needs priced usage)",
            )
        band = band / est.value
    ok = min_gain >= band
    reason = (
        None
        if ok
        else (
            f"the 95% rerun noise is {_fmt(band, objective)}, above "
            f"--min-gain {_fmt(min_gain, objective)}"
        )
    )
    return GateSplit(split, est.value, est.se, noise, band, ok, reason)


def _fmt(x: float, objective: str) -> str:
    return f"{x:.1%}" if objective == "cost" else f"{x:.3f}"


def noise_gate(
    train: Values,
    test: Values,
    *,
    min_gain: float,
    objective: Literal["score", "cost"] = "score",
    trials: int,
    samples: int = DEFAULT_SAMPLES,
    seed: int = 0,
) -> Gate:
    """Whether ``min_gain`` is above the noise on both splits.

    ``train``/``test`` hold the baseline's per-task values (rewards, or USD
    per trial for ``objective="cost"``). ``trials`` is the ``--trials`` the
    baseline ran with; the suggestions scale it. Noise falls as one over the
    square root of trials times tasks, so the fix for a noise band ``r``
    times too wide is ``r**2`` times the trials or the tasks.
    """
    gs = [
        _gate_split(
            name,
            values,
            min_gain=min_gain,
            objective=objective,
            samples=samples,
            seed=seed + 7 * i,
        )
        for i, (name, values) in enumerate((("train", train), ("test", test)))
    ]
    g_train, g_test = gs
    passed = g_train.ok and g_test.ok
    unit = "of cost" if objective == "cost" else "reward points"
    if passed:
        message = (
            f"Noise gate passed: --min-gain {_fmt(min_gain, objective)} ({unit}) is at "
            f"least the 95% rerun noise on train ({_fmt(g_train.threshold, objective)}) "
            f"and test ({_fmt(g_test.threshold, objective)})."
        )
        return Gate(True, objective, min_gain, g_train, g_test, message)

    failing = [g for g in gs if not g.ok]
    if any(g.noise.band95 is None for g in failing):
        need_trials = max(trials, 3)
        message = (
            "Refusing to climb: "
            + "; ".join(f"{g.split}: {g.reason}" for g in failing)
            + f". Run with --trials {need_trials} or more so the noise can be "
            "measured."
        )
        return Gate(
            False,
            objective,
            min_gain,
            g_train,
            g_test,
            message,
            Suggestion(trials=need_trials),
        )
    worst = max(g.threshold / min_gain if min_gain > 0 else math.inf for g in failing)
    factor = worst**2
    suggestion = Suggestion(
        trials=math.ceil(trials * factor),
        train_tasks=math.ceil(g_train.noise.tasks * (g_train.threshold / min_gain) ** 2)
        if not g_train.ok and min_gain > 0
        else None,
        test_tasks=math.ceil(g_test.noise.tasks * (g_test.threshold / min_gain) ** 2)
        if not g_test.ok and min_gain > 0
        else None,
        min_gain=math.ceil(max(g.threshold for g in gs) * 1000) / 1000,
    )
    fixes = [f"--trials {suggestion.trials} (now {trials})"]
    if suggestion.train_tasks:
        fixes.append(
            f"about {suggestion.train_tasks} train tasks (now {g_train.noise.tasks})"
        )
    if suggestion.test_tasks:
        fixes.append(
            f"about {suggestion.test_tasks} test tasks (now {g_test.noise.tasks}; "
            "raise --test-frac or add tasks)"
        )
    fixes.append(f"--min-gain {_fmt(suggestion.min_gain or 0.0, objective)}")
    message = (
        "Refusing to climb: "
        + "; ".join(f"{g.split}: {g.reason}" for g in failing)
        + ". A patch with no effect would clear --min-gain by chance too often, so "
        "kept patches could be noise. Any one of these fixes it: "
        + "; ".join(fixes)
        + ". Pass --force to climb anyway (the run is then marked ungated)."
    )
    return Gate(False, objective, min_gain, g_train, g_test, message, suggestion)
