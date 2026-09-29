"""pass@k, pass^k and partial-credit solve rates over repeated trials.

``pass@k`` is the chance that at least one of ``k`` trials of a task
succeeds; ``pass^k`` is the chance that all ``k`` succeed. Both use the
unbiased estimators over ``n`` scored trials with ``c`` successes:

- pass@k = 1 - C(n - c, k) / C(n, k)   (Chen et al. 2021, "Codex")
- pass^k = C(c, k) / C(n, k)           (tau-bench)

Each needs ``n >= k`` for a task; a task with fewer scored trials is left out
of that ``k`` and the result says how many were left out (it is never
extrapolated). The job-level value is the mean over tasks.

Counting follows the rest of BenchFlow: only *scored* trials are samples.
An unscored trial (agent error, verifier error, missing reward) is not a
failure and does not enter ``n``; it is reported as ``unscored``. Control
runs (oracle, empty/nop) are left out by the callers
(:meth:`benchflow.Job.solve_rates`, :func:`benchflow.compare`).

Success rule. By default a trial succeeds when it *passed*: reward = 1, or
the integrated-review gate verdict when the trial has one; this is the rule
behind every other pass count. For non-binary rewards pass a
``solve_threshold``: a scored trial then succeeds when ``reward >=
solve_threshold`` (the *partial-credit solve rate*). ``solve_rate`` is the
fraction of scored trials that succeed under the rule in force.

>>> round(pass_at_k(5, 2, 2), 3), round(pass_hat_k(5, 3, 2), 3)
(0.7, 0.3)
>>> pass_at_k(2, 1, 3) is None
True
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PassAtK",
    "Sample",
    "SolveRates",
    "default_ks",
    "pass_at_1_interval",
    "pass_at_k",
    "pass_hat_k",
    "solve_rates",
]


def _check(n: int, c: int, k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got n={n}, c={c}")


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k for one task; None when ``n < k``."""
    _check(n, c, k)
    if n < k:
        return None
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass^k (all ``k`` succeed) for one task; None when ``n < k``."""
    _check(n, c, k)
    if n < k:
        return None
    return math.comb(c, k) / math.comb(n, k)


def default_ks(min_trials: int) -> list[int]:
    """1, powers of two and multiples of five up to ``min_trials``."""
    ks = {1}
    k = 2
    while k <= min_trials:
        ks.add(k)
        k *= 2
    ks.update(range(5, min_trials + 1, 5))
    return sorted(ks)


@dataclass(frozen=True)
class Sample:
    """One trial: its task, its reward (None = unscored) and, when known,
    the pass verdict (the review gate's, which wins over the reward under
    the default pass rule)."""

    task: str
    reward: float | None
    passed: bool | None = None


@dataclass(frozen=True)
class PassAtK:
    """pass@k and pass^k at one ``k``; ``tasks`` had ``n >= k`` scored
    trials and are in the mean, ``tasks_short`` had fewer and are not."""

    k: int
    pass_at_k: float | None
    pass_hat_k: float | None
    tasks: int
    tasks_short: int


@dataclass(frozen=True)
class SolveRates:
    """Solve rates over repeated trials (see the module docstring)."""

    success_rule: str
    solve_threshold: float | None
    tasks: int
    trials: int
    unscored: int
    solve_rate: float | None
    min_trials_per_task: int
    max_trials_per_task: int
    nonbinary_rewards: int
    at_k: list[PassAtK] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    controls_excluded: int = 0

    def get(self, k: int) -> PassAtK | None:
        return next((p for p in self.at_k if p.k == k), None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (the ``solve_rates`` block of summary.json)."""
        return {
            "success_rule": self.success_rule,
            "solve_threshold": self.solve_threshold,
            "tasks": self.tasks,
            "trials": self.trials,
            "unscored": self.unscored,
            "controls_excluded": self.controls_excluded,
            "min_trials_per_task": self.min_trials_per_task,
            "max_trials_per_task": self.max_trials_per_task,
            "solve_rate": self.solve_rate,
            "nonbinary_rewards": self.nonbinary_rewards,
            "ks": [p.k for p in self.at_k],
            "pass_at_k": {str(p.k): p.pass_at_k for p in self.at_k},
            "pass_hat_k": {str(p.k): p.pass_hat_k for p in self.at_k},
            "tasks_at_k": {str(p.k): p.tasks for p in self.at_k},
            "tasks_short_at_k": {str(p.k): p.tasks_short for p in self.at_k},
            "caveats": list(self.caveats),
        }

    def lines(self) -> list[str]:
        """Short human-readable lines (CLI tables and markdown)."""
        out = []
        for p in self.at_k:
            if p.pass_at_k is None:
                out.append(f"pass@{p.k}: n/a (no task has {p.k} scored trials)")
                continue
            short = f", {p.tasks_short} short" if p.tasks_short else ""
            out.append(
                f"pass@{p.k} {p.pass_at_k:.1%} · pass^{p.k} {p.pass_hat_k:.1%} "
                f"({p.tasks} tasks{short})"
            )
        return out


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def solve_rates(
    samples: Iterable[Sample],
    *,
    ks: Iterable[int] | None = None,
    solve_threshold: float | None = None,
    controls_excluded: int = 0,
) -> SolveRates:
    """Aggregate pass@k, pass^k and the solve rate over ``samples``.

    ``ks`` defaults to :func:`default_ks` of the smallest per-task count of
    scored trials. ``solve_threshold`` switches the success rule from
    "passed" to ``reward >= solve_threshold``.
    """
    if solve_threshold is not None:
        solve_threshold = float(solve_threshold)
        if not math.isfinite(solve_threshold):
            raise ValueError("solve_threshold must be a finite number")
    per_task: dict[str, list[bool]] = {}
    unscored = 0
    nonbinary = 0
    for s in samples:
        reward = _finite(s.reward)
        if reward is None:
            unscored += 1
            continue
        if reward not in (0.0, 1.0):
            nonbinary += 1
        if solve_threshold is not None:
            ok = reward >= solve_threshold
        elif s.passed is not None:
            ok = bool(s.passed)
        else:
            ok = reward == 1.0
        per_task.setdefault(s.task, []).append(ok)

    counts = [(len(v), sum(v)) for v in per_task.values()]
    min_n = min((n for n, _ in counts), default=0)
    max_n = max((n for n, _ in counts), default=0)
    wanted = sorted(set(ks)) if ks is not None else default_ks(min_n)
    if any(k < 1 for k in wanted):
        raise ValueError("every k must be >= 1")

    at_k: list[PassAtK] = []
    caveats: list[str] = []
    for k in wanted:
        eligible = [(n, c) for n, c in counts if n >= k]
        short = len(counts) - len(eligible)
        at_k.append(
            PassAtK(
                k=k,
                pass_at_k=(
                    math.fsum(pass_at_k(n, c, k) or 0.0 for n, c in eligible)
                    / len(eligible)
                    if eligible
                    else None
                ),
                pass_hat_k=(
                    math.fsum(pass_hat_k(n, c, k) or 0.0 for n, c in eligible)
                    / len(eligible)
                    if eligible
                    else None
                ),
                tasks=len(eligible),
                tasks_short=short,
            )
        )
        if short:
            caveats.append(
                f"pass@{k}/pass^{k} cover {len(eligible)} of {len(counts)} tasks: "
                f"{short} task(s) have fewer than {k} scored trials (n < k) and are "
                "left out; the unbiased estimator needs n >= k."
            )
    if unscored:
        caveats.append(
            f"{unscored} unscored trial(s) are left out of n, not counted as failures."
        )
    if nonbinary and solve_threshold is None:
        caveats.append(
            f"{nonbinary} scored trial(s) have a reward other than 0 or 1 and count "
            "as not passed; pass --solve-threshold (solve_threshold=) to count "
            "partial credit."
        )
    total = sum(n for n, _ in counts)
    return SolveRates(
        success_rule=(
            "passed (reward = 1)"
            if solve_threshold is None
            else f"reward >= {solve_threshold:g}"
        ),
        solve_threshold=solve_threshold,
        tasks=len(counts),
        trials=total,
        unscored=unscored,
        solve_rate=(sum(c for _, c in counts) / total) if total else None,
        min_trials_per_task=min_n,
        max_trials_per_task=max_n,
        nonbinary_rewards=nonbinary,
        at_k=at_k,
        caveats=caveats,
        controls_excluded=controls_excluded,
    )


def pass_at_1_interval(
    samples: Iterable[Sample],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
    solve_threshold: float | None = None,
) -> tuple[float, float] | None:
    """A percentile bootstrap interval for pass@1 over ``samples``.

    pass@1 is the mean over tasks of each task's solve rate. Each replicate
    draws the tasks with replacement, then each drawn task's scored trials
    with replacement, so the interval covers both which tasks were run and
    trial-to-trial noise. Samples are counted as in :func:`solve_rates`
    (unscored ones left out, the same success rule). None when no task has a
    scored trial. Seeded, so the same samples give the same interval.

    >>> pass_at_1_interval([Sample("a", 1.0), Sample("b", 1.0)])
    (1.0, 1.0)
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    per_task: dict[str, list[float]] = {}
    for s in samples:
        reward = _finite(s.reward)
        if reward is None:
            continue
        if solve_threshold is not None:
            ok = reward >= solve_threshold
        elif s.passed is not None:
            ok = bool(s.passed)
        else:
            ok = reward == 1.0
        per_task.setdefault(s.task, []).append(1.0 if ok else 0.0)
    tasks = list(per_task.values())
    if not tasks:
        return None
    import random

    rng = random.Random(seed)
    n = len(tasks)
    reps = []
    for _ in range(max(1, resamples)):
        total = 0.0
        for _ in range(n):
            xs = tasks[rng.randrange(n)]
            total += sum(rng.choices(xs, k=len(xs))) / len(xs)
        reps.append(total / n)
    reps.sort()
    tail = (1 - confidence) / 2
    lo = reps[int(tail * (len(reps) - 1))]
    hi = reps[round((1 - tail) * (len(reps) - 1))]
    return lo, hi
