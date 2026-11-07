"""pass@k, pass^k and partial-credit solve rates over repeated trials.

``pass@k`` is the chance that at least one of ``k`` trials of a task
succeeds; ``pass^k`` is the chance that all ``k`` succeed. Both use the
unbiased estimators over ``n`` trials with ``c`` successes:

- pass@k = 1 - C(n - c, k) / C(n, k)   (Chen et al. 2021, "Codex")
- pass^k = C(c, k) / C(n, k)           (tau-bench)

Each needs ``n >= k`` for a task; a task with fewer trials is left out of
that ``k`` and the result says how many were left out (it is never
extrapolated). The job-level value is the mean over tasks.

Counting (as the Rust ``bench`` 0.8.2 counts): every trial is a
sample, so ``n`` is the number of trials. An unscored trial (agent error,
verifier error, missing reward) did not succeed: it counts as a fail in
``solve_rate``, ``pass@k`` and ``pass^k``, so an agent that errors never
scores above one that fails cleanly. The figures that leave unscored
trials out of ``n`` (the rule of BenchFlow 0.7's mean_reward) are kept beside
them as
``solve_rate_scored_only``, ``pass_at_k_scored_only`` and
``pass_hat_k_scored_only`` (``n`` = ``scored``), so numbers counted that
way stay comparable. Control runs (oracle, empty/nop) are left out by the
callers (:meth:`benchflow.Job.solve_rates`, :func:`benchflow.compare`).

Interval. Each solve rate comes with a 95% interval over its own ``n``.
With at most one trial per task it is the Wilson score interval over the
trials (``wilson``). With repeated trials, a task's trials are correlated,
so the interval is Wilson's on the effective sample size: the trials
divided by the design effect, the between-task variance of the solve rate
over its variance under independent trials (at least 1; with every trial
solved or none, the number of tasks) (``wilson-clustered``). It needs no
random resampling, so the same trials always give the same interval.

Success rule. By default a trial succeeds when it *passed*: reward = 1, or
the integrated-review gate verdict when the trial has one; this is the rule
behind every other pass count. For non-binary rewards pass a
``solve_threshold``: a scored trial then succeeds when ``reward >=
solve_threshold`` (the *partial-credit solve rate*). ``solve_rate`` is the
fraction of trials that succeed under the rule in force.

>>> round(pass_at_k(5, 2, 2), 3), round(pass_hat_k(5, 3, 2), 3)
(0.7, 0.3)
>>> pass_at_k(2, 1, 3) is None
True
>>> r = solve_rates([Sample("a", 1.0), Sample("a", None), Sample("b", 0.0)])
>>> r.solve_rate, r.trials, r.solve_rate_scored_only, r.scored
(0.3333333333333333, 3, 0.5, 2)
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
    "pass_at_k",
    "pass_hat_k",
    "solve_rate_interval",
    "solve_rates",
    "wilson_interval",
]

# How a reader counts the control runs a figure left out. The Python
# keyword; the CLI prints its flag instead (cli/results.py cli_wording).
CONTROLS_HINT = "include_controls=True counts them"


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
    """pass@k and pass^k at one ``k``; ``tasks`` had ``n >= k`` trials and
    are in the mean, ``tasks_short`` had fewer and are not (``n`` counts
    every trial, or only scored ones in ``SolveRates.at_k_scored_only``)."""

    k: int
    pass_at_k: float | None
    pass_hat_k: float | None
    tasks: int
    tasks_short: int


@dataclass(frozen=True)
class SolveRates:
    """Solve rates over repeated trials (see the module docstring).

    ``trials`` is the ``n`` of ``solve_rate`` and ``at_k`` (every trial;
    ``unscored`` of them got no score and count as fails); ``scored`` is the
    ``n`` of the ``*_scored_only`` figures, which leave unscored trials out.
    """

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
    # The solve rate's 95% interval and how it was computed (module docstring).
    interval: tuple[float, float] | None = None
    interval_method: str | None = None
    # The scored-only figures (0.7's mean_reward rule): scored trials only (``n`` = ``scored``, over
    # ``tasks_scored_only`` tasks with at least one scored trial).
    scored: int = 0
    tasks_scored_only: int = 0
    solve_rate_scored_only: float | None = None
    interval_scored_only: tuple[float, float] | None = None
    interval_method_scored_only: str | None = None
    at_k_scored_only: list[PassAtK] = field(default_factory=list)

    def get(self, k: int) -> PassAtK | None:
        """The pass@k / pass^k entry for ``k``, or None when it was not computed."""
        return next((p for p in self.at_k if p.k == k), None)

    def get_scored_only(self, k: int) -> PassAtK | None:
        """The scored-only pass@k / pass^k entry for ``k``, or None."""
        return next((p for p in self.at_k_scored_only if p.k == k), None)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (the ``solve_rates`` block of summary.json)."""
        return {
            "success_rule": self.success_rule,
            "solve_threshold": self.solve_threshold,
            "tasks": self.tasks,
            "trials": self.trials,
            "scored": self.scored,
            "unscored": self.unscored,
            "controls_excluded": self.controls_excluded,
            "min_trials_per_task": self.min_trials_per_task,
            "max_trials_per_task": self.max_trials_per_task,
            "solve_rate": self.solve_rate,
            "solve_rate_interval": list(self.interval) if self.interval else None,
            "solve_rate_interval_method": self.interval_method,
            "tasks_scored_only": self.tasks_scored_only,
            "solve_rate_scored_only": self.solve_rate_scored_only,
            "solve_rate_scored_only_interval": (
                list(self.interval_scored_only) if self.interval_scored_only else None
            ),
            "solve_rate_scored_only_interval_method": self.interval_method_scored_only,
            "nonbinary_rewards": self.nonbinary_rewards,
            "ks": [p.k for p in self.at_k],
            "pass_at_k": {str(p.k): p.pass_at_k for p in self.at_k},
            "pass_hat_k": {str(p.k): p.pass_hat_k for p in self.at_k},
            "tasks_at_k": {str(p.k): p.tasks for p in self.at_k},
            "tasks_short_at_k": {str(p.k): p.tasks_short for p in self.at_k},
            "pass_at_k_scored_only": {
                str(p.k): p.pass_at_k for p in self.at_k_scored_only
            },
            "pass_hat_k_scored_only": {
                str(p.k): p.pass_hat_k for p in self.at_k_scored_only
            },
            "tasks_at_k_scored_only": {
                str(p.k): p.tasks for p in self.at_k_scored_only
            },
            "tasks_short_at_k_scored_only": {
                str(p.k): p.tasks_short for p in self.at_k_scored_only
            },
            "caveats": list(self.caveats),
        }

    def rows(self) -> list[tuple[str, str]]:
        """``(label, value)`` rows: the solve rate and pass@k / pass^k at
        each ``k``, each with its ``n``. The scored-only rows follow only
        when some trial is unscored (otherwise they are the same numbers)."""
        if not self.trials and self.controls_excluded:
            # Every trial was a control run and was left out: each row says
            # so, not "no task has k trials" (DF-p3-eval-runner-3).
            gone = (
                f"n/a: the {self.controls_excluded} trial(s) are control runs "
                f"({CONTROLS_HINT})"
            )
            return [("Solve rate", gone)] + [(f"pass@{p.k}", gone) for p in self.at_k]
        both = self.unscored > 0
        spread = (
            f", n {self.min_trials_per_task}-{self.max_trials_per_task} per task"
            if self.max_trials_per_task > 1
            else ""
        )
        out = [
            (
                "Solve rate",
                _rate(self.solve_rate, self.trials, "trials", self.tasks, spread),
            )
        ]
        if both:
            out.append(
                (
                    "Solve rate, scored only",
                    _rate(
                        self.solve_rate_scored_only,
                        self.scored,
                        "scored trials",
                        self.tasks_scored_only,
                    ),
                )
            )
        for p in self.at_k:
            out.append((f"pass@{p.k}", _at_k_value(p, "trials")))
            q = self.get_scored_only(p.k) if both else None
            if q is not None:
                out.append(
                    (f"pass@{p.k}, scored only", _at_k_value(q, "scored trials"))
                )
        return out

    def lines(self) -> list[str]:
        """The pass@k / pass^k part of :meth:`rows` as text lines (markdown)."""
        return [
            f"{label}: {value}"
            for label, value in self.rows()
            if label.startswith("pass@")
        ]


def _rate(value: float | None, n: int, noun: str, tasks: int, extra: str = "") -> str:
    if value is None:
        return f"n/a (no {noun.removesuffix('s')})"
    return f"{value:.1%} ({round(value * n)} of {n} {noun}, {tasks} tasks{extra})"


def _at_k_value(p: PassAtK, noun: str) -> str:
    if p.pass_at_k is None and not p.tasks_short:
        # No task at all, not tasks with too few trials.
        return f"n/a (no {noun.removesuffix('s')})"
    if p.pass_at_k is None:
        return f"n/a (no task has {p.k} {noun})"
    short = f", {p.tasks_short} short" if p.tasks_short else ""
    return f"{p.pass_at_k:.1%} · pass^{p.k} {p.pass_hat_k:.1%} ({p.tasks} tasks{short})"


def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for ``successes`` out of ``n`` (``n > 0``;
    ``n`` may be an effective, non-integer sample size)."""
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def solve_rate_interval(
    counts: list[tuple[int, int]],
) -> tuple[tuple[float, float], str] | None:
    """The 95% interval of the pooled solve rate over ``(n, c)`` per task:
    ``(interval, method)``, or None without a trial (module docstring)."""
    counts = [(n, c) for n, c in counts if n > 0]
    total = sum(n for n, _ in counts)
    if not total:
        return None
    solved = sum(c for _, c in counts)
    if max(n for n, _ in counts) <= 1 or len(counts) < 2:
        return wilson_interval(solved, total), "wilson"
    p = solved / total
    tasks = len(counts)
    independent = p * (1 - p) / total
    if independent == 0:
        effective = float(tasks)  # every trial solved, or none
    else:
        clustered = (
            tasks / (tasks - 1) * math.fsum((c - p * n) ** 2 for n, c in counts)
        ) / (total * total)
        effective = total / max(1.0, clustered / independent)
    return wilson_interval(p * effective, effective), "wilson-clustered"


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pass_at_ks(counts: list[tuple[int, int]], wanted: list[int]) -> list[PassAtK]:
    """pass@k / pass^k at each ``k`` over ``(n, c)`` per task."""
    out: list[PassAtK] = []
    for k in wanted:
        eligible = [(n, c) for n, c in counts if n >= k]
        out.append(
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
                tasks_short=len(counts) - len(eligible),
            )
        )
    return out


def solve_rates(
    samples: Iterable[Sample],
    *,
    ks: Iterable[int] | None = None,
    solve_threshold: float | None = None,
    controls_excluded: int = 0,
) -> SolveRates:
    """Aggregate pass@k, pass^k and the solve rate over ``samples``.

    Every sample is a trial; an unscored one (reward None or not a finite
    number) counts as not solved, and the ``*_scored_only`` figures leave it
    out. Implicit ``ks`` are the union of :func:`default_ks` of the smallest
    per-task count in each population (all trials and scored-only trials).
    ``solve_threshold`` switches the success rule from
    "passed" to ``reward >= solve_threshold``.
    """
    if solve_threshold is not None:
        solve_threshold = float(solve_threshold)
        if not math.isfinite(solve_threshold):
            raise ValueError("solve_threshold must be a finite number")
    # Per task, one entry per trial: solved or not, None when unscored.
    per_task: dict[str, list[bool | None]] = {}
    unscored = 0
    nonbinary = 0
    # Rewards other than 0 or 1 that the reward rule failed; a rubric verdict decides its own trials.
    nonbinary_failed = 0
    for s in samples:
        reward = _finite(s.reward)
        if reward is None:
            unscored += 1
            per_task.setdefault(s.task, []).append(None)
            continue
        if reward not in (0.0, 1.0):
            nonbinary += 1
        if solve_threshold is not None:
            ok = reward >= solve_threshold
        elif s.passed is not None:
            ok = bool(s.passed)
        else:
            ok = reward == 1.0
            nonbinary_failed += reward not in (0.0, 1.0)
        per_task.setdefault(s.task, []).append(ok)

    # (n, c) per task: over every trial, and over scored trials only (a task
    # with no scored trial is not in the second).
    counts = [(len(v), sum(1 for x in v if x)) for v in per_task.values()]
    scored_counts: list[tuple[int, int]] = []
    for v in per_task.values():
        marks = [x for x in v if x is not None]
        if marks:
            scored_counts.append((len(marks), sum(marks)))
    min_n = min((n for n, _ in counts), default=0)
    max_n = max((n for n, _ in counts), default=0)
    if ks is not None:
        wanted = sorted(set(ks))
    else:
        # An all-unscored task must not erase historical, scored-only ks.
        # Use their union so both JSON and paired human rows retain them.
        scored_min = min((n for n, _ in scored_counts), default=0)
        wanted = sorted(set(default_ks(min_n)) | set(default_ks(scored_min)))
    if any(k < 1 for k in wanted):
        raise ValueError("every k must be >= 1")

    at_k = _pass_at_ks(counts, wanted)
    at_k_scored_only = _pass_at_ks(scored_counts, wanted)
    caveats: list[str] = []
    for p in at_k:
        if p.tasks_short:
            caveats.append(
                f"pass@{p.k}/pass^{p.k} cover {p.tasks} of {len(counts)} tasks: "
                f"{p.tasks_short} task(s) have fewer than {p.k} trials (n < k) and "
                "are left out; the unbiased estimator needs n >= k."
            )
    if unscored:
        caveats.append(
            f"{unscored} unscored trial(s) count as not solved; the scored-only "
            "figures leave them out of n, as BenchFlow 0.7's mean_reward did."
        )
        for p, q in zip(at_k, at_k_scored_only, strict=True):
            if q.tasks < p.tasks:
                fewer = (
                    "no scored trial" if q.k == 1 else f"fewer than {q.k} scored trials"
                )
                caveats.append(
                    f"pass@{q.k}/pass^{q.k} scored only cover {q.tasks} of "
                    f"{len(counts)} tasks: {len(counts) - q.tasks} task(s) have "
                    f"{fewer}."
                )
    if controls_excluded:
        caveats.append(
            f"{controls_excluded} control run(s) (oracle, empty/nop) are left out: "
            f"they check the task, not an agent ({CONTROLS_HINT})."
        )
    if nonbinary_failed and solve_threshold is None:
        caveats.append(
            f"{nonbinary_failed} scored trial(s) have a reward other than 0 or 1 and count "
            "as not passed; pass --solve-threshold (solve_threshold=) to count "
            "partial credit."
        )
    total = sum(n for n, _ in counts)
    scored = sum(n for n, _ in scored_counts)
    interval = solve_rate_interval(counts)
    interval_scored = solve_rate_interval(scored_counts)
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
        interval=interval[0] if interval else None,
        interval_method=interval[1] if interval else None,
        scored=scored,
        tasks_scored_only=len(scored_counts),
        solve_rate_scored_only=(
            (sum(c for _, c in scored_counts) / scored) if scored else None
        ),
        interval_scored_only=interval_scored[0] if interval_scored else None,
        interval_method_scored_only=interval_scored[1] if interval_scored else None,
        at_k_scored_only=at_k_scored_only,
    )
