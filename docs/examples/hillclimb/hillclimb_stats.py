"""Scores with bootstrap intervals, and the noise gate, for the hill-climb demo.

Input everywhere is one split's ``{task: [1.0 or 0.0 per scored trial]}``
(passed or not, BenchFlow's solve-rate rule). A split's score is the mean over
tasks of each task's solve rate, which is pass@1, the same number
``bf.load_job(...).solve_rates(ks=[1])`` reports. BenchFlow gives no interval
for it, so this file adds three bootstraps:

- **score interval**: draw the tasks with replacement, then each drawn task's
  trials with replacement; the 2.5th and 97.5th percentiles of the replicates;
- **paired delta** between two versions: draw the tasks once, resample each
  side's trials of those tasks, so differences between tasks cancel;
- **rerun noise**: how far the score moves between two runs of the *same*
  version, by chance alone. Draw the tasks, then two independent resamples of
  each task's trials, and take the difference. Each task's trials are first
  spread from their mean by ``sqrt(n / (n - 1))``, since a resample of ``n``
  trials underestimates their spread by ``(n - 1) / n``.

The noise gate asks for ``min_gain >= 1.96 x`` the rerun noise's standard error
on both splits: a patch with no effect then clears ``min_gain`` on train by
chance less than 2.5% of the time. Noise falls as one over the square root of
trials times tasks, so a band ``r`` times too wide needs ``r**2`` times either.
"""

from __future__ import annotations

import math
import random
import statistics

Z95 = 1.959963984540054
Values = dict[str, list[float]]


def _mean(xs: list[float]) -> float:
    return math.fsum(xs) / len(xs)


def _clean(values: Values) -> Values:
    return {task: xs for task, xs in values.items() if xs}


def score(values: Values) -> float | None:
    """Mean over tasks of the task means (pass@1 for 0/1 values)."""
    per_task = _clean(values)
    return _mean([_mean(xs) for xs in per_task.values()]) if per_task else None


def _resample(rng: random.Random, xs: list[float]) -> float:
    return xs[0] if len(xs) == 1 else _mean(rng.choices(xs, k=len(xs)))


def _summary(value: float | None, reps: list[float]) -> dict:
    reps.sort()

    def pct(q: float) -> float:
        pos = q * (len(reps) - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        return reps[lo] + (reps[hi] - reps[lo]) * (pos - lo)

    return {
        "value": value,
        "se": statistics.pstdev(reps),
        "ci": [pct(0.025), pct(0.975)],
    }


def bootstrap(values: Values, *, samples: int = 2000, seed: int = 0) -> dict:
    """A split's score with a two-stage (tasks, then trials) bootstrap."""
    per_task = list(_clean(values).values())
    if not per_task:
        return {"value": None, "se": None, "ci": None}
    rng, n = random.Random(seed), len(per_task)
    reps = [
        math.fsum(_resample(rng, per_task[rng.randrange(n)]) for _ in range(n)) / n
        for _ in range(samples)
    ]
    return _summary(score(values), reps)


def paired_delta(a: Values, b: Values, *, samples: int = 2000, seed: int = 0) -> dict:
    """``score(b) - score(a)`` over the tasks both scored, with a paired interval."""
    ca, cb = _clean(a), _clean(b)
    pairs = [(ca[t], cb[t]) for t in sorted(set(ca) & set(cb))]
    if not pairs:
        return {"value": None, "se": None, "ci": None, "paired_tasks": 0}
    rng, n = random.Random(seed), len(pairs)
    reps = []
    for _ in range(samples):
        drawn = (pairs[rng.randrange(n)] for _ in range(n))
        reps.append(
            math.fsum(_resample(rng, xb) - _resample(rng, xa) for xa, xb in drawn) / n
        )
    value = _mean([_mean(xb) - _mean(xa) for xa, xb in pairs])
    return {**_summary(value, reps), "paired_tasks": n}


def rerun_noise(values: Values, *, samples: int = 2000, seed: int = 0) -> float | None:
    """The standard error of the difference between two runs of the same version
    (None with fewer than two scored trials on some task)."""
    per_task = list(_clean(values).values())
    if not per_task or min(len(xs) for xs in per_task) < 2:
        return None
    spread = [
        [_mean(xs) + (x - _mean(xs)) * math.sqrt(len(xs) / (len(xs) - 1)) for x in xs]
        for xs in per_task
    ]
    rng, n = random.Random(seed), len(spread)
    reps = []
    for _ in range(samples):
        drawn = (spread[rng.randrange(n)] for _ in range(n))
        reps.append(math.fsum(_resample(rng, xs) - _resample(rng, xs) for xs in drawn) / n)
    return statistics.pstdev(reps)


def noise_gate(
    train: Values,
    test: Values,
    *,
    min_gain: float,
    trials: int,
    samples: int = 2000,
    seed: int = 0,
) -> dict:
    """Whether ``min_gain`` clears the 95% rerun noise on both splits."""
    splits = {}
    for i, (name, values) in enumerate((("train", train), ("test", test))):
        se = rerun_noise(values, samples=samples, seed=seed + i)
        band = None if se is None else Z95 * se
        splits[name] = {
            "tasks": len(_clean(values)),
            "noise_se": se,
            "noise_95": band,
            "ok": band is not None and min_gain >= band,
        }
    passed = all(g["ok"] for g in splits.values())
    failing = [n for n, g in splits.items() if not g["ok"]]
    if passed:
        message = (
            f"Noise gate passed: --min-gain {min_gain:g} is at least the 95% rerun "
            f"noise on train ({splits['train']['noise_95']:.3f}) and test "
            f"({splits['test']['noise_95']:.3f})."
        )
        suggestion = {}
    elif any(splits[n]["noise_95"] is None for n in failing):
        suggestion = {"trials": max(trials, 3)}
        message = (
            "Refusing to climb: with one scored trial per task the noise cannot be "
            f"measured. Run with --trials {suggestion['trials']} or more."
        )
    else:
        worst = max(splits[n]["noise_95"] / min_gain for n in failing)
        suggestion = {
            "trials": math.ceil(trials * worst**2),
            "min_gain": math.ceil(max(splits[n]["noise_95"] for n in failing) * 1000)
            / 1000,
            **{
                f"{n}_tasks": math.ceil(
                    splits[n]["tasks"] * (splits[n]["noise_95"] / min_gain) ** 2
                )
                for n in failing
            },
        }
        noise = "; ".join(
            f"{n} noise (95%) is {splits[n]['noise_95']:.3f}" for n in failing
        )
        tasks = "; ".join(
            f"about {suggestion[f'{n}_tasks']} {n} tasks (now {splits[n]['tasks']})"
            for n in failing
        )
        message = (
            f"Refusing to climb: {noise}, above --min-gain {min_gain:g}. A patch with "
            "no effect would clear it by chance too often. Any one of these fixes it: "
            f"--trials {suggestion['trials']} (now {trials}); {tasks}; --min-gain "
            f"{suggestion['min_gain']:g}. --force climbs anyway (the run is marked "
            "ungated)."
        )
    return {
        "passed": passed,
        "min_gain": min_gain,
        "splits": splits,
        "message": message,
        "suggestion": suggestion,
    }
