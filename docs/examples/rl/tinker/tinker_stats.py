"""Held-out scores with 95% intervals, and before/after differences. Stdlib only.

Two measures per policy: the solve rate (an episode is solved when every
check passes, reward 1) and the mean reward (partial credit). Each is a mean
over tasks of that task's mean over its episodes, so every task weighs the
same. Tasks differ far more than repeated episodes of one task, so intervals
come from a two-stage bootstrap (draw tasks, then each drawn task's episodes),
and a before/after difference is paired by task.

    python tinker_stats.py BEFORE AFTER

compares two evaluations: a folder written by the shared evaluator
(docs/examples/rl/common/evaluate.py: episodes.jsonl) or an eval.json written
by tinker_eval.py (first evaluation in it). Dropped episodes (infrastructure)
are not scores and are left out.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

# task -> rewards of its kept episodes
Results = Mapping[str, Sequence[float]]
Measure = Callable[[float], float]


def solved(reward: float) -> bool:
    return reward >= 1.0


MEASURES: dict[str, Measure] = {
    "solve_rate": lambda r: float(solved(r)),
    "mean_reward": lambda r: float(r),
}


def _kept(results: Results) -> list[str]:
    return [t for t, rewards in results.items() if rewards]


def score(results: Results, measure: Measure = MEASURES["solve_rate"]) -> float:
    tasks = _kept(results)
    if not tasks:
        return math.nan
    return sum(sum(map(measure, results[t])) / len(results[t]) for t in tasks) / len(
        tasks
    )


def solve_rate(results: Results) -> float:
    return score(results, MEASURES["solve_rate"])


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _resampled(rewards: Sequence[float], measure: Measure, rng: random.Random) -> float:
    return sum(measure(rewards[rng.randrange(len(rewards))]) for _ in rewards) / len(
        rewards
    )


def bootstrap_interval(
    results: Results,
    measure: Measure = MEASURES["solve_rate"],
    *,
    samples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Two-stage bootstrap interval of `score(results, measure)`."""
    tasks = _kept(results)
    if not tasks:
        return math.nan, math.nan
    rng = random.Random(seed)
    stats = []
    for _ in range(samples):
        drawn = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        stats.append(
            sum(_resampled(results[t], measure, rng) for t in drawn) / len(drawn)
        )
    tail = (1 - level) / 2
    return _percentile(stats, tail), _percentile(stats, 1 - tail)


def paired_difference(
    before: Results,
    after: Results,
    measure: Measure = MEASURES["solve_rate"],
    *,
    samples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """After minus before, over the tasks both scored, with a paired interval."""
    tasks = sorted(t for t in before if before[t] and after.get(t))
    if not tasks:
        return {"tasks": 0, "delta": math.nan, "low": math.nan, "high": math.nan}
    delta = score({t: after[t] for t in tasks}, measure) - score(
        {t: before[t] for t in tasks}, measure
    )
    rng = random.Random(seed)
    stats = []
    for _ in range(samples):
        drawn = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        stats.append(
            sum(
                _resampled(after[t], measure, rng) - _resampled(before[t], measure, rng)
                for t in drawn
            )
            / len(drawn)
        )
    tail = (1 - level) / 2
    return {
        "tasks": len(tasks),
        "delta": delta,
        "low": _percentile(stats, tail),
        "high": _percentile(stats, 1 - tail),
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, for pooled episodes treated as independent."""
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(results: Results, *, seed: int = 0) -> dict[str, float | int]:
    episodes = [r for rewards in results.values() for r in rewards]
    wins = sum(solved(r) for r in episodes)
    out: dict[str, float | int] = {
        "tasks": len(_kept(results)),
        "episodes": len(episodes),
        "solved": wins,
    }
    for name, measure in MEASURES.items():
        low, high = bootstrap_interval(results, measure, seed=seed)
        out[name] = score(results, measure)
        out[f"{name}_ci95_low"] = low
        out[f"{name}_ci95_high"] = high
    # The names tinker_eval.py and the README use for the solve rate's interval.
    out["ci95_low"], out["ci95_high"] = (
        out["solve_rate_ci95_low"],
        out["solve_rate_ci95_high"],
    )
    out["wilson95_low"], out["wilson95_high"] = wilson_interval(wins, len(episodes))
    return out


def load_results(path: Path | str) -> tuple[str, dict[str, list[float]]]:
    """(label, results) from an evaluate.py folder or a tinker_eval.py eval.json."""
    path = Path(path)
    if path.is_dir():
        results: dict[str, list[float]] = {}
        for line in (path / "episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("reward") is None:  # dropped: infrastructure
                results.setdefault(row["task_id"], [])
                continue
            results.setdefault(row["task_id"], []).append(float(row["reward"]))
        return path.name, results
    doc = json.loads(path.read_text())
    evaluation = doc["evaluations"][0]
    return evaluation["label"], {t: list(r) for t, r in evaluation["rewards"].items()}


def compare(before: Path | str, after: Path | str, *, seed: int = 0) -> dict:
    (label_a, a), (label_b, b) = load_results(before), load_results(after)
    return {
        "before": {"label": label_a, **summarize(a, seed=seed)},
        "after": {"label": label_b, **summarize(b, seed=seed)},
        "difference": {
            name: paired_difference(a, b, measure, seed=seed)
            for name, measure in MEASURES.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__.split("\n\n")[2], file=sys.stderr)
        return 2
    doc = compare(args[0], args[1])
    for side in ("before", "after"):
        s = doc[side]
        print(
            f"{side:6} {s['label']}: solve rate {s['solve_rate']:.3f} "
            f"[{s['solve_rate_ci95_low']:.3f}, {s['solve_rate_ci95_high']:.3f}], "
            f"mean reward {s['mean_reward']:.3f} "
            f"[{s['mean_reward_ci95_low']:.3f}, {s['mean_reward_ci95_high']:.3f}] "
            f"({s['episodes']} episodes, {s['tasks']} tasks)"
        )
    for name, d in doc["difference"].items():
        print(
            f"after - before, {name}: {d['delta']:+.3f} "
            f"[{d['low']:+.3f}, {d['high']:+.3f}] over {d['tasks']} tasks"
        )
    print(json.dumps(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
