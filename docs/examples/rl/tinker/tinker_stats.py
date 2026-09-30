"""Solve rates with 95% intervals, for held-out evaluations. Standard library only.

A score is pass@1: the mean over tasks of each task's solved fraction over
its episodes. Tasks differ far more than repeated episodes of one task, so the
interval is a two-stage bootstrap (draw tasks, then each drawn task's
episodes), and a before/after difference is paired by task.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

# task -> rewards of its scored episodes (infrastructure drops are not here)
Results = Mapping[str, Sequence[float]]


def solved(reward: float) -> bool:
    return reward >= 1.0


def task_rates(results: Results) -> dict[str, float]:
    return {
        task: sum(solved(r) for r in rewards) / len(rewards)
        for task, rewards in results.items()
        if rewards
    }


def solve_rate(results: Results) -> float:
    rates = task_rates(results)
    return sum(rates.values()) / len(rates) if rates else math.nan


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _resampled_rate(rewards: Sequence[float], rng: random.Random) -> float:
    draws = [rewards[rng.randrange(len(rewards))] for _ in rewards]
    return sum(solved(r) for r in draws) / len(draws)


def bootstrap_interval(
    results: Results, *, samples: int = 10_000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Two-stage bootstrap interval of the solve rate."""
    tasks = [t for t, rewards in results.items() if rewards]
    if not tasks:
        return math.nan, math.nan
    rng = random.Random(seed)
    stats = []
    for _ in range(samples):
        drawn = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        stats.append(sum(_resampled_rate(results[t], rng) for t in drawn) / len(drawn))
    tail = (1 - level) / 2
    return _percentile(stats, tail), _percentile(stats, 1 - tail)


def paired_difference(
    before: Results,
    after: Results,
    *,
    samples: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, float]:
    """After minus before, over the tasks both scored, with a paired interval."""
    tasks = sorted(t for t in before if before[t] and after.get(t))
    if not tasks:
        return {"tasks": 0, "delta": math.nan, "low": math.nan, "high": math.nan}
    delta = solve_rate({t: after[t] for t in tasks}) - solve_rate(
        {t: before[t] for t in tasks}
    )
    rng = random.Random(seed)
    stats = []
    for _ in range(samples):
        drawn = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        stats.append(
            sum(
                _resampled_rate(after[t], rng) - _resampled_rate(before[t], rng)
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
    low, high = bootstrap_interval(results, seed=seed)
    w_low, w_high = wilson_interval(wins, len(episodes))
    return {
        "tasks": sum(1 for rewards in results.values() if rewards),
        "episodes": len(episodes),
        "solved": wins,
        "solve_rate": solve_rate(results),
        "ci95_low": low,
        "ci95_high": high,
        "mean_reward": sum(episodes) / len(episodes) if episodes else math.nan,
        "wilson95_low": w_low,
        "wilson95_high": w_high,
    }
