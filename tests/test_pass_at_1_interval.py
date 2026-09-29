"""A bootstrap interval for pass@1 (benchflow.pass_at_k.pass_at_1_interval).

BenchFlow reported pass@1 and solve rates with no interval, and bf.compare
says it computes no significance; the hill-climb demo
(docs/examples/hillclimb) had to carry its own bootstrap to say whether a
score moved beyond noise.
"""

from __future__ import annotations

import random

import pytest

from benchflow.pass_at_k import Sample, pass_at_1_interval, solve_rates


def _samples(rates: list[float], trials: int, seed: int = 0) -> list[Sample]:
    rng = random.Random(seed)
    return [
        Sample(f"t{i}", float(rng.random() < p))
        for i, p in enumerate(rates)
        for _ in range(trials)
    ]


def test_the_interval_surrounds_pass_at_1_and_is_seeded():
    samples = _samples([0.2, 0.5, 0.8] * 10, trials=4)
    point = solve_rates(samples, ks=[1]).get(1).pass_at_k
    low, high = pass_at_1_interval(samples, seed=3)
    assert low < point < high and high - low < 0.5
    assert pass_at_1_interval(samples, seed=3) == (low, high)


def test_more_tasks_and_trials_narrow_it():
    few = pass_at_1_interval(_samples([0.5] * 5, trials=2))
    many = pass_at_1_interval(_samples([0.5] * 50, trials=8))
    assert (many[1] - many[0]) < (few[1] - few[0]) / 2


def test_unscored_samples_and_the_success_rule_follow_solve_rates():
    assert pass_at_1_interval([Sample("a", 1.0), Sample("a", None)]) == (1.0, 1.0)
    partial = [Sample("b", 0.6), Sample("b", 0.6)]
    assert pass_at_1_interval(partial) == (0.0, 0.0)
    assert pass_at_1_interval(partial, solve_threshold=0.5) == (1.0, 1.0)
    assert pass_at_1_interval([Sample("a", None)]) is None
    with pytest.raises(ValueError):
        pass_at_1_interval(partial, confidence=1.5)
