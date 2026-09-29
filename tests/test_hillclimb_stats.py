"""Bootstrap scores, paired deltas, rerun noise and the noise gate."""

from __future__ import annotations

import math
import random

import pytest

from benchflow.hillclimbing import stats


def _binary(n_tasks: int, n_trials: int, p: float, seed: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    return {
        f"t{i}": [float(rng.random() < p) for _ in range(n_trials)]
        for i in range(n_tasks)
    }


def test_the_score_is_the_mean_of_task_means():
    values = {"a": [1.0, 0.0], "b": [1.0], "c": []}
    assert stats.score(values) == pytest.approx(0.75)
    est = stats.bootstrap_score(values, samples=500, seed=1)
    assert est.value == pytest.approx(0.75)
    assert est.tasks == 2 and est.trials == 3
    assert est.ci is not None and est.ci.low <= 0.75 <= est.ci.high


def test_the_bootstrap_standard_error_matches_the_textbook_one():
    """With one trial per task the two-stage bootstrap is the plain bootstrap of
    a mean, whose SE is the population SD over sqrt(n)."""
    rng = random.Random(0)
    values = {f"t{i}": [rng.random()] for i in range(60)}
    xs = [v[0] for v in values.values()]
    mean = sum(xs) / len(xs)
    textbook = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs)) / math.sqrt(len(xs))
    est = stats.bootstrap_score(values, samples=4000, seed=3)
    assert est.se == pytest.approx(textbook, rel=0.1)


def test_the_bootstrap_is_seeded():
    values = _binary(10, 3, 0.5, 1)
    a = stats.bootstrap_score(values, samples=300, seed=5)
    b = stats.bootstrap_score(values, samples=300, seed=5)
    assert a == b


def test_a_paired_delta_cancels_differences_between_tasks():
    """Tasks differ wildly, the change adds exactly 0.1 to each: the paired
    interval is tight, where comparing two unpaired intervals would not be."""
    rng = random.Random(2)
    a = {f"t{i}": [rng.choice([0.0, 0.4, 0.8])] * 3 for i in range(20)}
    b = {t: [x + 0.1 for x in xs] for t, xs in a.items()}
    delta = stats.paired_delta(a, b, samples=500, seed=0)
    assert delta.value == pytest.approx(0.1)
    assert delta.ci.low == pytest.approx(0.1) and delta.ci.high == pytest.approx(0.1)
    assert delta.paired_tasks == 20
    unpaired = stats.bootstrap_score(a, samples=500, seed=0)
    assert unpaired.ci.high - unpaired.ci.low > 0.1


def test_a_paired_delta_only_pairs_tasks_scored_on_both_sides():
    delta = stats.paired_delta({"a": [0.0], "b": [0.0]}, {"a": [1.0], "c": [1.0]})
    assert delta.paired_tasks == 1 and delta.value == pytest.approx(1.0)
    assert stats.paired_delta({"a": [0.0]}, {"b": [1.0]}).value is None


def test_rerun_noise_matches_the_analytic_value():
    """SE of a rerun's difference = sqrt(sum_t 2 s_t^2 / n_t) / T, with s_t the
    unbiased trial variance; the adjusted-residual bootstrap recovers it."""
    values = _binary(40, 3, 0.5, 11)
    analytic = 0.0
    for xs in values.values():
        n = len(xs)
        m = sum(xs) / n
        s2 = sum((x - m) ** 2 for x in xs) / (n - 1)
        analytic += 2 * s2 / n
    analytic = math.sqrt(analytic) / len(values)
    noise = stats.rerun_noise(values, samples=4000, seed=9)
    assert noise.se == pytest.approx(analytic, rel=0.12)
    assert noise.band95 == pytest.approx(stats.Z95 * noise.se)


def test_rerun_noise_needs_two_trials_per_task():
    noise = stats.rerun_noise({"a": [1.0], "b": [0.0, 1.0]})
    assert noise.se is None and "fewer than 2 scored trials" in (noise.reason or "")


def test_deterministic_tasks_have_no_rerun_noise():
    noise = stats.rerun_noise({"a": [1.0, 1.0], "b": [0.0, 0.0]}, samples=200)
    assert noise.se == 0.0


def test_the_gate_passes_when_min_gain_clears_the_noise():
    train = _binary(40, 8, 0.9, 1)
    test = _binary(20, 8, 0.9, 2)
    gate = stats.noise_gate(train, test, min_gain=0.2, trials=8, samples=500)
    assert gate.passed, gate.message
    assert "passed" in gate.message


def test_the_gate_refuses_and_says_what_would_fix_it():
    train = _binary(12, 2, 0.5, 3)
    test = _binary(6, 2, 0.5, 4)
    gate = stats.noise_gate(train, test, min_gain=0.05, trials=2, samples=500)
    assert not gate.passed
    worst = max(gate.train.threshold, gate.test.threshold)
    factor = (worst / 0.05) ** 2
    assert gate.suggestion.trials == math.ceil(2 * factor)
    assert gate.suggestion.min_gain >= worst
    assert f"--trials {gate.suggestion.trials} (now 2)" in gate.message
    assert "--force" in gate.message


def test_one_trial_per_task_cannot_be_gated():
    train = {"a": [1.0], "b": [0.0]}
    gate = stats.noise_gate(train, train, min_gain=0.5, trials=1, samples=200)
    assert not gate.passed
    assert gate.suggestion.trials == 3
    assert "--trials 3" in gate.message


def test_the_cost_gate_works_in_fractions_of_cost():
    cheap = {f"t{i}": [0.010, 0.011, 0.009] for i in range(20)}
    gate = stats.noise_gate(
        cheap, cheap, min_gain=0.1, objective="cost", trials=3, samples=500
    )
    assert gate.passed, gate.message
    assert gate.train.threshold < 0.1
