"""pass@k, pass^k and partial-credit solve rates.

Harbor reports pass@k (``harbor/utils/pass_at_k.py``) and Prime Intellect's
verifiers users ask for pass@k and partial-credit solve rates; BenchFlow had
neither. These tests pin the unbiased estimators (Chen et al. 2021 for pass@k,
the tau-bench form for pass^k), the n >= k rule, and BenchFlow's counting:
unscored trials are left out rather than counted as failures (Harbor counts a
missing reward as 0) and control runs are left out of agent numbers.
"""

from __future__ import annotations

import math

import pytest

from benchflow.pass_at_k import (
    Sample,
    default_ks,
    pass_at_k,
    pass_hat_k,
    solve_rates,
)


def test_pass_at_k_matches_the_closed_form() -> None:
    # n=5, c=2, k=2: 1 - C(3,2)/C(5,2) = 1 - 3/10
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert pass_at_k(5, 0, 3) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    # n - c < k: every draw of k contains a success
    assert pass_at_k(4, 3, 2) == 1.0
    # pass@1 is the plain success fraction
    assert pass_at_k(7, 3, 1) == pytest.approx(3 / 7)


def test_pass_hat_k_is_all_k_succeed() -> None:
    # C(3,2)/C(5,2) = 3/10
    assert pass_hat_k(5, 3, 2) == pytest.approx(0.3)
    assert pass_hat_k(5, 1, 2) == 0.0
    assert pass_hat_k(4, 4, 4) == 1.0
    assert pass_hat_k(7, 3, 1) == pytest.approx(3 / 7)


def test_estimators_refuse_n_below_k() -> None:
    assert pass_at_k(2, 1, 3) is None
    assert pass_hat_k(2, 2, 3) is None
    with pytest.raises(ValueError):
        pass_at_k(3, 4, 1)
    with pytest.raises(ValueError):
        pass_at_k(3, 1, 0)


def test_default_ks_follow_the_smallest_task() -> None:
    assert default_ks(1) == [1]
    assert default_ks(5) == [1, 2, 4, 5]
    assert default_ks(10) == [1, 2, 4, 5, 8, 10]
    assert default_ks(0) == [1]


def _samples(spec: dict[str, list[float | None]]) -> list[Sample]:
    return [Sample(task, r) for task, rewards in spec.items() for r in rewards]


def test_aggregate_is_the_mean_over_tasks_and_leaves_unscored_out() -> None:
    rates = solve_rates(
        _samples({"a": [1.0, 0.0, 1.0, None], "b": [0.0, 0.0, 0.0]}), ks=[1, 2, 3]
    )
    # a: n=3 (the unscored trial is not a failure), c=2; b: n=3, c=0
    at = {p.k: p for p in rates.at_k}
    assert at[1].pass_at_k == pytest.approx((2 / 3 + 0) / 2)
    assert at[2].pass_at_k == pytest.approx((1.0 + 0) / 2)
    assert at[2].pass_hat_k == pytest.approx((1 / 3 + 0) / 2)
    assert at[3].pass_hat_k == 0.0
    assert rates.tasks == 2
    assert rates.trials == 6
    assert rates.unscored == 1
    assert rates.solve_rate == pytest.approx(2 / 6)
    assert rates.success_rule == "passed (reward = 1)"


def test_tasks_with_too_few_trials_are_left_out_of_that_k_and_said_so() -> None:
    rates = solve_rates(_samples({"a": [1.0, 0.0, 1.0], "b": [1.0]}), ks=[1, 3])
    at = {p.k: p for p in rates.at_k}
    assert at[1].tasks == 2 and at[1].tasks_short == 0
    assert at[3].tasks == 1 and at[3].tasks_short == 1
    assert at[3].pass_at_k == 1.0
    assert any("n < k" in c and "pass@3" in c for c in rates.caveats)


def test_no_task_has_enough_trials_gives_none_not_zero() -> None:
    rates = solve_rates(_samples({"a": [1.0], "b": [0.0]}), ks=[2])
    (p,) = rates.at_k
    assert p.pass_at_k is None and p.pass_hat_k is None
    assert p.tasks == 0 and p.tasks_short == 2
    assert any("pass@2" in c for c in rates.caveats)


def test_partial_credit_threshold_counts_non_binary_rewards() -> None:
    samples = _samples({"a": [0.6, 0.4], "b": [0.9, 1.0]})
    strict = solve_rates(samples, ks=[1])
    assert strict.nonbinary_rewards == 3
    assert strict.solve_rate == pytest.approx(1 / 4)
    assert any("--solve-threshold" in c for c in strict.caveats)

    lenient = solve_rates(samples, ks=[1, 2], solve_threshold=0.5)
    assert lenient.solve_threshold == 0.5
    assert lenient.success_rule == "reward >= 0.5"
    assert lenient.solve_rate == pytest.approx(3 / 4)
    at = {p.k: p for p in lenient.at_k}
    assert at[2].pass_at_k == pytest.approx(1.0)
    assert at[2].pass_hat_k == pytest.approx((0 + 1) / 2)
    assert not any("--solve-threshold" in c for c in lenient.caveats)


def test_an_explicit_gate_verdict_wins_over_the_reward_under_the_pass_rule() -> None:
    rates = solve_rates([Sample("a", 0.8, passed=True), Sample("a", 1.0, passed=False)])
    assert rates.solve_rate == pytest.approx(0.5)


def test_threshold_must_be_finite() -> None:
    with pytest.raises(ValueError):
        solve_rates([Sample("a", 1.0)], solve_threshold=math.nan)


def test_to_dict_is_json_ready() -> None:
    import json

    d = solve_rates(_samples({"a": [1.0, 0.0]}), controls_excluded=2).to_dict()
    json.dumps(d)
    assert d["controls_excluded"] == 2
    assert d["pass_at_k"]["1"] == pytest.approx(0.5)
    assert d["pass_hat_k"]["2"] == 0.0
    assert d["ks"] == [1, 2]


def test_left_out_control_runs_are_named_in_the_caveats(tmp_path):
    """``bench eval metrics`` over only an oracle run said ``Score 100%``
    beside ``Solve rate n/a (0 scored trials)`` with no word on why.

    Guards the dx/sdk fix of a first-run finding (dx/first-run, 2026-09-30):
    control runs are left out of solve rates, and now the caveats say so.
    """
    import json

    from typer.testing import CliRunner

    from benchflow.cli.main import app

    trial = tmp_path / "job" / "hello__1"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {"task_name": "hello", "agent": "oracle", "rewards": {"reward": 1.0}}
        )
    )
    out = CliRunner().invoke(app, ["eval", "metrics", str(tmp_path / "job")])
    assert out.exit_code == 0, out.output
    assert "1 control run(s) (oracle, empty/nop) are left out" in " ".join(
        out.output.split()
    )


def test_the_solve_rate_comes_with_a_95_percent_interval():
    """``Job.solve_rates()`` gave a point estimate only, so every reader (the
    hill-climb demo, docs/examples/hillclimb) carried its own bootstrap."""
    from benchflow.pass_at_k import Sample, solve_rates

    once = solve_rates([Sample(f"t{i}", 1.0 if i < 9 else 0.0) for i in range(20)])
    assert once.interval_method == "wilson"
    assert once.interval == pytest.approx((0.2582, 0.6579), abs=1e-3)

    # Four tasks, five trials each, two always solved and two never: about
    # three independent observations, not twenty.
    rows = [
        Sample(t, 1.0 if t in ("a", "b") else 0.0) for t in "abcd" for _ in range(5)
    ]
    clustered = solve_rates(rows)
    assert clustered.solve_rate == 0.5
    assert clustered.interval_method == "wilson-clustered"
    low, high = clustered.interval
    assert low < 0.2 and high > 0.8
    solved = solve_rates([Sample(t, 1.0) for t in "ab" for _ in range(5)])
    assert solved.interval[1] == 1.0 and solved.interval[0] < 0.5
    assert solve_rates([Sample("a", None)]).interval is None
    assert clustered.to_dict()["solve_rate_interval"] == list(clustered.interval)
