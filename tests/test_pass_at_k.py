"""pass@k, pass^k and partial-credit solve rates.

BenchFlow reports pass@k and partial-credit solve rates. These tests pin the unbiased estimators (Chen et al. 2021 for pass@k,
the tau-bench form for pass^k), the n >= k rule, and BenchFlow's counting:
every trial is a sample and an unscored one counts as not solved (as the Rust
bench 0.8.2 counts), with BenchFlow 0.8's scored-only figures kept beside
them, and control runs are left out of agent numbers.
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


def test_aggregate_counts_every_trial_and_keeps_the_scored_only_figures() -> None:
    """An unscored trial counts as a fail; 0.8's figures stay as *_scored_only.

    Guards the PY8 P4 change (one denominator for pass@k): 0.8 left an
    errored trial out of n, so an agent that errored scored higher than one
    that failed cleanly.
    """
    rates = solve_rates(
        _samples({"a": [1.0, 0.0, 1.0, None], "b": [0.0, 0.0, 0.0]}), ks=[1, 2, 3]
    )
    # Every trial: a: n=4 c=2; b: n=3 c=0
    at = {p.k: p for p in rates.at_k}
    assert at[1].pass_at_k == pytest.approx((2 / 4 + 0) / 2)
    assert at[2].pass_at_k == pytest.approx((1 - 1 / 6 + 0) / 2)
    assert at[2].pass_hat_k == pytest.approx((1 / 6 + 0) / 2)
    assert at[3].pass_hat_k == 0.0
    assert rates.tasks == 2
    assert rates.trials == 7
    assert rates.scored == 6
    assert rates.unscored == 1
    assert rates.solve_rate == pytest.approx(2 / 7)
    assert rates.success_rule == "passed (reward = 1)"
    # Scored only (0.8): a: n=3 c=2; b: n=3 c=0
    so = {p.k: p for p in rates.at_k_scored_only}
    assert so[1].pass_at_k == pytest.approx((2 / 3 + 0) / 2)
    assert so[2].pass_at_k == pytest.approx((1.0 + 0) / 2)
    assert so[2].pass_hat_k == pytest.approx((1 / 3 + 0) / 2)
    assert rates.solve_rate_scored_only == pytest.approx(2 / 6)
    assert rates.tasks_scored_only == 2
    d = rates.to_dict()
    assert d["pass_at_k"]["1"] == pytest.approx(0.25)
    assert d["pass_at_k_scored_only"]["1"] == pytest.approx(1 / 3)
    assert (d["trials"], d["scored"], d["unscored"]) == (7, 6, 1)
    assert any("count as not solved" in c for c in rates.caveats)


def test_an_errored_agent_no_longer_outscores_one_that_fails_cleanly() -> None:
    """Agent A errors on 2 of 4 trials and passes the rest; agent B passes
    the same 2 and fails the other 2 cleanly. 0.8 gave A 100% and B 50%.

    Guards the PY8 P4 change (one denominator for pass@k).
    """
    a = solve_rates(_samples({"t": [1.0, 1.0, None, None]}))
    b = solve_rates(_samples({"t": [1.0, 1.0, 0.0, 0.0]}))
    assert a.solve_rate == b.solve_rate == 0.5
    assert a.get(1).pass_at_k == b.get(1).pass_at_k == 0.5
    assert a.get(4).pass_hat_k == b.get(4).pass_hat_k == 0.0
    assert a.solve_rate_scored_only == 1.0 and b.solve_rate_scored_only == 0.5


def test_a_task_with_no_scored_trial_counts_as_unsolved() -> None:
    """0.8 dropped such a task from every figure; the Rust bench's
    an_unscored_task_does_not_empty_pass_at_k is the same case."""
    rates = solve_rates(_samples({"a": [1.0, 0.0], "b": [None, None]}))
    assert rates.tasks == 2 and rates.tasks_scored_only == 1
    assert rates.get(1).pass_at_k == pytest.approx(0.25)
    assert rates.get(2).pass_at_k == pytest.approx(0.5)
    assert rates.get_scored_only(1).pass_at_k == pytest.approx(0.5)
    assert rates.get_scored_only(2).pass_at_k == 1.0
    assert rates.interval is not None and rates.interval_scored_only is not None
    only_unscored = solve_rates(_samples({"a": [None]}))
    assert only_unscored.solve_rate == 0.0
    assert only_unscored.solve_rate_scored_only is None
    assert only_unscored.get_scored_only(1).pass_at_k is None


def test_rows_show_both_figures_with_their_n_only_when_they_differ() -> None:
    rates = solve_rates(_samples({"a": [1.0, None], "b": [0.0, 1.0]}), ks=[1])
    rows = dict(rates.rows())
    assert rows["Solve rate"] == "50.0% (2 of 4 trials, 2 tasks, n 2-2 per task)"
    assert rows["Solve rate, scored only"] == "66.7% (2 of 3 scored trials, 2 tasks)"
    assert rows["pass@1"].startswith("50.0% · pass^1 50.0%")
    assert rows["pass@1, scored only"].startswith("75.0% · pass^1 75.0%")
    clean = dict(solve_rates(_samples({"a": [1.0, 0.0]}), ks=[1]).rows())
    assert set(clean) == {"Solve rate", "pass@1"}


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
    the caveats say how control runs were counted. Since DF-p3-eval-runner-3
    a folder of only control runs is counted (as inspect counts it), so its
    solve rate is the oracle's; with an agent run beside it, they are left
    out and the caveat names the flag that counts them.
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
    text = " ".join(out.output.split())
    assert (
        "Every trial is a control run (oracle, empty/nop), so they are counted" in text
    )
    assert "100.0% (1 of 1 trials" in text
    agent = tmp_path / "job" / "hello__2"
    agent.mkdir()
    (agent / "result.json").write_text(
        json.dumps(
            {"task_name": "hello", "agent": "claude-agent-acp", "model": "m",
             "rewards": {"reward": 0.0}}
        )
    )  # fmt: skip
    out = CliRunner().invoke(app, ["eval", "metrics", str(tmp_path / "job")])
    assert "1 control run(s) (oracle, empty/nop) are left out" in " ".join(
        out.output.split()
    )
    assert "--include-controls counts them" in " ".join(out.output.split())


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
    # An unscored trial is a trial (a fail), but not a scored one.
    assert solve_rates([Sample("a", None)]).interval == pytest.approx(
        (0.0, 0.7935), abs=1e-3
    )
    assert solve_rates([Sample("a", None)]).interval_scored_only is None
    assert solve_rates([]).interval is None
    assert clustered.to_dict()["solve_rate_interval"] == list(clustered.interval)
