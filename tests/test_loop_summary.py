"""Tests for the job-level loop convergence report (loop_summary)."""

from __future__ import annotations

import pytest

from benchflow._utils.evaluation_results import loop_summary


def _r(loop: dict | None) -> dict:
    return {"loop": loop} if loop is not None else {}


def test_empty_when_no_real_loop():
    assert loop_summary({}) == {}
    assert loop_summary({"a": _r(None)}) == {}
    assert loop_summary({"a": _r({"strategy": "single-shot"})}) == {}
    assert loop_summary({"a": _r({"strategy": None})}) == {}


def test_pass_at_iteration_curve_and_convergence():
    results = {
        "t1": _r(
            {
                "strategy": "verify-retry",
                "reward_trajectory": [0.0, 1.0],
                "first_pass_iteration": 1,
                "iterations_run": 2,
                "stop_reason": "passed_bar",
            }
        ),
        "t2": _r(
            {
                "strategy": "verify-retry",
                "reward_trajectory": [1.0],
                "first_pass_iteration": 0,
                "iterations_run": 1,
                "stop_reason": "passed_bar",
            }
        ),
        "t3": _r(
            {
                "strategy": "verify-retry",
                "reward_trajectory": [0.0, 0.0, 0.0],
                "first_pass_iteration": None,
                "iterations_run": 3,
                "stop_reason": "max_iterations",
            }
        ),
    }
    s = loop_summary(results)["loop_summary"]
    assert s["strategy"] == "verify-retry"
    assert s["n_tasks"] == 3
    assert s["fraction_converged"] == pytest.approx(2 / 3)
    # t2 passes @0, t1 @1, t3 never → cumulative pass@iteration over 3 slots.
    # (values are rounded to 4 decimals in the report)
    assert s["pass_at_iteration"] == [
        pytest.approx(1 / 3, abs=1e-3),
        pytest.approx(2 / 3, abs=1e-3),
        pytest.approx(2 / 3, abs=1e-3),
    ]
    assert s["mean_iterations_to_converge"] == pytest.approx(0.5)  # (1 + 0) / 2
    assert s["mean_iterations_run"] == pytest.approx(2.0)  # (2 + 1 + 3) / 3
    assert s["stop_reasons"] == {"passed_bar": 2, "max_iterations": 1}


def test_none_converged_curve_is_flat_zero():
    results = {
        "t1": _r(
            {
                "strategy": "verify-retry",
                "reward_trajectory": [0.0, 0.0],
                "first_pass_iteration": None,
                "iterations_run": 2,
                "stop_reason": "max_iterations",
            }
        ),
    }
    s = loop_summary(results)["loop_summary"]
    assert s["fraction_converged"] == 0.0
    assert s["mean_iterations_to_converge"] is None
    assert s["pass_at_iteration"] == [0.0, 0.0]


def test_ignores_single_shot_rows_in_a_mixed_job():
    results = {
        "loop": _r(
            {
                "strategy": "verify-retry",
                "reward_trajectory": [1.0],
                "first_pass_iteration": 0,
                "iterations_run": 1,
                "stop_reason": "passed_bar",
            }
        ),
        "baseline": _r({"strategy": "single-shot"}),
    }
    s = loop_summary(results)["loop_summary"]
    assert s["n_tasks"] == 1  # single-shot row excluded
    assert s["fraction_converged"] == 1.0
