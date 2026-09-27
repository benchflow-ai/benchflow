"""A job whose trials reported no USD cost has an unknown cost, not $0.

A batch on a subscription login (every trial ``cost_usd: null``,
``usage_source: agent_native_acp``) wrote ``total_cost_usd: 0.0`` and
``avg_cost_per_trial_usd: 0.0`` to summary.json, and ``bench eval metrics``
did the same: the sums treated a missing cost as zero. The totals now count
only trials that reported a cost, and are null when none did.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchflow._utils.evaluation_results import usage_summary
from benchflow.metrics import collect_metrics


def _row(cost: float | None, tokens: int) -> dict:
    return {
        "rewards": {"reward": 1.0},
        "agent_result": {
            "usage_source": "agent_native_acp",
            "n_input_tokens": 10,
            "n_output_tokens": 5,
            "total_tokens": tokens,
            "cost_usd": cost,
        },
    }


def test_summary_cost_is_null_when_no_trial_reported_one() -> None:
    usage = usage_summary({"a": _row(None, 100), "b": _row(None, 200)})

    assert usage["total_tokens"] == 300
    assert usage["total_cost_usd"] is None
    assert usage["avg_cost_per_trial_usd"] is None


def test_summary_cost_averages_over_priced_trials() -> None:
    usage = usage_summary({"a": _row(0.004, 100), "b": _row(None, 200)})

    assert usage["total_cost_usd"] == 0.004
    assert usage["avg_cost_per_trial_usd"] == 0.004


def test_metrics_cost_is_null_when_no_trial_reported_one(tmp_path: Path) -> None:
    for name in ("a", "b"):
        trial = tmp_path / "job" / f"{name}__1"
        trial.mkdir(parents=True)
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": name,
                    "error": None,
                    "verifier_error": None,
                    "n_tool_calls": 1,
                    "started_at": "2026-03-24 10:00:00.000000",
                    "finished_at": "2026-03-24 10:01:00.000000",
                    **_row(None, 100),
                }
            )
        )

    summary = collect_metrics(str(tmp_path)).summary()

    assert summary["total_tokens"] == 200
    assert summary["total_cost_usd"] is None
    assert summary["avg_cost_per_trial_usd"] is None
