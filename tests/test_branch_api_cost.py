"""bf.branch's result carries the branch cost."""

from __future__ import annotations

from benchflow.branch_api import _result
from benchflow.branch_run import BranchTrialOutcome


def test_branch_result_exposes_child_and_trial_cost(tmp_path):
    child_cost = {"tokens": 100, "usd": None, "sandbox_seconds": 12.5}
    totals = {"tokens": 100, "usd": None, "usd_known": False, "sandbox_seconds": 40.0}
    outcome = BranchTrialOutcome(
        task="t",
        value=1.0,
        fork_status="completed",
        children=[
            {
                "label": "a",
                "node_id": "n1",
                "status": "scored",
                "reward": 1.0,
                "reward_source": "verifier",
                "path": None,
                "fork_id": "f",
                "parent_label": None,
                "cost": child_cost,
            }
        ],
        forks=[{"id": "f", "from": "checkpoint", "children": 1, "value": 1.0}],
        cost=totals,
    )
    result = _result(outcome, tmp_path)
    assert result.cost == totals
    assert result.forks == outcome.forks
    assert result.children[0].cost == child_cost
