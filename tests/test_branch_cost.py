"""Cost and time accounting per branch child, fork and run.

A fork's children used to record tokens and
phase seconds, but not what a branch cost in sandbox time or money, so
parallel (more sandboxes) and in-place (one sandbox, longer) forks could not
be compared on cost. Each child now records ``cost`` (tokens, USD when the
provider reported it, sandbox-seconds); each fork records its wall time, the
parent sandbox's seconds during the fork, the isolated children's sandbox
seconds, and totals; result.json's ``branches`` block sums them over forks
without counting a nested fork's parent (a child's sandbox) twice.

Unit tests against fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import json

import pytest

from benchflow.branch_lineage import branch_summary
from tests.test_branch_isolated import IMAGES, IsoRollout, _root, _runner, _tree
from tests.test_branch_restore_parent import _fork, _rollout


@pytest.fixture(autouse=True)
def _reset():
    IMAGES.clear()
    IsoRollout.all = []


async def test_in_place_fork_costs_one_sandbox(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)
    usages = iter([100, 250])

    async def child(_node):
        rollout._native_usage_metrics["total_tokens"] = next(usages)
        rollout._native_usage_metrics["cost_usd"] = None
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    fork = _fork(rollout)
    for record in fork["children"]:
        cost = record["cost"]
        assert cost["usd"] is None
        assert isinstance(cost["sandbox_seconds"], float)
    assert [c["cost"]["tokens"] for c in fork["children"]] == [100, 250]
    total = fork["cost"]
    assert total["tokens"] == 350
    assert total["usd"] is None and total["usd_known"] is False
    # In place, the children run inside the parent's sandbox.
    assert total["children_sandbox_seconds"] == 0.0
    assert total["parent_sandbox_seconds"] == total["wall_seconds"]
    assert total["sandbox_seconds"] == total["wall_seconds"]


async def test_known_usd_is_summed(tmp_path):
    rollout, _sandbox = _rollout(tmp_path)
    prices = iter([0.25, 0.5])

    async def child(_node):
        rollout._native_usage_metrics["cost_usd"] = next(prices)
        return 1.0

    await rollout.branch(2, child, snapshot_layers={"sandbox"})
    total = _fork(rollout)["cost"]
    assert total["usd"] == 0.75 and total["usd_known"] is True


async def test_isolated_children_add_their_own_sandboxes(tmp_path):
    root = await _root(tmp_path)
    await root.branch(
        2,
        _runner({"a": "Do it.", "b": "Do it."}),
        snapshot_layers={"sandbox"},
        child_labels=["a", "b"],
        isolate_children=True,
        concurrency=2,
    )
    fork = _tree(root)["forks"][0]
    child_seconds = [c["cost"]["sandbox_seconds"] for c in fork["children"]]
    assert all(isinstance(s, float) for s in child_seconds)
    total = fork["cost"]
    assert total["children_sandbox_seconds"] == pytest.approx(sum(child_seconds))
    assert total["sandbox_seconds"] == pytest.approx(
        total["parent_sandbox_seconds"] + sum(child_seconds)
    )


def _fork_record(fork_id, rollout, parent_seconds, children):
    return {
        "id": fork_id,
        "rollout": rollout,
        "status": "completed",
        "value": 1.0,
        "parent_node": "root",
        "parent_restore": "restored",
        "children": [
            {
                "index": i,
                "node_id": node,
                "status": "scored",
                "reward": 1.0,
                "reward_source": "verifier",
                "intervention": {"label": node},
                "artifacts": {"path": None},
                "usage": {"total_tokens": 10},
                "timing_sec": {},
                "cost": {"tokens": 10, "usd": None, "sandbox_seconds": seconds},
            }
            for i, (node, seconds) in enumerate(children)
        ],
        "cost": {
            "tokens": 10 * len(children),
            "usd": None,
            "usd_known": False,
            "wall_seconds": parent_seconds,
            "parent_sandbox_seconds": parent_seconds,
            "children_sandbox_seconds": sum(s for _, s in children),
            "sandbox_seconds": parent_seconds + sum(s for _, s in children),
        },
    }


def test_result_totals_do_not_count_a_nested_forks_parent_twice():
    outer = _fork_record("f1", "task__x", 100.0, [("n1", 60.0), ("n2", 50.0)])
    # Forked by child n1: its parent sandbox is n1's, already in n1's 60 s.
    nested = _fork_record("f2", "n1", 30.0, [("n5", 20.0), ("n6", 25.0)])
    summary = branch_summary([outer, nested])
    assert summary["cost"] == {
        "tokens": 40,
        "usd": None,
        "usd_known": False,
        "sandbox_seconds": 100.0 + 60.0 + 50.0 + 20.0 + 25.0,
    }
    assert summary["forks"][1]["cost"]["sandbox_seconds"] == 75.0
    json.dumps(summary)
