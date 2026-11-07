"""A branched rollout's result.json points at its branches and their cost.

Regression test: ``final_metrics`` and ``timing`` in result.json covered only
the parent's own turns. A run's children can use far more tokens and time than
the parent's own turns, and nothing in result.json pointed at
``tree.json``, so dashboards built on result.json undercounted branched runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY

from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.rollout._results import _build_rollout_result


class _Environment:
    async def snapshot(self):
        return StateSnapshot(id="env", path="/tmp/env")

    async def restore(self, _snap):
        return None


async def test_result_json_summarises_branch_children(tmp_path: Path):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._environment = _Environment()
    rollout._rollout_dir = tmp_path / "run"
    rollout._rollout_dir.mkdir()
    rewards = iter([1.0, 0.0])

    async def run_child(node):
        rollout._native_usage_metrics["n_input_tokens"] = 4
        rollout._native_usage_metrics["n_cache_read_tokens"] = 39_000
        rollout._native_usage_metrics["total_tokens"] = 39_150
        rollout._timing["agent_execution"] = 60.0
        rollout._timing["verifier"] = 12.5
        return next(rewards)

    await rollout.branch(2, run_child, child_labels=["baseline", "hint"])
    fork_id = json.loads((rollout._rollout_dir / "tree.json").read_text())["forks"][0][
        "id"
    ]

    _build_rollout_result(
        rollout._rollout_dir,
        task_name="task",
        rollout_name="task__abc",
        agent="dummy",
        agent_name="dummy",
        model=None,
        n_tool_calls=0,
        prompts=["p"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=datetime.now(),
        timing={"agent_execution": 5.0},
        branches=rollout._branch_summary(),
    )
    payload = json.loads((rollout._rollout_dir / "result.json").read_text())
    assert payload["timing"]["agent_execution"] == 5.0  # parent only, unchanged
    assert payload["branches"] == {
        "tree": "tree.json",
        # Discarded vs kept parent: tests/test_branch_parent_discarded.py.
        "parent": "kept",
        "forks": [
            {
                "id": fork_id,
                "status": "completed",
                "value": 0.5,
                "children": 2,
                "scored": 2,
                # Per-child lineage: tests/test_branch_lineage_records.py.
                "parent_node": ANY,
                "parent_restore": "restored",
                "nodes": ANY,
                # Cost accounting: tests/test_branch_cost.py.
                "cost": ANY,
            }
        ],
        "children": 2,
        "cost": ANY,
        "child_usage": {
            "n_input_tokens": 8,
            "n_output_tokens": 0,
            "n_cache_read_tokens": 78_000,
            "n_cache_creation_tokens": 0,
            "total_tokens": 78_300,
        },
        "child_timing_sec": {"agent_execution": 120.0, "verifier": 25.0},
    }


def test_unbranched_rollout_has_no_branches_key(tmp_path: Path):
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    assert rollout._branch_summary() is None
