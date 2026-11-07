"""Branch lineage reaches result.json, results.jsonl and each child's record.

result.json's ``branches`` block used to count a
fork's children but did not say which node was which child, which label it
had, what it scored or where its archive lives; results.jsonl carried no
lineage at all; and a child's ``observation.json`` did not name its parent
rollout or fork, so a child archive copied out of the run folder lost its
place in the tree.

Unit tests against fakes; no Docker, Daytona or credentials.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.rollout._results import _build_rollout_result


class _Environment:
    async def snapshot(self):
        return StateSnapshot(id="env", path="/tmp/env")

    async def restore(self, _snap):
        return None


async def _branched(tmp_path: Path) -> Rollout:
    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy --agent")]
        )
    )
    rollout._environment = _Environment()
    rollout._rollout_dir = tmp_path / "task__abc"
    rollout._rollout_dir.mkdir()
    rollout._rollout_name = "task__abc"
    rewards = iter([1.0, 0.0])

    async def run_child(_node):
        return next(rewards)

    await rollout.branch(2, run_child, child_labels=["baseline", "hint"])
    return rollout


def _write_result(rollout: Rollout) -> dict:
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
        timing={},
        branches=rollout._branch_summary(),
    )
    return json.loads((rollout._rollout_dir / "result.json").read_text())


async def test_result_json_names_each_child(tmp_path: Path):
    rollout = await _branched(tmp_path)
    tree = json.loads((rollout._rollout_dir / "tree.json").read_text())
    fork = tree["forks"][0]
    summary = _write_result(rollout)["branches"]["forks"][0]

    assert summary["parent_node"] == fork["parent_node"]
    assert summary["parent_restore"] == "restored"
    assert summary["nodes"] == [
        {
            "index": child["index"],
            "node_id": child["node_id"],
            "label": label,
            "status": "scored",
            "reward": reward,
            "reward_source": "runner_return",
            "path": child["artifacts"]["path"],
        }
        for child, label, reward in zip(
            fork["children"], ["baseline", "hint"], [1.0, 0.0], strict=True
        )
    ]
    for node in summary["nodes"]:
        assert (rollout._rollout_dir / node["path"] / "observation.json").is_file()


async def test_results_jsonl_row_carries_the_branches_block(tmp_path: Path):
    rollout = await _branched(tmp_path)
    result = _write_result(rollout)
    row = json.loads((rollout._rollout_dir / "results.jsonl").read_text())
    assert row["info"]["branches"] == result["branches"]


def test_unbranched_results_jsonl_row_has_no_branches(tmp_path: Path):
    rollout_dir = tmp_path / "task__abc"
    rollout_dir.mkdir()
    _build_rollout_result(
        rollout_dir,
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
        timing={},
    )
    row = json.loads((rollout_dir / "results.jsonl").read_text())
    assert "branches" not in row["info"]


async def test_child_observation_names_its_place_in_the_tree(tmp_path: Path):
    rollout = await _branched(tmp_path)
    fork = json.loads((rollout._rollout_dir / "tree.json").read_text())["forks"][0]
    for child, label in zip(fork["children"], ["baseline", "hint"], strict=True):
        observation = json.loads(
            (
                rollout._rollout_dir / child["artifacts"]["path"] / "observation.json"
            ).read_text()
        )
        assert observation["lineage"] == {
            "parent_rollout": "task__abc",
            "fork_id": fork["id"],
            "parent_node": fork["parent_node"],
            "index": child["index"],
            "label": label,
        }
