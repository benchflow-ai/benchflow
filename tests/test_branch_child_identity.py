"""A branch runner can learn which child it is running.

``ChildRunner`` used to receive only the opaque pending node
(``n5``, ``n9``), so a runner that needed per-child prompts kept a mutable
call counter, which breaks silently when a runner is retried or reordered.
A runner that declares a keyword-only ``child`` parameter (or ``**kwargs``)
now receives a :class:`BranchChild` with the index, label, node and fork id;
runners that take only the node keep working unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchflow.environment.protocol import StateSnapshot
from benchflow.rollout import Rollout, RolloutConfig, Scene
from benchflow.rollout_branch import BranchChild


class _Environment:
    async def snapshot(self):
        return StateSnapshot(id="env", path="/tmp/env")

    async def restore(self, _snap):
        return None


def _rollout(tmp_path: Path) -> Rollout:
    rollout = Rollout(
        RolloutConfig(task_path=tmp_path / "task", scenes=[Scene.single(agent="dummy")])
    )
    rollout._environment = _Environment()
    rollout._rollout_dir = tmp_path / "run"
    rollout._rollout_dir.mkdir()
    return rollout


async def test_keyword_runner_receives_index_label_node_and_fork(tmp_path):
    rollout = _rollout(tmp_path)
    seen: list[BranchChild] = []
    prompts = {"baseline": 1.0, "hint": 0.0}

    async def run_child(node, *, child: BranchChild) -> float:
        assert child.node is node
        seen.append(child)
        return prompts[child.label]

    value = await rollout.branch(2, run_child, child_labels=["baseline", "hint"])
    assert value == 0.5
    assert [(c.index, c.label) for c in seen] == [(0, "baseline"), (1, "hint")]
    fork = json.loads((rollout._rollout_dir / "tree.json").read_text())["forks"][0]
    assert {c.fork_id for c in seen} == {fork["id"]}
    assert [c.node.id for c in seen] == [ch["node_id"] for ch in fork["children"]]
    assert [ch["reward"] for ch in fork["children"]] == [1.0, 0.0]


async def test_var_keyword_runner_receives_the_identity(tmp_path):
    rollout = _rollout(tmp_path)
    indexes: list[int] = []

    async def run_child(node, **kwargs) -> float:
        indexes.append(kwargs["child"].index)
        return 1.0

    await rollout.branch(3, run_child)
    assert indexes == [0, 1, 2]


async def test_node_only_runner_named_child_keeps_working(tmp_path):
    """Back-compat: a positional parameter that happens to be named ``child``
    must still receive the node, and nothing else."""
    rollout = _rollout(tmp_path)
    nodes = []

    async def run_child(child):
        nodes.append(child)
        return 1.0

    assert await rollout.branch(2, run_child) == 1.0
    assert [node.id for node in nodes] == [
        node.id for node in rollout.tree.root.children
    ]
