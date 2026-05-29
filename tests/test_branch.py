"""Tests for the Branch operation — the credit-assignment engine.

A Branch turns a RolloutNode into a fork point: checkpoint the environment
(roll-back point), fork N children, later restore to the checkpoint, and
aggregate the children's returns into V(parent).
"""

import pytest

from benchflow.branch import (
    aggregate,
    aggregate_verify_result,
    branch,
    checkpoint,
    fork,
    restore,
)
from benchflow.environment.protocol import StateSnapshot
from benchflow.rewards.protocol import VerifyResult
from benchflow.trajectories.tree import RolloutTree, branch_points


class FakeEnv:
    """Minimal Environment stand-in recording snapshot/restore calls."""

    def __init__(self) -> None:
        self.snapshots: list[StateSnapshot] = []
        self.restored: list[StateSnapshot] = []

    async def snapshot(self) -> StateSnapshot:
        snap = StateSnapshot(id=f"snap-{len(self.snapshots) + 1}", path="/tmp/x")
        self.snapshots.append(snap)
        return snap

    async def restore(self, snap: StateSnapshot) -> None:
        self.restored.append(snap)


async def test_checkpoint_snapshots_env_and_records_on_node():
    tree = RolloutTree()
    env = FakeEnv()
    snap = await checkpoint(tree.root, env)
    assert isinstance(snap, StateSnapshot)
    assert len(env.snapshots) == 1
    assert tree.root.state["snapshot"] is snap


def test_fork_creates_n_children_and_makes_a_branch_point():
    tree = RolloutTree()
    children = fork(tree, tree.root, 3)
    assert len(children) == 3
    assert all(c.parent is tree.root for c in children)
    assert tree.root in branch_points(tree)


def test_fork_rejects_fewer_than_two_children():
    tree = RolloutTree()
    with pytest.raises(ValueError, match=">= 2"):
        fork(tree, tree.root, 1)


async def test_restore_rolls_env_back_to_a_nodes_checkpoint():
    tree = RolloutTree()
    env = FakeEnv()
    snap = await checkpoint(tree.root, env)
    await restore(tree.root, env)
    assert env.restored == [snap]


async def test_restore_without_a_checkpoint_raises():
    tree = RolloutTree()
    with pytest.raises(ValueError, match="no checkpoint"):
        await restore(tree.root, FakeEnv())


async def test_branch_checkpoints_then_forks():
    tree = RolloutTree()
    env = FakeEnv()
    children = await branch(tree, tree.root, env, 3)
    assert len(children) == 3
    assert len(env.snapshots) == 1
    assert tree.root.state["snapshot"] is env.snapshots[0]
    assert tree.root in branch_points(tree)


def test_aggregate_averages_children_returns_into_value():
    tree = RolloutTree()
    children = fork(tree, tree.root, 2)
    children[0].state["reward"] = 1.0
    children[1].state["reward"] = 0.0
    assert aggregate(tree.root) == 0.5


def test_aggregate_verify_result_composes_children():
    """V(parent) is a node-scored VerifyResult: mean reward, namespaced items,
    concatenated events (the canonical Phase 1b aggregation)."""
    tree = RolloutTree()
    children = fork(tree, tree.root, 2)
    children[0].state["verify_result"] = VerifyResult(reward=1.0, items={"acc": 1.0})
    children[1].state["verify_result"] = VerifyResult(reward=0.0)
    vr = aggregate_verify_result(tree.root)
    assert isinstance(vr, VerifyResult)
    assert vr.reward == 0.5
    assert vr.space == "output" and vr.granularity == "terminal"
    assert vr.items["child-0"] == 1.0 and vr.items["child-1"] == 0.0
    assert vr.items["child-0/acc"] == 1.0  # child items namespaced, no collision
    assert vr.error is None


def test_aggregate_verify_result_error_is_conservative():
    """If any child failed scoring, V(parent).error is populated (a partial
    scoring failure must not masquerade as a clean value)."""
    tree = RolloutTree()
    children = fork(tree, tree.root, 2)
    children[0].state["verify_result"] = VerifyResult(reward=1.0)
    children[1].state["verify_result"] = VerifyResult(reward=0.0, error="boom")
    vr = aggregate_verify_result(tree.root)
    assert vr.reward == 0.5  # value still defined from the children that scored
    assert vr.error is not None and "1/2" in vr.error


def test_aggregate_verify_result_tolerates_bare_float_children():
    """A child scored with only a bare float reward is lifted, not dropped."""
    tree = RolloutTree()
    children = fork(tree, tree.root, 2)
    children[0].state["reward"] = 1.0  # no verify_result
    children[1].state["reward"] = 0.0
    vr = aggregate_verify_result(tree.root)
    assert isinstance(vr, VerifyResult)
    assert vr.reward == 0.5


def test_aggregate_without_children_raises():
    tree = RolloutTree()
    with pytest.raises(ValueError, match="no children"):
        aggregate(tree.root)
