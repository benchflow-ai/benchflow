"""The Branch operation — the credit-assignment engine of the tree-native Rollout.

Han's insight: a checkpoint forks one state into N continuations, which turns a
reward function into a value function V(s). A Branch is four operations on a
``RolloutTree`` node and an ``Environment``:

* ``checkpoint`` — snapshot the environment; the roll-back point for the fork.
* ``fork`` — split the node into N child continuations.
* ``restore`` — roll the environment back to a node's checkpoint.
* ``aggregate`` — average the children's returns into V(node).

These are the Branch *operations*. Wiring them into the rollout engine — running
the forked children as sub-rollouts, quiescing the agent first — is the engine's
job; these primitives stay pure and independently testable.
"""

from __future__ import annotations

from typing import Any

from benchflow.environment.protocol import StateSnapshot
from benchflow.trajectories.tree import RolloutNode, RolloutTree, Step

_SNAPSHOT_KEY = "snapshot"
_REWARD_KEY = "reward"


async def checkpoint(node: RolloutNode, environment: Any) -> StateSnapshot:
    """Snapshot the environment at ``node`` and record the snapshot on it.

    The recorded ``StateSnapshot`` is the roll-back point every child forked
    from ``node`` restores to before it runs.
    """
    snap = await environment.snapshot()
    node.state[_SNAPSHOT_KEY] = snap
    return snap


def fork(tree: RolloutTree, node: RolloutNode, n: int) -> list[RolloutNode]:
    """Fork ``node`` into ``n`` child continuations, making it a branch point."""
    if n < 2:
        raise ValueError(f"a branch forks into >= 2 children, got n={n}")
    return [tree.advance(node, Step(id=f"{node.id}-branch-{i}")) for i in range(n)]


async def restore(node: RolloutNode, environment: Any) -> None:
    """Roll the environment back to ``node``'s recorded checkpoint."""
    snap = node.state.get(_SNAPSHOT_KEY)
    if snap is None:
        raise ValueError(
            f"node {node.id!r} has no checkpoint — call checkpoint() before restore()"
        )
    await environment.restore(snap)


async def branch(
    tree: RolloutTree, node: RolloutNode, environment: Any, n: int
) -> list[RolloutNode]:
    """Checkpoint ``node`` then fork it into ``n`` children — the full Branch."""
    await checkpoint(node, environment)
    return fork(tree, node, n)


def aggregate(node: RolloutNode) -> float:
    """V(node) — the mean of the children's returns.

    Each child carries its return in ``state["reward"]`` (written by the Reward
    plane after a child rollout is scored). Averaging them estimates the value
    of ``node``'s state — a reward function become a value function.
    """
    if not node.children:
        raise ValueError(f"node {node.id!r} has no children to aggregate")
    returns = [float(child.state.get(_REWARD_KEY, 0.0)) for child in node.children]
    return sum(returns) / len(returns)
