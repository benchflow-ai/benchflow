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
from benchflow.rewards.events import RewardEvent
from benchflow.rewards.node import verify_result_from_reward_map
from benchflow.rewards.protocol import VerifyResult
from benchflow.trajectories.tree import RolloutNode, RolloutTree, Step

_SNAPSHOT_KEY = "snapshot"
_REWARD_KEY = "reward"
_VERIFY_RESULT_KEY = "verify_result"


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
    """V(node) — the mean of the children's returns (scalar form).

    Each child carries its return in ``state["reward"]`` (written by the Reward
    plane after a child rollout is scored). Averaging them estimates the value
    of ``node``'s state — a reward function become a value function. Kept for
    back-compat; :func:`aggregate_verify_result` is the canonical form.
    """
    if not node.children:
        raise ValueError(f"node {node.id!r} has no children to aggregate")
    returns = [float(child.state.get(_REWARD_KEY, 0.0)) for child in node.children]
    return sum(returns) / len(returns)


def aggregate_verify_result(node: RolloutNode) -> VerifyResult:
    """V(node) as a composed ``VerifyResult`` — the canonical value function.

    The node-scored counterpart to :func:`aggregate`: instead of averaging bare
    floats, it reads each child's ``VerifyResult`` (recorded under
    ``child.state["verify_result"]`` by the Reward plane after the child is
    scored) and composes them into V(node) — mean child reward, per-child
    ``items``, concatenated ``events``. This carries the architecture's Reward
    contract end to end ("from a reward function to a value function"), so a
    branch's value estimate is a tagged ``VerifyResult``, not a lossy scalar.

    A child that was scored with only a bare float (a custom runner, legacy
    path) is tolerated: its ``state["reward"]`` is lifted into a minimal
    ``VerifyResult``. The parent ``error`` is conservative — populated unless
    *every* child scored cleanly — so a partial scoring failure can't masquerade
    as a clean value.
    """
    if not node.children:
        raise ValueError(f"node {node.id!r} has no children to aggregate")
    results: list[VerifyResult] = []
    for child in node.children:
        vr = child.state.get(_VERIFY_RESULT_KEY)
        if not isinstance(vr, VerifyResult):
            # One float->VerifyResult lift policy across the codebase (the same
            # one branch()'s runner-normalisation uses), so a bare-float child
            # composes identically however it was scored.
            vr = verify_result_from_reward_map(
                {"reward": float(child.state.get(_REWARD_KEY, 0.0))}
            )
        results.append(vr)
    mean = sum(r.reward for r in results) / len(results)
    items: dict[str, float] = {}
    events: list[RewardEvent] = []
    for i, r in enumerate(results):
        items[f"child-{i}"] = r.reward
        for source, value in r.items.items():
            items[f"child-{i}/{source}"] = value
        events.extend(r.events)
    failed = sum(1 for r in results if r.error is not None)
    error = None if failed == 0 else f"{failed}/{len(results)} children failed scoring"
    return VerifyResult(
        reward=mean,
        items=items,
        events=events,
        error=error,
        space="output",
        granularity="terminal",
    )
