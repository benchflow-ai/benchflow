"""The Branch operation — the credit-assignment engine of the tree-native Rollout.

Han's insight: a checkpoint forks one state into N continuations, which turns a
reward function into a value function V(s). A Branch is four operations on a
``RolloutTree`` node and an ``Environment``:

* ``checkpoint`` — snapshot the environment; the roll-back point for the fork.
* ``fork`` — split the node into N child continuations.
* ``restore`` — roll the environment back to a node's checkpoint.
* ``aggregate`` — average the children's returns into V(node).

``checkpoint_composed`` / ``restore_composed`` are the layered variants from
the rollout-branching RFC (§3.1): they compose the container layer
(``Sandbox.snapshot``) with the environment-state layer into one
:class:`StageSnapshot`, in a fixed order — environment first on checkpoint,
sandbox first on restore.

These are the Branch *operations*. Wiring them into the rollout engine — running
the forked children as sub-rollouts, quiescing the agent first — is the engine's
job; these primitives stay pure and independently testable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from benchflow.environment.protocol import StateSnapshot
from benchflow.sandbox.protocol import SandboxImage
from benchflow.trajectories.tree import RolloutNode, RolloutTree, Step

_SNAPSHOT_KEY = "snapshot"
_REWARD_KEY = "reward"


@dataclass(frozen=True)
class StageSnapshot:
    """A composed checkpoint — one ref per snapshot layer (RFC §3.1).

    ``environment_ref`` and ``sandbox_ref`` each hold that layer's roll-back
    handle iff the layer was requested at checkpoint time; ``None`` means the
    layer was *not requested*, and :func:`restore_composed` rejects a live
    object for it. ``stage`` optionally names the lifecycle boundary the
    snapshot was taken at (``env-ready``, ``pre-verify``, ...); ``meta``
    carries diagnostics.
    """

    environment_ref: StateSnapshot | None
    sandbox_ref: SandboxImage | None
    stage: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


async def checkpoint(node: RolloutNode, environment: Any) -> StateSnapshot:
    """Snapshot the environment at ``node`` and record the snapshot on it.

    The recorded ``StateSnapshot`` is the roll-back point every child forked
    from ``node`` restores to before it runs.
    """
    snap = await environment.snapshot()
    node.state[_SNAPSHOT_KEY] = snap
    return snap


async def checkpoint_composed(
    node: RolloutNode,
    *,
    environment: Any = None,
    sandbox: Any = None,
    stage: str | None = None,
) -> StageSnapshot:
    """Snapshot the requested layers at ``node`` — the composed checkpoint.

    Layer order is fixed (RFC §3.1): ``environment.snapshot()`` first, then
    ``sandbox.snapshot()``. A layer is included iff its live object is passed
    (``None`` = layer not requested); quiescing the agent first is the
    engine's job, not this op's. If either layer's snapshot raises, the error
    propagates and *nothing* is recorded on ``node`` — a partial
    :class:`StageSnapshot` is never a roll-back point.
    """
    if environment is None and sandbox is None:
        raise ValueError(
            "checkpoint_composed() needs at least one layer — pass "
            "environment=, sandbox=, or both"
        )
    env_ref = await environment.snapshot() if environment is not None else None
    sandbox_ref = await sandbox.snapshot() if sandbox is not None else None
    snap = StageSnapshot(environment_ref=env_ref, sandbox_ref=sandbox_ref, stage=stage)
    node.state[_SNAPSHOT_KEY] = snap
    return snap


def adopt_checkpoint(node: RolloutNode, snapshot: StageSnapshot) -> RolloutNode:
    """Record an already-taken :class:`StageSnapshot` as ``node``'s roll-back point.

    The stage-boundary policy (RFC §3.2) snapshots at a lifecycle boundary and
    branches from it later; when it does, the recorded snapshot *is* the
    node's checkpoint and must not be re-taken (the world has moved on since).
    This is the one seam that writes a checkpoint the node did not take
    itself, so the snapshot key stays private to this module.
    """
    node.state[_SNAPSHOT_KEY] = snapshot
    return node


def fork(tree: RolloutTree, node: RolloutNode, n: int) -> list[RolloutNode]:
    """Fork ``node`` into ``n`` child continuations, making it a branch point."""
    if n < 2:
        raise ValueError(f"a branch forks into >= 2 children, got n={n}")
    return [tree.advance(node, Step(id=f"{node.id}-branch-{i}")) for i in range(n)]


async def restore(node: RolloutNode, environment: Any) -> None:
    """Roll the environment back to ``node``'s recorded checkpoint.

    Legacy environment-only restore: it accepts the bare ``StateSnapshot``
    recorded by :func:`checkpoint`. A :class:`StageSnapshot` recorded by
    :func:`checkpoint_composed` is rejected *before* the environment is
    touched — its layer refs are not an environment snapshot, and passing
    one through would fail deep inside the environment implementation.
    """
    snap = node.state.get(_SNAPSHOT_KEY)
    if snap is None:
        raise ValueError(
            f"node {node.id!r} has no checkpoint — call checkpoint() before restore()"
        )
    if isinstance(snap, StageSnapshot):
        raise ValueError(
            f"node {node.id!r} was checkpointed with checkpoint_composed(); "
            "use restore_composed() to roll back a composed StageSnapshot"
        )
    await environment.restore(snap)


async def restore_composed(
    node: RolloutNode,
    *,
    environment: Any = None,
    sandbox: Any = None,
) -> None:
    """Roll the requested layers back to ``node``'s composed checkpoint.

    Restore order is the reverse of checkpoint (RFC §3.1):
    ``sandbox.restore()`` first, then ``environment.restore()``. Accepts both
    checkpoint shapes on ``node.state["snapshot"]`` — a legacy bare
    :class:`StateSnapshot` recorded by :func:`checkpoint` (environment-only)
    or a :class:`StageSnapshot`. Every layer present in the checkpoint must be
    matched by its live object and vice versa; a mismatch is a caller bug and
    raises ``ValueError`` before either layer is touched.
    """
    snap = node.state.get(_SNAPSHOT_KEY)
    if snap is None:
        raise ValueError(
            f"node {node.id!r} has no checkpoint — call checkpoint_composed() "
            "before restore_composed()"
        )
    if isinstance(snap, StateSnapshot):
        # Legacy shape: checkpoint() recorded a bare env-state snapshot.
        snap = StageSnapshot(environment_ref=snap, sandbox_ref=None)
    elif not isinstance(snap, StageSnapshot):
        raise ValueError(
            f"node {node.id!r} holds an unrecognized checkpoint of type "
            f"{type(snap).__name__!r} — expected StateSnapshot or StageSnapshot"
        )
    # Validate both layers before restoring either — never a partial restore.
    for layer, ref, live in (
        ("sandbox", snap.sandbox_ref, sandbox),
        ("environment", snap.environment_ref, environment),
    ):
        if ref is not None and live is None:
            raise ValueError(
                f"node {node.id!r}'s checkpoint has a {layer} layer but no "
                f"live {layer} was passed to restore_composed()"
            )
        if ref is None and live is not None:
            raise ValueError(
                f"a live {layer} was passed to restore_composed() but node "
                f"{node.id!r}'s checkpoint has no {layer} layer"
            )
    if snap.sandbox_ref is not None:
        await sandbox.restore(snap.sandbox_ref)
    if snap.environment_ref is not None:
        await environment.restore(snap.environment_ref)


async def branch(
    tree: RolloutTree, node: RolloutNode, environment: Any, n: int
) -> list[RolloutNode]:
    """Checkpoint ``node`` then fork it into ``n`` children — the full Branch."""
    await checkpoint(node, environment)
    return fork(tree, node, n)


def aggregate(node: RolloutNode, *, over: Sequence[RolloutNode] | None = None) -> float:
    """V(node) — the mean of the children's returns.

    Each child carries its return in ``state["reward"]`` (written by the Reward
    plane after a child rollout is scored). Averaging them estimates the value
    of ``node``'s state — a reward function become a value function.

    ``over`` narrows the average to one fork's children. It matters once a
    branch point can be a node that already has children: a stage-boundary
    branch (RFC §3.2) forks the node the stage was captured on, which may
    already carry the linear continuation — and that linear child has no
    ``reward``, so averaging over ``node.children`` would silently drag V
    toward zero. ``None`` keeps the every-child default.
    """
    children = list(node.children if over is None else over)
    if not children:
        raise ValueError(f"node {node.id!r} has no children to aggregate")
    returns = [float(child.state.get(_REWARD_KEY, 0.0)) for child in children]
    return sum(returns) / len(returns)
