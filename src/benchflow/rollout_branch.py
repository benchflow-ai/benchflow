"""The Branch -> Rollout engine wiring.

The pure Branch primitives live in :mod:`benchflow.branch` — ``checkpoint``,
``restore``, ``aggregate`` operate on a ``RolloutTree`` node and an
``Environment`` with no I/O beyond the env contract. This module is the
*engine*: it drives those primitives against a live
:class:`~benchflow.rollout.Rollout` — quiescing the agent, running each forked
child as an **isolated sub-rollout**, and restoring the parent's linear state
afterward.

Why a separate module: ``rollout.py`` is the 5-phase lifecycle; the Branch
path is a distinct, optional capability. Keeping it here holds ``rollout.py``
under the size threshold and keeps the branch logic independently testable.

The engine functions are free functions taking a ``Rollout`` as their first
argument — ``Rollout.branch`` is a thin one-line entry point that delegates
here.

Isolation invariant (the architecture's "tree is additive / no-regression"):
after :func:`branch` returns, the parent Rollout's linear state — ``_cursor``,
``_trajectory``, ``_rewards``, ``_phase``, ``_n_tool_calls`` (and the session
bookkeeping) — is *exactly* what it was before. A branch child never
re-entrantly mutates the shared instance: it runs against a scoped snapshot of
that state, captured before and restored after each child, and its real
continuation Steps attach to a *pending* branch-child node so the reward and
value land on the right node.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchflow.branch import aggregate as _aggregate_branch
from benchflow.branch import checkpoint as _checkpoint_branch
from benchflow.branch import restore as _restore_branch
from benchflow.models import TrajectorySource
from benchflow.trajectories.tree import RolloutNode

if TYPE_CHECKING:
    from benchflow.rollout import Rollout

# The per-child runner: given the child's branch node, run its continuation and
# return the scalar return. No ``int`` index — a caller that needs per-child
# prompts binds them into a closure (see ``run_child`` in :func:`branch`).
ChildRunner = Callable[[RolloutNode], Awaitable[float]]


@dataclass
class _LinearState:
    """A scoped snapshot of a Rollout's linear (non-tree) execution state.

    Captured before a branch child runs and restored after — this is what
    makes a branch child an *isolated sub-rollout* rather than a re-entrant
    mutation of the shared Rollout instance.
    """

    cursor: RolloutNode
    trajectory: list[dict]
    n_tool_calls: int
    phase: str
    rewards: dict | None
    trajectory_source: TrajectorySource | None
    partial_trajectory: bool
    session_tool_count: int
    session_traj_count: int

    @classmethod
    def capture(cls, rollout: Rollout) -> _LinearState:
        """Snapshot ``rollout``'s linear state — a shallow copy of the trajectory."""
        return cls(
            cursor=rollout._cursor,
            trajectory=list(rollout._trajectory),
            n_tool_calls=rollout._n_tool_calls,
            phase=rollout._phase,
            rewards=rollout._rewards,
            trajectory_source=rollout._trajectory_source,
            partial_trajectory=rollout._partial_trajectory,
            session_tool_count=getattr(rollout, "_session_tool_count", 0),
            session_traj_count=getattr(rollout, "_session_traj_count", 0),
        )

    def restore_onto(self, rollout: Rollout) -> None:
        """Write this snapshot back onto ``rollout`` — undoing a child's mutations."""
        rollout._cursor = self.cursor
        rollout._trajectory = list(self.trajectory)
        rollout._n_tool_calls = self.n_tool_calls
        rollout._phase = self.phase
        rollout._rewards = self.rewards
        rollout._trajectory_source = self.trajectory_source
        rollout._partial_trajectory = self.partial_trajectory
        rollout._session_tool_count = self.session_tool_count
        rollout._session_traj_count = self.session_traj_count


async def branch(
    rollout: Rollout,
    n: int,
    run_child: ChildRunner | None = None,
    *,
    require_sandbox_snapshot: bool = False,
) -> float:
    """Branch ``rollout`` at its cursor into ``n`` child continuations.

    The Branch lifecycle (``docs/architecture.md``, "Lifecycles"):

    1. ``quiesce`` — pause the agent at a stable point (disconnect ACP).
    2. ``checkpoint`` — snapshot the Environment at the cursor; the roll-back
       point every child restores to.
    3. ``run children`` — for each child, ``restore`` the env to the
       checkpoint, then run the continuation as an **isolated sub-rollout**:
       its own scoped linear state, a fresh agent session. The child's real
       continuation Steps attach directly to its branch node (a *pending* node,
       no content-free placeholder Step), so the reward lands on the real leaf.
    4. ``score / aggregate`` — each child's return is recorded on
       ``child.state["reward"]``; their mean is V(parent), recorded on
       ``parent.state["value"]`` and returned.

    The branch point is always the current cursor. ``run_child`` is the
    per-child runner — injected for unit tests; the default
    (:func:`make_default_runner`) restores the env, connects a fresh agent,
    runs the continuation, scores it, and disconnects. A caller that needs
    per-child prompts binds them into the ``run_child`` closure.

    After this returns, ``rollout``'s linear state is exactly what it was
    before — the tree gained ``n`` children at the cursor, nothing else moved.
    """
    if rollout._environment is None:
        raise RuntimeError(
            "branch() needs the Environment plane — there is no world to "
            "snapshot. Pass RolloutConfig(environment_manifest=...)."
        )
    if n < 2:
        raise ValueError(f"a branch forks into >= 2 children, got n={n}")

    # Fail closed when the run requires a three-layer checkpoint but the
    # sandbox cannot snapshot the container layer (#384). The Branch
    # lifecycle composes container ⊃ environment-state ⊃ agent-session;
    # without the container layer, restoring only environment state can
    # produce inconsistent state for runs that mutate process/service state
    # the Environment manifest does not capture.
    if require_sandbox_snapshot:
        sandbox = getattr(rollout, "_env", None)
        supports = getattr(sandbox, "supports_snapshot", False)
        if not supports:
            sandbox_name = type(sandbox).__name__ if sandbox else "<none>"
            raise RuntimeError(
                f"branch(require_sandbox_snapshot=True) cannot run: the active "
                f"sandbox {sandbox_name!r} does not implement container-level "
                "snapshot/restore. Use a provider whose Sandbox satisfies the "
                "checkpoint contract (DockerSandbox or DaytonaSandbox in direct "
                "mode), or drop require_sandbox_snapshot if Environment-state "
                "checkpoint is sufficient for this run."
            )

    parent = rollout._cursor
    runner = run_child if run_child is not None else make_default_runner(rollout)

    # quiesce — pause the agent before snapshotting so the checkpoint is
    # consistent (the Branch lifecycle quiesces first).
    await rollout.disconnect()

    # checkpoint — snapshot the env at the parent; the roll-back point.
    await _checkpoint_branch(parent, rollout._environment)

    # The parent's linear state, captured once. Each child runs against a fresh
    # restore of this; the parent is restored to it at the end.
    saved = _LinearState.capture(rollout)

    for _ in range(n):
        # Attach a *pending* branch-child node — its real continuation Step is
        # filled in place by the child's first execute(), so the child's work
        # lands on the child node, not a descendant placeholder.
        child = rollout._tree.attach(parent)

        # restore the env to the parent's checkpoint, reset the parent's linear
        # state, and point the cursor at the pending child for the sub-rollout.
        await _restore_branch(parent, rollout._environment)
        saved.restore_onto(rollout)
        rollout._cursor = child

        ret = await runner(child)
        child.state["reward"] = float(ret)

    # restore the parent's linear state — the tree grew, nothing else moved.
    saved.restore_onto(rollout)

    # aggregate — per-child return -> V(parent).
    value = _aggregate_branch(parent)
    parent.state["value"] = value
    rollout._phase = "branched"
    return value


def make_default_runner(rollout: Rollout) -> ChildRunner:
    """Build the default per-child runner bound to ``rollout``.

    The default runner re-runs the child from the parent's env checkpoint with
    a *fresh agent session* — agent-session snapshot is the unsolved hard part
    (``docs/architecture.md``, "The hard part"), so the agent restarts per
    child. Each child connects a fresh agent and disconnects it at the end, so
    no two children's agents overlap (the next child connects only after the
    previous one disconnected). ``verify()`` returning ``None`` or an empty
    dict falls back to a ``0.0`` return.
    """

    async def _runner(child: RolloutNode) -> float:
        await rollout.connect()
        # Fill the pending branch-child node in place — the continuation Step
        # lands on `child` itself, no content-free placeholder.
        await rollout.execute(node=child)
        rewards = await rollout.verify()
        await rollout.disconnect()
        if not rewards:
            return 0.0
        return float(rewards.get("reward", 0.0))

    return _runner
