"""The branch transaction — checkpoint, per-child restore/run, state scoping.

The execution heart of a fork (the Branch lifecycle's steps 2 and 3): once
:mod:`benchflow.branch_policy` has admitted the request, this module owns
everything that must happen *transactionally* so the parent comes out exactly
as it went in — the composed checkpoint at the branch point
(:func:`checkpoint_parent`), the scoped capture of the parent's linear and
result-bearing state (:class:`LinearState`), and the child loop
(:meth:`BranchTransaction.run_children`): restore the checkpointed layers,
reset the linear state, point the cursor at a pending child node, run the
child under the runner :mod:`benchflow.branch_children` selects, record its
reward / unscored reason / wall clock on the node, and hand its shared-mount
output into custody before the next child can inherit or destroy it.

:class:`BranchTransaction` is one fork's execution context as a value — the
fourteen positional/keyword arguments the old ``_run_children`` took, named
once. The orchestrator (:func:`benchflow.rollout_branch.branch`) builds it
after quiesce + checkpoint and brackets it with artifact custody and the
final linear-state restore.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from benchflow.branch import UNSCORED_KEY, StageSnapshot, UnscoredChildError
from benchflow.branch import adopt_checkpoint as _adopt_checkpoint
from benchflow.branch import checkpoint as _checkpoint_branch
from benchflow.branch import checkpoint_composed as _checkpoint_composed
from benchflow.branch import restore as _restore_branch
from benchflow.branch import restore_composed as _restore_composed
from benchflow.branch_artifacts import MountedArtifacts, child_mount_dir
from benchflow.branch_children import (
    EXECUTION_FRESH_ROLLOUT,
    ChildRunner,
    make_fresh_child_runner,
    select_child_runner,
)
from benchflow.branch_delta import BranchDelta
from benchflow.branch_result import (
    scope_child_result_state,
    write_in_place_child_result,
)
from benchflow.models import TrajectorySource
from benchflow.trajectories.tree import RolloutNode

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)

#: Node-state key holding how long a branch child took, in seconds. Recorded
#: on the child node (even when the child raised) so a caller comparing
#: children — ``bench eval ablate``'s per-arm cost column — can attribute time
#: without re-deriving it from per-child artifacts, which only a
#: fresh-rollout child leaves behind. Deliberately *not* serialized by
#: :mod:`benchflow.branch_lineage`: a measured duration would make
#: ``tree.json`` non-deterministic.
CHILD_WALL_CLOCK_KEY = "wall_clock_sec"

#: The result-bearing Rollout attributes a child mutates that are not part of
#: the linear execution state above. Derived from an audit of everything
#: :meth:`Rollout._build_result` reads, against what a child's
#: ``connect() -> execute() -> verify() -> disconnect()`` writes:
#:
#: * ``_timing`` — ``execute()`` *accumulates* ``agent_execution`` and
#:   ``verify()`` writes into the same dict, so without this the parent's
#:   ``timing.json`` reported the parent's time plus every arm's (observed
#:   live by @Galius5136 on a two-arm ``pre-verify`` ablation).
#: * ``_verifier_error`` — ``verify()`` assigns it unconditionally, so a
#:   clean parent inherited the last child's verifier failure, and a failed
#:   parent had its own diagnostic erased by a child that scored cleanly.
#: * ``_diagnostics`` — ``execute()`` records prompt-timeout diagnostics and
#:   ``verify()`` records verifier-timeout ones; a child's timeout would
#:   surface in the parent's ``result.json`` as the parent's own.
#: * ``_native_usage_metrics`` / ``_native_usage_checkpoint`` — ``execute()``
#:   accumulates native ACP token usage into the first and re-bases the
#:   second; ``cleanup()``'s ``_finalize_usage_metrics`` then promotes the
#:   first into ``_usage_metrics``, i.e. the children's tokens are billed to
#:   the parent. The same accumulation bug as ``_timing``, one field further
#:   from the result.
#:
#: The remaining four are result-bearing and reachable from a caller-supplied
#: ``run_child`` (which may drive any phase it likes), so they are scoped for
#: the same reason even though the engine's own runners do not write them:
#: ``_error``, ``_export_error``, ``_evolved_skills``, ``_usage_metrics``.
#:
#: Audited and deliberately *not* scoped: ``_rollout_dir`` / ``_rollout_name``
#: / ``_resolved_prompts`` / ``_task_skill_policy`` (written by ``setup()``,
#: which no child re-runs on the shared instance), ``_agent_name`` (written by
#: ``connect()``, but from the parent's own config — a child cannot change what
#: it resolves to), ``_started_at`` (the parent's start is the parent's start,
#: and rolling it back per child would be wrong anyway), ``_terminal_timeout``
#: (only ``_record_agent_timeout`` writes it, which lives in ``run()``), and
#: the ``_provider_*_cached`` trio (written by ``cleanup()`` only). A
#: fresh-rollout child touches none of these: it drives a Rollout of its own.
_RESULT_STATE_FIELDS: tuple[str, ...] = (
    "_timing",
    "_verifier_error",
    "_diagnostics",
    "_native_usage_metrics",
    "_native_usage_checkpoint",
    "_error",
    "_export_error",
    "_evolved_skills",
    "_usage_metrics",
)


@dataclass
class LinearState:
    """A scoped snapshot of a Rollout's linear (non-tree) execution state.

    Captured before a branch child runs and restored after — this is what
    makes a branch child an *isolated sub-rollout* rather than a re-entrant
    mutation of the shared Rollout instance. The stage registry (RFC §3.2) is
    scoped the same way: a child that runs through its own ``pre-verify``
    boundary records that snapshot on its own node, and it must not become the
    *parent's* pre-verify — a later branch there would fork the child's world.

    Two kinds of state, scoped for two different reasons. The named fields are
    the *execution* state — where the rollout is and what it has done. The
    ``result_state`` bag is the **result-bearing** state
    (:data:`_RESULT_STATE_FIELDS`): fields no child needs but every in-place
    child writes, which end up in the parent's own ``result.json`` /
    ``timing.json`` if they are left where the last child put them. An
    in-place child continues the *shared* Rollout, so "the parent's linear
    state is exactly what it was before" has to cover what the parent
    *reports*, not only where its cursor sits.

    Everything mutable in that bag is deep-copied in both directions.
    ``restore_onto`` runs once per child — not only at the end — so handing a
    child the captured dict itself would let child *k* mutate the snapshot
    child *k+1* and the parent are both restored from.
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
    executed_prompts: list[str]
    stage_snapshots: dict[str, StageSnapshot]
    stage_nodes: dict[str, RolloutNode]
    result_state: dict[str, Any]

    @classmethod
    def capture(cls, rollout: Rollout) -> LinearState:
        """Snapshot ``rollout``'s linear state — a shallow copy of the trajectory."""
        return cls(
            cursor=rollout._cursor,
            trajectory=list(rollout._trajectory),
            n_tool_calls=rollout._n_tool_calls,
            phase=rollout._phase,
            rewards=copy.deepcopy(rollout._rewards),
            trajectory_source=rollout._trajectory_source,
            partial_trajectory=rollout._partial_trajectory,
            session_tool_count=getattr(rollout, "_session_tool_count", 0),
            session_traj_count=getattr(rollout, "_session_traj_count", 0),
            executed_prompts=list(rollout._executed_prompts),
            stage_snapshots=dict(getattr(rollout, "_stage_snapshots", {})),
            stage_nodes=dict(getattr(rollout, "_stage_nodes", {})),
            # hasattr, not a default: tests build a Rollout through
            # ``__new__()`` (the established pattern in rollout.py), and
            # inventing an attribute the instance never had would be a
            # different kind of mutation. What is absent stays absent.
            result_state={
                name: copy.deepcopy(getattr(rollout, name))
                for name in _RESULT_STATE_FIELDS
                if hasattr(rollout, name)
            },
        )

    def restore_onto(self, rollout: Rollout) -> None:
        """Write this snapshot back onto ``rollout`` — undoing a child's mutations."""
        rollout._cursor = self.cursor
        rollout._trajectory = list(self.trajectory)
        rollout._n_tool_calls = self.n_tool_calls
        rollout._phase = self.phase
        rollout._rewards = copy.deepcopy(self.rewards)
        rollout._trajectory_source = self.trajectory_source
        rollout._partial_trajectory = self.partial_trajectory
        rollout._session_tool_count = self.session_tool_count
        rollout._session_traj_count = self.session_traj_count
        rollout._executed_prompts = list(self.executed_prompts)
        rollout._stage_snapshots = dict(self.stage_snapshots)
        rollout._stage_nodes = dict(self.stage_nodes)
        for name, value in self.result_state.items():
            setattr(rollout, name, copy.deepcopy(value))


async def checkpoint_parent(
    rollout: Rollout,
    parent: RolloutNode,
    *,
    stage_snapshot: StageSnapshot | None,
    composed: bool,
    snap_env: Any,
    snap_sandbox: Any,
    layers: frozenset[str],
) -> None:
    """Record the fork's roll-back point on ``parent`` (Branch step 2).

    A stage branch adopts the snapshot the stage boundary already took —
    re-snapshotting there would capture a world that has since moved on. A
    cursor branch takes it now: the composed checkpoint when more than the
    legacy environment layer is requested, else the legacy environment-only
    path (same behavior, same bare ``StateSnapshot`` shape on the node). A
    checkpoint failure fails the branch closed with a note naming the layers —
    no partial checkpoint is recorded on the node.
    """
    if stage_snapshot is not None:
        _adopt_checkpoint(parent, stage_snapshot)
        return
    try:
        if composed:
            await _checkpoint_composed(
                parent, environment=snap_env, sandbox=snap_sandbox
            )
        else:
            await _checkpoint_branch(parent, rollout._environment)
    except Exception as exc:
        exc.add_note(
            f"branch(snapshot_layers={sorted(layers)!r}) failed during "
            "checkpoint — the branch fails closed; no partial checkpoint "
            "was recorded on the node."
        )
        raise


@dataclass
class BranchTransaction:
    """One fork's execution context — the parameters of the child loop, named.

    Built by :func:`benchflow.rollout_branch.branch` after quiesce and
    checkpoint; :meth:`run_children` is the successor of the old
    ``_run_children`` free function, whose fourteen arguments live here as
    fields instead. ``children`` is the transaction's output: every child is
    appended as soon as it is attached, so a caller that catches a propagating
    child failure still sees the nodes that were forked — which is how ``bench
    eval ablate`` reports the arm that raised and the arms that never ran.
    """

    rollout: Rollout
    n: int
    deltas: Sequence[BranchDelta | None] | None
    parent: RolloutNode
    saved: LinearState
    runner: ChildRunner
    run_child: ChildRunner | None
    composed: bool
    snap_env: Any
    snap_sandbox: Any
    fresh_children: bool
    run_dir: Any
    holder: MountedArtifacts | None
    children: list[RolloutNode] = field(default_factory=list)
    # Injected by the orchestrator from *its* module globals so
    # ``benchflow.rollout_branch`` stays the call-time resolution point — the
    # monkeypatch seam tests/test_branch_child_result.py pins (a fake
    # fresh-child runner factory, a recording in-place result writer). The
    # defaults are the canonical implementations for direct users.
    fresh_runner_factory: Callable[..., ChildRunner] = make_fresh_child_runner
    write_child_result: Callable[..., None] = write_in_place_child_result

    async def run_children(self) -> None:
        """Run the fork's ``n`` children in order, appending to ``children``.

        Each iteration is the same bracket: attach a *pending* child node
        carrying its delta provenance, restore the checkpointed layers and the
        parent's linear state, run the child under the runner the boundary
        selects, record its outcome on the node, and hand its shared-mount
        output into custody. An **in-place** child of a run-dir-bearing
        rollout additionally leaves its own ``result.json`` set
        (:mod:`benchflow.branch_result`): its result-bearing state is scoped
        to zero before it runs — so what the fields hold afterwards is the
        child's own, never the parent's showing through — and the result is
        synthesized from that state after the child completes, *before* the
        next restore discards it. A fresh-rollout child writes its own result
        already and is left alone; a child that raised ends the fork and is
        evidenced by its ``mounted/`` archive instead.
        """
        in_place_results = not self.fresh_children and self.run_dir is not None
        for index in range(self.n):
            delta = self.deltas[index] if self.deltas is not None else None
            child = self._attach_pending_child(delta)
            await self._restore_for(child, scope_result_state=in_place_results)
            started_wall = datetime.now()
            child_runner = select_child_runner(
                self.rollout,
                delta=delta,
                default=self.runner,
                run_child=self.run_child,
                fresh_children=self.fresh_children,
                parent=self.parent,
                run_dir=self.run_dir,
                fresh_runner_factory=self.fresh_runner_factory,
            )
            await self._run_child(child, child_runner)
            if in_place_results:
                # The instance still carries the child's own state (the
                # restore happens on the next iteration / at the end of the
                # fork), so this is the one window where a first-class result
                # can be built for an in-place child. Best-effort by contract:
                # never raises.
                self.write_child_result(
                    self.rollout,
                    parent=self.parent,
                    child=child,
                    run_dir=self.run_dir,
                    started_at=started_wall,
                    base_trajectory_len=len(self.saved.trajectory),
                    base_prompt_count=len(self.saved.executed_prompts),
                    base_tool_calls=self.saved.n_tool_calls,
                )
            if self.holder is not None:
                # Custody fails closed: a hand-off failure means the mounts
                # may still carry this child's files, which the next child
                # would both inherit and destroy. The child's own outcome
                # (reward / unscored reason / result.json) is recorded above,
                # so raising here loses no observation — the fork stops before
                # evidence can cross arms.
                self.holder.raise_pending()

    def _attach_pending_child(self, delta: BranchDelta | None) -> RolloutNode:
        """Attach a *pending* branch-child node carrying its delta provenance.

        The child's real continuation Step is filled in place by its first
        execute(), so the child's work lands on the child node, not a
        descendant placeholder. The delta provenance is recorded on the node
        itself at fork time (``None`` = the zero delta), so lineage
        serialization reads it from the node — never by positional alignment —
        and a second branch() at the same parent can never misattribute
        deltas. How the delta is *executed* is recorded for the same reason:
        an env-ready child re-runs installation as its own Rollout, and the
        artifacts must say so rather than leaving a reader to infer it from
        the delta — which they could not, since a zero-delta child of that
        boundary is a fresh rollout too.
        """
        child = self.rollout._tree.attach(self.parent)
        child.state["delta"] = (
            delta if delta is not None else BranchDelta()
        ).provenance_dict()
        if self.fresh_children:
            child.state["delta_execution"] = EXECUTION_FRESH_ROLLOUT
        self.children.append(child)
        return child

    async def _restore_for(
        self, child: RolloutNode, *, scope_result_state: bool
    ) -> None:
        """Roll the world and the parent's linear state back for one child.

        Restore the checkpointed layers (sandbox first, then env — the reverse
        of checkpoint order), reset the parent's linear state, and point the
        cursor at the pending child for the sub-rollout. For an in-place child
        the restore just put the *parent's* result-bearing values back; they
        are scoped to zero so the child's result reports only what the child
        itself produced — the next restore brings the parent's back.
        """
        if self.composed:
            await _restore_composed(
                self.parent, environment=self.snap_env, sandbox=self.snap_sandbox
            )
        else:
            await _restore_branch(self.parent, self.rollout._environment)
        self.saved.restore_onto(self.rollout)
        self.rollout._cursor = child
        if scope_result_state:
            scope_child_result_state(self.rollout)

    async def _run_child(self, child: RolloutNode, child_runner: ChildRunner) -> None:
        """Run one child and record its outcome on its node.

        A child that ran but was never scored records *why*
        (:data:`~benchflow.branch.UNSCORED_KEY`), keeps its reward unset, and
        does not end the fork: the next child restores the checkpoint for
        itself, so one lost score does not cost the fork. The wall clock and
        the mount hand-off happen in a ``finally`` so a child that raised
        still reports what it cost — and still hands off whatever it wrote to
        the shared mounts (a crashed child's verifier output is often the only
        evidence of why it crashed). Handing off also empties the mounts, so
        the next child cannot inherit this one's files. ``hand_off`` never
        raises here — a raise inside this finally would replace the child's
        own exception — it records the failure on the holder, and
        ``raise_pending()`` in the loop surfaces it once the child's outcome
        is safely on the node.
        """
        started = time.monotonic()
        unscored: str | None = None
        try:
            ret = await child_runner(child)
        except UnscoredChildError as exc:
            unscored = exc.reason
        finally:
            child.state[CHILD_WALL_CLOCK_KEY] = time.monotonic() - started
            if self.holder is not None and self.run_dir is not None:
                self.holder.hand_off(
                    child_mount_dir(self.run_dir, self.parent.id, child.id)
                )
        if unscored is not None:
            child.state[UNSCORED_KEY] = unscored
            logger.error("branch child %s is unscored: %s", child.id, unscored)
        else:
            child.state["reward"] = float(ret)
