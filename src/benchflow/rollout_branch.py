"""The Branch -> Rollout engine wiring — the fork orchestrator.

The pure Branch primitives live in :mod:`benchflow.branch` — ``checkpoint``,
``restore``, ``aggregate`` operate on a ``RolloutTree`` node and an
``Environment`` with no I/O beyond the env contract. This module is the
*orchestrator*: :func:`branch` drives one fork end to end against a live
:class:`~benchflow.rollout.Rollout` — admit the request, quiesce the agent,
checkpoint, run the children, restore the parent, aggregate, leave lineage —
delegating each concern to its focused module:

* :mod:`benchflow.branch_policy` — what may be captured and forked, and when:
  layer resolution and capability gating, the stage-snapshot capture path
  (:func:`~benchflow.branch_policy.capture_stage`), recorded-stage resolution,
  and the whole delta-vector gate. Everything fails closed before anything is
  quiesced.
* :mod:`benchflow.branch_transaction` — the transactional heart: the composed
  checkpoint at the branch point, the scoped
  :class:`~benchflow.branch_transaction.LinearState` capture that makes each
  child an isolated sub-rollout, and the per-child restore/run/record loop
  (:class:`~benchflow.branch_transaction.BranchTransaction`).
* :mod:`benchflow.branch_children` — how a child executes its delta: the
  in-place default runner, and (via :mod:`benchflow.branch_skill`) the
  fresh-rollout path every ``env-ready`` child takes.
* :mod:`benchflow.branch_report` — lineage and ablation reporting; the
  fork's failure-isolated lineage write stays here (:func:`_write_lineage`)
  because its writers must resolve through *this* module's namespace — the
  patch seam ``tests/test_rollout_branch.py`` pins.

``Rollout.branch`` / ``Rollout.mark_stage`` / ``Rollout.branch_at_stage`` are
thin entry points that delegate here, and the public names the engine has
always exported (``ChildRunner``, ``CHILD_WALL_CLOCK_KEY``, the gate
exceptions, ``capture_stage``, ``make_default_runner``, the fresh-child API)
are re-exported so existing import paths keep working.

Isolation invariant (the architecture's "tree is additive / no-regression"):
after :func:`branch` returns, the parent Rollout's linear state — ``_cursor``,
``_trajectory``, ``_rewards``, ``_phase``, ``_n_tool_calls`` (and the session
bookkeeping) — is *exactly* what it was before, and so is the result-bearing
state an in-place child writes on its way through (``_timing``,
``_verifier_error``, ``_diagnostics``, the usage counters — see
:data:`~benchflow.branch_transaction._RESULT_STATE_FIELDS`): the parent's own
``result.json`` describes the parent's run, never the last arm's. A branch
child never re-entrantly mutates the shared instance: it runs against a scoped
snapshot of that state, captured before and restored after each child, and its
real continuation Steps attach to a *pending* branch-child node so the reward
and value land on the right node.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from benchflow.branch import (  # noqa: F401  (StageSnapshot/UnscoredChildError re-exported)
    UNSCORED_KEY,
    StageSnapshot,
    UnscoredChildError,
)
from benchflow.branch import aggregate as _aggregate_branch
from benchflow.branch_artifacts import MountedArtifacts, child_mount_dir  # noqa: F401
from benchflow.branch_children import (  # noqa: F401  (re-exports)
    EXECUTION_FRESH_ROLLOUT,
    FRESH_CHILD_LAYER,
    FRESH_CHILD_STAGE,
    SKILL_DELTA_LAYER,
    SKILL_DELTA_STAGE,
    ChildRunner,
    make_default_runner,
    make_fresh_child_runner,
    resolve_child_skill_policy,
    resolve_environment_ref_delta,
)
from benchflow.branch_delta import BranchDelta, BranchDeltaNotSupported  # noqa: F401
from benchflow.branch_lineage import (
    write_branch_artifacts,
    write_stage_snapshots,
)
from benchflow.branch_policy import (  # noqa: F401  (re-exports)
    _EXECUTABLE_DELTA_FIELDS,
    _UNSUPPORTED_DELTA_FIELDS,
    BranchChildExecutionNotSupported,
    BranchParentSkillModeConflict,
    capture_stage,
    finalize_stage_snapshots,
    gate_layers,
    recorded_stage_checkpoint,
    resolve_layers,
    runs_fresh_children,
    validate_deltas,
    validate_fresh_children,
)
from benchflow.branch_result import (  # noqa: F401  (scope_… re-exported)
    scope_child_result_state,
    write_in_place_child_result,
)
from benchflow.branch_transaction import (  # noqa: F401  (wall-clock key re-exported)
    CHILD_WALL_CLOCK_KEY,
    BranchTransaction,
    LinearState,
    checkpoint_parent,
)

if TYPE_CHECKING:
    from benchflow.rollout import Rollout
    from benchflow.trajectories.tree import RolloutNode

logger = logging.getLogger(__name__)

#: Back-compat alias — tests and callers imported the scoped-state class under
#: its original private name.
_LinearState = LinearState


async def branch(
    rollout: Rollout,
    n: int,
    run_child: ChildRunner | None = None,
    *,
    require_sandbox_snapshot: bool = False,
    snapshot_layers: frozenset[str] | set[str] | None = None,
    deltas: Sequence[BranchDelta | None] | None = None,
    at_stage: str | None = None,
) -> float | None:
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
       ``child.state["reward"]`` (and its duration on
       :data:`CHILD_WALL_CLOCK_KEY`, in memory only); their mean is V(parent),
       recorded on ``parent.state["value"]`` and returned. A child that ran
       but produced no reward is recorded as *unscored*
       (:data:`~benchflow.branch.UNSCORED_KEY`) with no ``reward`` of its own,
       and makes the fork's value ``None`` — an unobserved score is never
       averaged in as a zero.

    The branch point is always the current cursor. ``run_child`` is the
    per-child runner — injected for unit tests; the default
    (:func:`make_default_runner`) restores the env, connects a fresh agent,
    runs the continuation, scores it, and disconnects. A caller that needs
    per-child prompts binds them into the ``run_child`` closure.

    ``snapshot_layers`` selects which checkpoint layers compose the roll-back
    point (RFC §3.1). The default ``{"environment"}`` keeps the legacy
    environment-state-only checkpoint — same behavior, same bare
    ``StateSnapshot`` shape on the node. Adding ``"sandbox"`` composes a
    container-level snapshot with it (environment first on checkpoint, sandbox
    first on restore); ``{"sandbox"}`` alone branches a stateless environment
    on the container layer only, never touching environment snapshot/restore.
    Missing capability on any requested layer fails closed before anything is
    snapshotted.

    ``deltas`` records the exactly-one-controlled-change each child runs
    under (RFC §3.3) — one :class:`BranchDelta` (or ``None`` = zero delta)
    per child, validated before anything runs (see
    :func:`~benchflow.branch_policy.validate_deltas` for the per-field gates
    and :mod:`benchflow.branch_children` for how each executable field runs).

    **How a child is executed depends on the boundary, not on the delta.**
    ``at_stage="env-ready"`` precedes ``install_agent()``, so every child the
    engine runs from there is a *fresh rollout* over the restored sandbox
    (``use_prebuilt_env``) that installs the agent for itself — including a
    child with no delta at all, which re-installs the parent's own recorded
    skill mode. An in-place child there would connect to an agent the restore
    deleted, or score a world missing everything installation deploys and
    report it as an ordinary single-delta child; that fork therefore requires
    the container layer and raises
    :class:`~benchflow.branch_policy.BranchChildExecutionNotSupported` without
    it. At every other boundary the agent is already installed and children
    run in place.

    After this returns, ``rollout``'s linear state is exactly what it was
    before — the tree gained ``n`` children at the cursor, nothing else
    moved; when the rollout knows its run directory, the branch also leaves
    lineage artifacts (``tree.json``,
    ``branches/<branch-node-id>/children/<child-node-id>/``), with any
    artifact-write failure logged and isolated from the branch result.

    A child that raises anything other than
    :class:`~benchflow.branch.UnscoredChildError` ends the fork and propagates
    — but the parent's state is restored and the *partial* lineage is written
    on the way out, because the caller may report on it: ``bench eval ablate``
    catches that exception and publishes the completed, failed and skipped
    arms, and an experiment that reports arms it cannot evidence is worse than
    one that reports none.

    ``at_stage`` branches from a recorded stage boundary instead of
    checkpointing at the cursor (RFC §3.2): the stage's :class:`StageSnapshot`
    becomes the roll-back point every child restores to, the fork happens at
    the node that stage was captured on (so the children continue that stage's
    world, and V aggregates over this fork only), and the stage name — not the
    ``cursor:<id>`` fallback — is the ``branch_stage`` recorded in provenance.
    """
    stage_snapshot: StageSnapshot | None = None
    if at_stage is not None:
        stage_snapshot, stage_node, layers = recorded_stage_checkpoint(
            rollout, at_stage, snapshot_layers
        )
        subject = f"branch_at_stage({at_stage!r})"
    else:
        subject = "branch"
        layers = resolve_layers(
            frozenset({"environment"}) if snapshot_layers is None else snapshot_layers,
            subject=subject,
        )
    if n < 2:
        raise ValueError(f"a branch forks into >= 2 children, got n={n}")

    # Admit the request — the whole delta vector and the boundary's own gate
    # fail closed with nothing quiesced, checkpointed, or run.
    validate_deltas(
        rollout, deltas, n=n, at_stage=at_stage, layers=layers, run_child=run_child
    )
    fresh_children = runs_fresh_children(at_stage, run_child)
    if fresh_children:
        validate_fresh_children(layers, at_stage=str(at_stage))

    # Fail closed when a requested layer's plane or capability is missing
    # (#384). ``require_sandbox_snapshot`` keeps its original check-only
    # semantics; requesting the layer via ``snapshot_layers`` gates the same
    # way and then actually composes the sandbox into the checkpoint. A stage
    # branch gates too — the planes must still be live to restore what the
    # stage captured.
    snap_env, snap_sandbox = gate_layers(
        rollout,
        layers,
        subject=subject,
        requested=(
            "require_sandbox_snapshot=True"
            if require_sandbox_snapshot
            else f"snapshot_layers={sorted(layers)!r}"
        ),
        gate_sandbox=require_sandbox_snapshot,
    )
    composed = stage_snapshot is not None or layers != frozenset({"environment"})
    parent = stage_node if stage_snapshot is not None else rollout._cursor
    runner = run_child if run_child is not None else make_default_runner(rollout)
    run_dir = getattr(rollout, "_rollout_dir", None)

    # quiesce — pause the agent before snapshotting so the checkpoint is
    # consistent (the Branch lifecycle quiesces first); then checkpoint — the
    # roll-back point (a stage branch adopts the snapshot the boundary took).
    await rollout.disconnect()
    await checkpoint_parent(
        rollout,
        parent,
        stage_snapshot=stage_snapshot,
        composed=composed,
        snap_env=snap_env,
        snap_sandbox=snap_sandbox,
        layers=layers,
    )

    # The parent's linear state, captured once. Each child runs against a fresh
    # restore of this; the parent is restored to it at the end.
    saved = LinearState.capture(rollout)

    # The parent's *on-disk* state needs the same treatment, for the same
    # reason. Children share the parent's sandbox, and the sandbox bind-mounts
    # the parent rollout's agent/artifacts/verifier directories into the
    # container — so without this every child's verifier writes (and clears)
    # the parent's own evidence, and the run ends with the last child's
    # reward.txt sitting where the parent's belongs. Held aside before the
    # first child, handed to each child after it ran, put back at the end.
    # :mod:`benchflow.branch_artifacts` fails closed: a custody failure
    # preserves the hold directory and raises ``ArtifactCustodyError`` — the
    # fork must not report a world whose evidence it lost — except while a
    # child's own exception is propagating, where the failure is recorded and
    # logged instead so the child's failure stays the diagnosis.
    holder = (
        MountedArtifacts.hold(run_dir=run_dir, parent_id=parent.id)
        if run_dir is not None
        else None
    )
    transaction = BranchTransaction(
        rollout=rollout,
        n=n,
        deltas=deltas,
        parent=parent,
        saved=saved,
        runner=runner,
        run_child=run_child,
        composed=composed,
        snap_env=snap_env,
        snap_sandbox=snap_sandbox,
        fresh_children=fresh_children,
        run_dir=run_dir,
        holder=holder,
        # Resolved through *this* module's namespace at call time — the patch
        # seam tests/test_branch_child_result.py pins (a fake fresh-child
        # runner, a recording result writer).
        fresh_runner_factory=make_fresh_child_runner,
        write_child_result=write_in_place_child_result,
    )
    try:
        try:
            await transaction.run_children()
        except BaseException:
            # The exception on its way out — a child's crash, or a custody
            # failure the transaction itself surfaced — is the caller's
            # diagnosis. release() must not replace it: it records and logs a
            # failure of its own (preserving the hold directory) instead of
            # raising over the real one.
            if holder is not None:
                holder.release(raising=False)
            raise
        # Success path: a release failure here is the most important thing
        # that happened — it raises ArtifactCustodyError (the hold directory
        # is preserved), which the except below treats like any other fork
        # failure: restore the parent, persist the partial lineage, propagate.
        if holder is not None:
            holder.release()
    except BaseException:
        # A child raised something other than UnscoredChildError (an agent
        # connection failure, a verifier crash) and the fork ends here. The
        # caller may well survive it — ``run_ablation`` catches exactly this
        # and reports the completed, failed and skipped arms — so the evidence
        # for the arms that *did* run has to be on disk before the exception
        # leaves: without this, a partially completed experiment reported arms
        # whose tree.json and per-child provenance were never written.
        #
        # Same two steps as the success path, minus the aggregate: a fork that
        # did not finish has no V, and the failing child carries neither a
        # reward nor an unscored reason — which is what marks the tree partial.
        # The linear-state restore comes first so the parent's own state (and
        # its stage registry, re-serialized below) is the parent's again
        # before anything reads it.
        #
        # Nothing on this path may raise: the exception on its way out is the
        # caller's diagnosis, and neither best-effort step is worth replacing
        # it with. (On the success path a failed restore is loud, as it should
        # be — there is no more important failure to preserve there.)
        try:
            saved.restore_onto(rollout)
        except Exception:
            logger.warning(
                "branch could not restore the parent's linear state after a "
                "child failed — the parent's own result may carry the child's",
                exc_info=True,
            )
        _write_lineage(
            rollout, run_dir=run_dir, parent=parent, children=transaction.children
        )
        raise

    # restore the parent's linear state — the tree grew, nothing else moved.
    saved.restore_onto(rollout)
    value = _aggregate_fork(parent, transaction.children)
    rollout._phase = "branched"
    _write_lineage(
        rollout, run_dir=run_dir, parent=parent, children=transaction.children
    )
    return value


def _aggregate_fork(
    parent: RolloutNode, children: Sequence[RolloutNode]
) -> float | None:
    """Per-child return -> V(parent), over *this* fork's children only.

    A stage branch point can already carry the linear continuation, whose node
    has no reward to average. An unscored child makes V *undefined*: averaging
    it in as a zero would report a number computed from a score that was never
    observed, so the value is left unrecorded and returned as ``None``.

    "Unrecorded" has to mean *removed*, not merely "not written": a node can
    be branched more than once, and skipping the write would leave the
    previous fork's V sitting on the node — ``branch()`` returning None while
    ``tree.json`` publishes a number for a fork that has none.
    """
    unscored_children = [child for child in children if UNSCORED_KEY in child.state]
    if unscored_children:
        parent.state.pop("value", None)
        logger.error(
            "branch at %s left %d of %d children unscored — V(parent) is "
            "undefined and no value is recorded",
            parent.id,
            len(unscored_children),
            len(children),
        )
        return None
    value = _aggregate_branch(parent, over=children)
    parent.state["value"] = value
    return value


def _write_lineage(
    rollout: Rollout,
    *,
    run_dir: Any,
    parent: RolloutNode,
    children: Sequence[RolloutNode],
) -> None:
    """Write the fork's lineage artifacts (RFC §3.4), failure-isolated.

    Written only when the rollout knows its run directory, and never allowed
    to propagate: an artifact-write error is logged and neither corrupts the
    branch result nor — on the partial path, where this is called from an
    ``except`` block — masks the child failure that is on its way out. That
    isolation is the reason this is a plain ``except Exception`` rather than a
    chained re-raise: the exception being carried is the one the caller has to
    see, and it is the more important of the two.

    Called on both paths, so a fork that died mid-way leaves the same shape of
    evidence as one that finished: ``tree.json`` for the nodes that exist and
    per-child ``provenance.json`` / ``reward.json`` for the children that were
    attached, plus the parent's own stage registry (a child that ran through a
    stage boundary of its own already overwrote the file with its version).

    Kept in this module (not :mod:`benchflow.branch_report`) so the writers
    resolve through *this* namespace — ``tests/test_rollout_branch.py``
    patches ``benchflow.rollout_branch.write_branch_artifacts`` to prove the
    isolation.
    """
    if run_dir is None:
        return
    try:
        write_branch_artifacts(
            run_dir=run_dir,
            tree=rollout._tree,
            parent=parent,
            children=children,
        )
        stages = getattr(rollout, "_stage_snapshots", {})
        if stages:
            write_stage_snapshots(run_dir=run_dir, snapshots=stages)
    except Exception:
        logger.warning(
            "branch lineage artifact write under %s failed — the branch "
            "result is unaffected",
            run_dir,
            exc_info=True,
        )
