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
argument — ``Rollout.branch`` / ``Rollout.mark_stage`` /
``Rollout.branch_at_stage`` are thin entry points that delegate here.
:func:`capture_stage` is the stage-boundary policy's one capture path (RFC
§3.2): the lifecycle calls it at the boundaries named in
``RolloutConfig.snapshot_stages``, and :func:`branch` with ``at_stage=`` forks
from what it recorded.

Most children run *in place* on the parent Rollout (a fresh agent session over
the restored world). Children forked from ``env-ready`` cannot: that boundary
precedes ``install_agent()``, so the restored world has no agent to connect to
and none of the skills, sandbox user or lockdown installation deploys — each
child re-runs installation as its own Rollout over the restored sandbox,
whatever its delta. This module keeps the gates
(:func:`_validate_skill_delta`, :func:`_validate_fresh_children`);
:mod:`benchflow.branch_skill` owns that child.

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

import dataclasses
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchflow.branch import UNSCORED_KEY, StageSnapshot, UnscoredChildError
from benchflow.branch import adopt_checkpoint as _adopt_checkpoint
from benchflow.branch import aggregate as _aggregate_branch
from benchflow.branch import checkpoint as _checkpoint_branch
from benchflow.branch import checkpoint_composed as _checkpoint_composed
from benchflow.branch import restore as _restore_branch
from benchflow.branch import restore_composed as _restore_composed
from benchflow.branch_artifacts import MountedArtifacts, child_mount_dir
from benchflow.branch_delta import BranchDelta
from benchflow.branch_lineage import write_branch_artifacts, write_stage_snapshots
from benchflow.branch_skill import (
    EXECUTION_FRESH_ROLLOUT,
    FRESH_CHILD_LAYER,
    FRESH_CHILD_STAGE,
    SKILL_DELTA_LAYER,
    SKILL_DELTA_STAGE,
    make_fresh_child_runner,
    resolve_child_skill_policy,
)
from benchflow.branch_stage import (
    BranchStageNotCaptured,
    captured_stages,
    validate_stage,
)
from benchflow.models import TrajectorySource
from benchflow.skill_policy import SKILL_MODE_NO_SKILL
from benchflow.trajectories.tree import RolloutNode

if TYPE_CHECKING:
    from benchflow.rollout import Rollout

logger = logging.getLogger(__name__)

# The checkpoint layers a branch can compose (RFC §3.1). Agent-session is
# layer three and explicitly out of scope for v1.
_SNAPSHOT_LAYERS = frozenset({"environment", "sandbox"})

#: Node-state key holding how long a branch child took, in seconds. Recorded
#: on the child node (even when the child raised) so a caller comparing
#: children — ``bench eval ablate``'s per-arm cost column — can attribute time
#: without re-deriving it from per-child artifacts, which only a
#: fresh-rollout child leaves behind. Deliberately *not* serialized by
#: :mod:`benchflow.branch_lineage`: a measured duration would make
#: ``tree.json`` non-deterministic.
CHILD_WALL_CLOCK_KEY = "wall_clock_sec"

# The BranchDelta fields the engine executes vs records-only.
# ``injected_prompt`` runs in place on the parent's rollout; ``skill_mode``
# runs the child as a fresh rollout from the env-ready snapshot (RFC §3.3,
# gated by :func:`_validate_skill_delta`). The rest exist in the schema and
# provenance so the artifact format is stable, and fail closed here.
# Derived from the schema (all BranchDelta fields minus the executable set)
# so a future BranchDelta field is unsupported-by-default — it fails closed
# here instead of being silently ignored. Sorted for deterministic errors.
_EXECUTABLE_DELTA_FIELDS = frozenset({"injected_prompt", "skill_mode"})
_UNSUPPORTED_DELTA_FIELDS = tuple(
    sorted(
        {f.name for f in dataclasses.fields(BranchDelta)} - _EXECUTABLE_DELTA_FIELDS
    )
)


class BranchDeltaNotSupported(NotImplementedError):
    """A BranchDelta a branch request cannot execute as asked (fail closed).

    Two cases. ``environment_ref`` and ``config_override`` are schema- and
    provenance-stable but have no execution path at all yet. ``skill_mode``
    does — as a fresh child rollout — but only from the ``env-ready`` stage
    snapshot, only when that snapshot carries the container layer, and only
    when the parent itself ran ``no-skill`` (see
    :class:`BranchParentSkillModeConflict`); a request that does not satisfy
    those preconditions fails closed here, before any child runs, rather than
    running a child that silently ignores its delta or measures a world its
    restore could not roll back.
    """


class BranchParentSkillModeConflict(BranchDeltaNotSupported):
    """A ``skill_mode`` delta forked from a parent carrying a baked-in pack.

    The precondition the other skill gates cannot see: the *parent's own*
    skill mode. A ``with-skill`` parent's ``setup()`` injects the pack into
    the Dockerfile it builds, so the pack is in the image — and the
    ``env-ready`` snapshot is a commit of that image. A ``no-skill`` child
    restoring it finds ``/skills`` already there, ``deploy_skills`` correctly
    deploys nothing on top, and the arm is reported as ``no-skill`` while the
    agent runs *with* the parent's skills. Nothing downstream can detect that:
    the child's config, provenance and ablation row all say ``no-skill``.

    So the delta is refused unless the snapshot's world provably carries no
    pack — i.e. the parent's own recorded skill mode is ``no-skill``, the one
    mode whose skill policy resolves no host directory and therefore never
    reaches ``inject_skills_into_dockerfile``. Each arm then deploys its own
    skills (or none) at install time, over a pack-free image.
    """


class BranchChildExecutionNotSupported(NotImplementedError):
    """The engine cannot run this fork's children safely (fail closed).

    Raised when the *boundary*, not the delta, makes execution unsound: an
    ``env-ready`` fork whose snapshot omits the container layer. Every child of
    that stage re-runs ``install_agent()`` for itself (see
    :mod:`benchflow.branch_skill`), and without the container layer the restore
    cannot undo the previous child's installation — the second child would
    install on top of the first and report the result as a controlled
    comparison.
    """

# The per-child runner: given the child's branch node, run its continuation and
# return the scalar return. No ``int`` index — a caller that needs per-child
# prompts binds them into a closure (see ``run_child`` in :func:`branch`).
ChildRunner = Callable[[RolloutNode], Awaitable[float]]


@dataclass
class _LinearState:
    """A scoped snapshot of a Rollout's linear (non-tree) execution state.

    Captured before a branch child runs and restored after — this is what
    makes a branch child an *isolated sub-rollout* rather than a re-entrant
    mutation of the shared Rollout instance. The stage registry (RFC §3.2) is
    scoped the same way: a child that runs through its own ``pre-verify``
    boundary records that snapshot on its own node, and it must not become the
    *parent's* pre-verify — a later branch there would fork the child's world.
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
            executed_prompts=list(rollout._executed_prompts),
            stage_snapshots=dict(getattr(rollout, "_stage_snapshots", {})),
            stage_nodes=dict(getattr(rollout, "_stage_nodes", {})),
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
        rollout._executed_prompts = list(self.executed_prompts)
        rollout._stage_snapshots = dict(self.stage_snapshots)
        rollout._stage_nodes = dict(self.stage_nodes)


def _resolve_layers(snapshot_layers: Iterable[str], *, subject: str) -> frozenset[str]:
    """Validate a requested layer set — the one gate both snapshot paths use.

    ``subject`` names the operation in the diagnostic ("branch", "stage
    snapshot 'env-ready'").
    """
    layers = frozenset(snapshot_layers)
    unknown = layers - _SNAPSHOT_LAYERS
    if unknown:
        raise ValueError(
            f"unknown snapshot_layers {sorted(unknown)!r} — allowed layers "
            f"are {sorted(_SNAPSHOT_LAYERS)!r}"
        )
    if not layers:
        raise ValueError(
            "snapshot_layers must request at least one layer — a "
            f"{subject} without a checkpoint has no roll-back point"
        )
    return layers


def _gate_layers(
    rollout: Rollout,
    layers: frozenset[str],
    *,
    subject: str,
    requested: str,
    gate_sandbox: bool = False,
) -> tuple[Any, Any]:
    """Fail closed when a requested layer's plane or capability is missing.

    Returns the ``(environment, sandbox)`` live objects for the requested
    layers — ``None`` for a layer that was not requested, which is exactly the
    "layer not requested" signal :func:`checkpoint_composed` /
    :func:`restore_composed` take. Every diagnostic names ``subject``, so a
    stage capture says *which stage* could not be taken (RFC §3.2) instead of
    the generic branch message.

    The container layer keeps its #384 semantics: the Branch lifecycle
    composes container ⊃ environment-state ⊃ agent-session, and restoring only
    environment state can produce inconsistent state for runs that mutate
    process/service state the Environment manifest does not capture — so a
    missing ``supports_snapshot`` never silently degrades.
    """
    if "environment" in layers and getattr(rollout, "_environment", None) is None:
        raise RuntimeError(
            f"{subject} needs the Environment plane — there is no world to "
            "snapshot. Pass RolloutConfig(environment_manifest=...)."
        )
    sandbox = getattr(rollout, "_env", None)
    if (gate_sandbox or "sandbox" in layers) and not getattr(
        sandbox, "supports_snapshot", False
    ):
        sandbox_name = type(sandbox).__name__ if sandbox else "<none>"
        raise RuntimeError(
            f"{subject} cannot run with {requested}: the active "
            f"sandbox {sandbox_name!r} does not implement container-level "
            "snapshot/restore. Use a provider whose Sandbox satisfies the "
            "checkpoint contract (DockerSandbox or DaytonaSandbox in direct "
            "mode), or drop the sandbox layer if Environment-state "
            "checkpoint is sufficient for this run."
        )
    return (
        rollout._environment if "environment" in layers else None,
        sandbox if "sandbox" in layers else None,
    )


def _validate_skill_delta(
    rollout: Rollout,
    skill_mode: str,
    *,
    index: int,
    at_stage: str | None,
    layers: frozenset[str],
    run_child: ChildRunner | None,
) -> None:
    """Gate a ``skill_mode`` delta before anything is quiesced (RFC §3.3).

    Four preconditions, each a way the delta would otherwise measure nothing:

    * **the stage** — skills are deployed by ``install_agent()``, so only the
      ``env-ready`` boundary (captured before it) can vary them; a cursor
      branch forks a world that already resolved the question.
    * **the container layer** — skills live in the container filesystem, so an
      environment-state-only checkpoint cannot roll the parent's pack back and
      a ``no-skill`` child would still find it mounted.
    * **the parent's own skill mode** — the layer above rolls the container
      back *to the parent's env-ready image*, and a ``with-skill`` parent
      baked the pack into that image at build time. Rolling back to it hands
      a ``no-skill`` child the pack it is supposed to be running without, and
      every artifact still labels the arm ``no-skill``. Only a parent whose
      recorded mode is ``no-skill`` provably carries no pack, so anything else
      raises :class:`BranchParentSkillModeConflict`.
    * **the runner** — a caller-supplied ``run_child`` owns the child's
      execution, so it, not the engine, would decide what the child installs.

    The task's own skill policy is resolved here too, so ``with-skill``
    against a task shipping no bundled pack raises the same typed error
    ``setup()`` would raise, before the branch restored anything.
    """
    where = f"stage {at_stage!r}" if at_stage is not None else "the cursor"
    if at_stage != SKILL_DELTA_STAGE:
        raise BranchDeltaNotSupported(
            f"deltas[{index}].skill_mode={skill_mode!r} executes only at "
            f"the {SKILL_DELTA_STAGE!r} stage boundary, but this branch forks "
            f"from {where}: skills are deployed by install_agent(), which has "
            "already run by then. Capture the boundary with "
            "RolloutConfig(snapshot_stages={'env-ready'}, "
            "snapshot_layers={'environment', 'sandbox'}) and fork with "
            "branch_at_stage('env-ready', ...) — the child then runs as a "
            "fresh rollout over the restored sandbox (use_prebuilt_env)."
        )
    if SKILL_DELTA_LAYER not in layers:
        raise BranchDeltaNotSupported(
            f"deltas[{index}].skill_mode={skill_mode!r} needs the "
            f"{SKILL_DELTA_LAYER!r} layer in the {SKILL_DELTA_STAGE!r} "
            f"snapshot, which captured layers={sorted(layers)!r}: skills are "
            "deployed into the container filesystem, and an "
            "environment-state-only checkpoint cannot roll them back — the "
            "child would re-install against the parent's skills. Re-capture "
            "the stage with snapshot_layers={'environment', 'sandbox'} (or "
            "{'sandbox'} alone for a stateless environment)."
        )
    parent_mode = rollout._config.recorded_skill_mode
    if parent_mode != SKILL_MODE_NO_SKILL:
        raise BranchParentSkillModeConflict(
            f"deltas[{index}].skill_mode={skill_mode!r} cannot fork a parent "
            f"whose own skill mode is {parent_mode!r}: only a "
            f"{SKILL_MODE_NO_SKILL!r} parent provably bakes no skill pack "
            f"into the image its setup() builds, and the "
            f"{SKILL_DELTA_STAGE!r} snapshot every child restores is a commit "
            "of that image. A 'no-skill' child of a parent that did bake one "
            "restores the pack, deploys nothing on top of it, runs *with* the "
            "parent's skills, and is still recorded as 'no-skill' in its "
            "config, its provenance and its ablation row. Run the parent with "
            f"skill_mode={SKILL_MODE_NO_SKILL!r} and let each arm's delta "
            "deploy its own skills at install time."
        )
    if run_child is not None:
        raise ValueError(
            f"deltas[{index}].skill_mode cannot be combined with an explicit "
            "run_child — the caller's runner owns the child's execution, so "
            "the engine cannot re-run install_agent() under the switched "
            "mode. Drop run_child so the fresh-rollout runner executes the "
            "delta, or switch skills inside your own runner."
        )
    resolve_child_skill_policy(rollout._config, skill_mode)


def _runs_fresh_children(at_stage: str | None, run_child: ChildRunner | None) -> bool:
    """Whether the engine will run this fork's children as fresh rollouts.

    True for every engine-run child of ``env-ready``, whatever its delta. That
    boundary precedes ``install_agent()``, so the restored world has no agent
    binary, no sandbox user, no seeded verifier workspace, no path lockdown and
    no skill pack: an in-place child there connects to something the restore
    deleted, or — when the agent happens to survive in the base image — scores
    a world missing everything installation deploys and reports it as an
    ordinary child under one recorded delta. A caller-supplied ``run_child``
    owns its own execution and is left alone.
    """
    return run_child is None and at_stage == FRESH_CHILD_STAGE


def _validate_fresh_children(layers: frozenset[str], *, at_stage: str) -> None:
    """Gate a fresh-child fork before anything is quiesced.

    One precondition beyond the per-delta gates: the stage snapshot must carry
    the container layer. Every child re-runs ``install_agent()``, which writes
    to the container filesystem, so an environment-state-only checkpoint cannot
    put the world back between children — child *k+1* would install on top of
    child *k*'s agent, user, lockdown and skill pack. The skill-delta gate says
    the same thing in skill terms and fires first when a skill delta is
    present; this covers the children that carry no skill delta at all.
    """
    if FRESH_CHILD_LAYER not in layers:
        raise BranchChildExecutionNotSupported(
            f"a branch at the {at_stage!r} stage boundary runs every child as "
            f"a fresh rollout, which needs the {FRESH_CHILD_LAYER!r} layer in "
            f"that snapshot; it captured layers={sorted(layers)!r}. "
            f"{at_stage!r} precedes install_agent(), so each child installs "
            "the agent (and its skills, user and lockdown) for itself, and an "
            "environment-state-only checkpoint cannot roll that back between "
            "children. Re-capture the stage with "
            "snapshot_layers={'environment', 'sandbox'} (or {'sandbox'} alone "
            "for a stateless environment), or supply your own run_child if "
            "you are executing the children yourself."
        )


async def capture_stage(
    rollout: Rollout,
    stage: str,
    *,
    snapshot_layers: Iterable[str] = frozenset({"environment"}),
) -> StageSnapshot:
    """Take the composed stage snapshot at ``rollout``'s cursor (RFC §3.2).

    The stage-boundary policy's one capture path: validate the stage name and
    the requested layers, fail closed on a missing plane or capability (the
    diagnostic names the stage), compose the checkpoint on the cursor node —
    so the stage tag flows into ``tree.json`` through the node's own recorded
    state — and register it under ``stage`` on the rollout. The registry is
    re-serialized to ``stage_snapshots.json`` after every capture, with write
    failures logged and isolated from the rollout.

    Callers are the lifecycle's own boundaries (``start()``/``verify()``, gated
    on ``RolloutConfig.snapshot_stages``) and ``Rollout.mark_stage`` for the
    boundaries only the caller can see. Two stages captured without an
    intervening Step land on the same node (``pre-verify`` and ``post-verify``
    of a rollout that verifies once, say): the registry keeps both, and the
    node's own tag is the most recent — a node has one checkpoint, and the
    branch that adopts it re-tags it with the stage it branched from.
    """
    validate_stage(stage)
    subject = f"stage snapshot {stage!r}"
    layers = _resolve_layers(snapshot_layers, subject=subject)
    snap_env, snap_sandbox = _gate_layers(
        rollout,
        layers,
        subject=subject,
        requested=f"snapshot_layers={sorted(layers)!r}",
    )
    try:
        snapshot = await _checkpoint_composed(
            rollout._cursor,
            environment=snap_env,
            sandbox=snap_sandbox,
            stage=stage,
        )
    except Exception as exc:
        exc.add_note(
            f"stage snapshot {stage!r} (snapshot_layers={sorted(layers)!r}) "
            "failed during checkpoint — the rollout fails closed; no partial "
            "checkpoint was recorded on the node."
        )
        raise
    rollout._stage_snapshots[stage] = snapshot
    rollout._stage_nodes[stage] = rollout._cursor

    run_dir = getattr(rollout, "_rollout_dir", None)
    if run_dir is not None:
        try:
            write_stage_snapshots(run_dir=run_dir, snapshots=rollout._stage_snapshots)
        except Exception:
            logger.warning(
                "stage snapshot artifact write under %s failed — the recorded "
                "stage registry is unaffected",
                run_dir,
                exc_info=True,
            )
    return snapshot


def _recorded_stage_checkpoint(
    rollout: Rollout, stage: str, snapshot_layers: Iterable[str] | None
) -> tuple[StageSnapshot, RolloutNode, frozenset[str]]:
    """Resolve a recorded stage into ``(snapshot, branch node, layers)``.

    A stage that was never captured fails closed with
    :class:`~benchflow.branch_stage.BranchStageNotCaptured` naming what *was*
    captured — degrading to a checkpoint at the cursor would fork a different
    world state than the caller asked for and mislabel it in provenance. The
    layers are derived from the recorded snapshot; an explicit
    ``snapshot_layers`` that disagrees is a caller bug, not a request to
    re-snapshot.
    """
    validate_stage(stage)
    registry: dict[str, StageSnapshot] = getattr(rollout, "_stage_snapshots", {})
    snapshot = registry.get(stage)
    if snapshot is None:
        raise BranchStageNotCaptured(
            f"no snapshot recorded at stage {stage!r} — this rollout captured "
            f"{captured_stages(registry)!r}. Request the stage up front with "
            "RolloutConfig(snapshot_stages={...}), or record it at the cut "
            "point with Rollout.mark_stage()."
        )
    layers = frozenset(
        layer
        for layer, ref in (
            ("environment", snapshot.environment_ref),
            ("sandbox", snapshot.sandbox_ref),
        )
        if ref is not None
    )
    if snapshot_layers is not None and frozenset(snapshot_layers) != layers:
        raise ValueError(
            f"branch_at_stage({stage!r}, snapshot_layers="
            f"{sorted(frozenset(snapshot_layers))!r}) disagrees with the "
            f"layers stage {stage!r} actually captured ({sorted(layers)!r}) — "
            "a stage branch restores exactly what the stage snapshotted; "
            "re-run with the layers the capture used, or capture the stage "
            "with the layers you want."
        )
    node = getattr(rollout, "_stage_nodes", {}).get(stage, rollout._cursor)
    return snapshot, node, layers


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
    per child, validated before anything runs. Two fields execute.
    ``injected_prompt`` becomes the child's continuation prompt, delivered as
    the child's user-visible first message and recorded in provenance as a
    content hash. ``skill_mode`` re-runs ``install_agent()`` under the switched
    mode — the with-skill / no-skill ablation; it requires
    ``at_stage="env-ready"`` with the container layer in that snapshot **and a
    parent whose own recorded skill mode is ``no-skill``** (a with-skill
    parent baked its pack into the image the snapshot commits, so a no-skill
    child would restore it and be mislabeled), and fails closed otherwise.
    The remaining fields (``environment_ref``,
    ``config_override``) raise :class:`BranchDeltaNotSupported` — fail closed
    before any child runs — until the RFC follow-on executes them.

    **How a child is executed depends on the boundary, not on the delta.**
    ``at_stage="env-ready"`` precedes ``install_agent()``, so every child the
    engine runs from there is a *fresh rollout* over the restored sandbox
    (``use_prebuilt_env``) that installs the agent for itself — including a
    child with no delta at all, which re-installs the parent's own recorded
    skill mode. An in-place child there would connect to an agent the restore
    deleted, or score a world missing everything installation deploys and
    report it as an ordinary single-delta child; that fork therefore requires
    the container layer and raises
    :class:`BranchChildExecutionNotSupported` without it. At every other
    boundary the agent is already installed and children run in place.

    After this returns, ``rollout``'s linear state is exactly what it was
    before — the tree gained ``n`` children at the cursor, nothing else
    moved; when the rollout knows its run directory, the branch also leaves
    lineage artifacts (``tree.json``,
    ``branches/<branch-node-id>/children/<child-node-id>/``), with any
    artifact-write failure logged and isolated from the branch result.

    ``at_stage`` branches from a recorded stage boundary instead of
    checkpointing at the cursor (RFC §3.2): the stage's :class:`StageSnapshot`
    becomes the roll-back point every child restores to, the fork happens at
    the node that stage was captured on (so the children continue that stage's
    world, and V aggregates over this fork only), and the stage name — not the
    ``cursor:<id>`` fallback — is the ``branch_stage`` recorded in provenance.
    """
    stage_snapshot: StageSnapshot | None = None
    if at_stage is not None:
        stage_snapshot, stage_node, layers = _recorded_stage_checkpoint(
            rollout, at_stage, snapshot_layers
        )
        subject = f"branch_at_stage({at_stage!r})"
    else:
        subject = "branch"
        layers = _resolve_layers(
            frozenset({"environment"}) if snapshot_layers is None else snapshot_layers,
            subject=subject,
        )
    if n < 2:
        raise ValueError(f"a branch forks into >= 2 children, got n={n}")

    # Validate the whole delta vector before anything runs — a bad or
    # not-yet-executable delta fails closed with nothing quiesced,
    # checkpointed, or run.
    if deltas is not None:
        if len(deltas) != n:
            raise ValueError(
                f"deltas must carry exactly one entry per child: got "
                f"{len(deltas)} deltas for n={n}"
            )
        for index, delta in enumerate(deltas):
            if delta is None:
                continue
            for field_name in _UNSUPPORTED_DELTA_FIELDS:
                if getattr(delta, field_name) is not None:
                    raise BranchDeltaNotSupported(
                        f"deltas[{index}].{field_name} is set, but the branch "
                        "engine executes injected_prompt and skill_mode only "
                        f"— {field_name!r} runs in the rollout-branching RFC "
                        "follow-on (child-as-fresh-rollout via "
                        "use_prebuilt_env). The field is already recorded in "
                        "the schema and provenance; drop it to branch today."
                    )
            skill_mode = delta.skill_mode
            if skill_mode is not None:
                _validate_skill_delta(
                    rollout,
                    skill_mode,
                    index=index,
                    at_stage=at_stage,
                    layers=layers,
                    run_child=run_child,
                )
            if delta.injected_prompt is not None and run_child is not None:
                raise ValueError(
                    f"deltas[{index}].injected_prompt cannot be combined "
                    "with an explicit run_child — the caller's runner owns "
                    "the child's prompts. Bind the prompt into the "
                    "run_child closure, or drop run_child so the default "
                    "runner delivers it."
                )

    # The boundary's own gate: at ``env-ready`` every engine-run child is a
    # fresh rollout (whatever its delta, including no delta), so the container
    # layer is required even for a fork carrying no deltas at all.
    fresh_children = _runs_fresh_children(at_stage, run_child)
    if fresh_children:
        _validate_fresh_children(layers, at_stage=str(at_stage))

    # Fail closed when a requested layer's plane or capability is missing
    # (#384). ``require_sandbox_snapshot`` keeps its original check-only
    # semantics; requesting the layer via ``snapshot_layers`` gates the same
    # way and then actually composes the sandbox into the checkpoint. A stage
    # branch gates too — the planes must still be live to restore what the
    # stage captured.
    snap_env, snap_sandbox = _gate_layers(
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
    # consistent (the Branch lifecycle quiesces first).
    await rollout.disconnect()

    # checkpoint — snapshot the requested layers at the parent; the roll-back
    # point. The default environment-only request keeps the legacy path (and
    # the legacy bare-StateSnapshot shape on node.state["snapshot"]) intact.
    # A stage branch adopts the snapshot the stage boundary already took —
    # re-snapshotting there would capture a world that has since moved on.
    if stage_snapshot is not None:
        _adopt_checkpoint(parent, stage_snapshot)
    else:
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

    # The parent's linear state, captured once. Each child runs against a fresh
    # restore of this; the parent is restored to it at the end.
    saved = _LinearState.capture(rollout)

    # The parent's *on-disk* state needs the same treatment, for the same
    # reason. Children share the parent's sandbox, and the sandbox bind-mounts
    # the parent rollout's agent/artifacts/verifier directories into the
    # container — so without this every child's verifier writes (and clears)
    # the parent's own evidence, and the run ends with the last child's
    # reward.txt sitting where the parent's belongs. Held aside before the
    # first child, handed to each child after it ran, put back in the finally
    # below. :mod:`benchflow.branch_artifacts` isolates its own failures: a
    # lost copy of the evidence must not cost the rewards.
    holder = (
        MountedArtifacts.hold(run_dir=run_dir, parent_id=parent.id)
        if run_dir is not None
        else None
    )

    children: list[RolloutNode] = []
    try:
        await _run_children(
            rollout,
            n=n,
            deltas=deltas,
            parent=parent,
            children=children,
            saved=saved,
            runner=runner,
            run_child=run_child,
            composed=composed,
            snap_env=snap_env,
            snap_sandbox=snap_sandbox,
            fresh_children=fresh_children,
            run_dir=run_dir,
            holder=holder,
        )
    finally:
        if holder is not None:
            holder.release()

    # restore the parent's linear state — the tree grew, nothing else moved.
    saved.restore_onto(rollout)

    # aggregate — per-child return -> V(parent), over *this* fork's children:
    # a stage branch point can already carry the linear continuation, whose
    # node has no reward to average. An unscored child makes V *undefined*:
    # averaging it in as a zero would report a number computed from a score
    # that was never observed, so the value is left unrecorded and returned
    # as None.
    unscored_children = [child for child in children if UNSCORED_KEY in child.state]
    value: float | None
    if unscored_children:
        value = None
        logger.error(
            "branch at %s left %d of %d children unscored — V(parent) is "
            "undefined and no value is recorded",
            parent.id,
            len(unscored_children),
            len(children),
        )
    else:
        value = _aggregate_branch(parent, over=children)
        parent.state["value"] = value
    rollout._phase = "branched"

    # Lineage artifacts (RFC §3.4) — written only when the rollout knows its
    # run directory, and failure-isolated: an artifact-write error is logged
    # and never corrupts the branch result.
    if run_dir is not None:
        try:
            write_branch_artifacts(
                run_dir=run_dir,
                tree=rollout._tree,
                parent=parent,
                children=children,
            )
            # Re-serialize the (restored) parent registry: a child that ran
            # through a stage boundary of its own already overwrote the file.
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
    return value


async def _run_children(
    rollout: Rollout,
    *,
    n: int,
    deltas: Sequence[BranchDelta | None] | None,
    parent: RolloutNode,
    children: list[RolloutNode],
    saved: _LinearState,
    runner: ChildRunner,
    run_child: ChildRunner | None,
    composed: bool,
    snap_env: Any,
    snap_sandbox: Any,
    fresh_children: bool,
    run_dir: Any,
    holder: MountedArtifacts | None,
) -> None:
    """Run the fork's ``n`` children in order, appending them to ``children``.

    Split out of :func:`branch` only so the mounted-artifact custody around the
    loop reads as one bracket instead of burying the aggregation step inside a
    ``try``. Every child is appended to ``children`` as soon as it is attached,
    so a caller that catches a propagating child failure still sees the nodes
    that were forked — which is how ``bench eval ablate`` reports the arm that
    raised and the arms that never ran.
    """
    for index in range(n):
        # Attach a *pending* branch-child node — its real continuation Step is
        # filled in place by the child's first execute(), so the child's work
        # lands on the child node, not a descendant placeholder.
        delta = deltas[index] if deltas is not None else None
        child = rollout._tree.attach(parent)
        # The child's delta provenance is recorded on the node itself at fork
        # time (``None`` = the zero delta), so lineage serialization reads it
        # from the node — never by positional alignment — and a second
        # branch() at the same parent can never misattribute deltas.
        child.state["delta"] = (
            delta if delta is not None else BranchDelta()
        ).provenance_dict()
        # How the delta is executed, recorded on the node for the same reason:
        # an env-ready child re-runs installation as its own Rollout, and the
        # artifacts must say so rather than leaving a reader to infer it from
        # the delta — which they could not, since a zero-delta child of that
        # boundary is a fresh rollout too.
        if fresh_children:
            child.state["delta_execution"] = EXECUTION_FRESH_ROLLOUT
        children.append(child)

        # restore the checkpointed layers (sandbox first, then env — the
        # reverse of checkpoint order), reset the parent's linear state, and
        # point the cursor at the pending child for the sub-rollout.
        if composed:
            await _restore_composed(parent, environment=snap_env, sandbox=snap_sandbox)
        else:
            await _restore_branch(parent, rollout._environment)
        saved.restore_onto(rollout)
        rollout._cursor = child

        # Pick the per-child runner (every case validated above to not combine
        # with an explicit run_child). A child forked from ``env-ready`` runs
        # as a fresh Rollout over the just-restored sandbox — that boundary
        # precedes install_agent(), so there is no agent in the restored world
        # to connect to and no installed skills to keep; the child re-runs
        # installation for itself, under the delta's skill_mode when it has
        # one and under the parent's own recorded mode when it does not.
        # Everywhere else the agent is already installed and the child runs in
        # place, with an injected-prompt delta binding the child's continuation
        # prompt into a per-child default runner — the formalized version of
        # the caller's per-child-prompt closure.
        child_runner = runner
        if fresh_children:
            child_runner = make_fresh_child_runner(
                rollout,
                delta=delta,
                parent=parent,
                branch_stage=FRESH_CHILD_STAGE,
                run_dir=run_dir,
            )
        elif (
            run_child is None and delta is not None and delta.injected_prompt is not None
        ):
            child_runner = make_default_runner(rollout, prompts=[delta.injected_prompt])

        started = time.monotonic()
        unscored: str | None = None
        try:
            ret = await child_runner(child)
        except UnscoredChildError as exc:
            # The child ran but was never scored. Record *why*, leave the
            # node's reward unset, and keep going: the next child restores the
            # checkpoint for itself, so one lost score does not cost the fork.
            unscored = exc.reason
        finally:
            # Recorded in a finally so a child that raised still reports what
            # it cost before it did — and, for the same reason, a child that
            # raised still hands off whatever it wrote to the shared mounts
            # (a crashed child's verifier output is often the only evidence of
            # why it crashed). Handing off also empties the mounts, so the
            # next child cannot inherit this one's files.
            child.state[CHILD_WALL_CLOCK_KEY] = time.monotonic() - started
            if holder is not None and run_dir is not None:
                holder.hand_off(child_mount_dir(run_dir, parent.id, child.id))
        if unscored is not None:
            child.state[UNSCORED_KEY] = unscored
            logger.error("branch child %s is unscored: %s", child.id, unscored)
        else:
            child.state["reward"] = float(ret)


def make_default_runner(
    rollout: Rollout, *, prompts: list[str] | None = None
) -> ChildRunner:
    """Build the default per-child runner bound to ``rollout``.

    The default runner re-runs the child from the parent's env checkpoint with
    a *fresh agent session* — agent-session snapshot is the unsolved hard part
    (``docs/architecture.md``, "The hard part"), so the agent restarts per
    child. Each child connects a fresh agent and disconnects it at the end, so
    no two children's agents overlap (the next child connects only after the
    previous one disconnected). ``verify()`` returning ``None``, an empty dict,
    or a dict with a ``None`` reward raises
    :class:`~benchflow.branch.UnscoredChildError`: the child ran but was never
    scored, and a fabricated ``0.0`` would read as a real failing score.

    ``prompts`` — the child's continuation prompts; ``None`` keeps the
    rollout's resolved prompts. An ``injected_prompt`` delta (RFC §3.3) binds
    here, so the injection is the child's user-visible first message — never
    silently merged into other prompt content (#908).
    """

    async def _runner(child: RolloutNode) -> float:
        await rollout.connect()
        # Fill the pending branch-child node in place — the continuation Step
        # lands on `child` itself, no content-free placeholder.
        await rollout.execute(prompts, node=child)
        rewards = await rollout.verify()
        await rollout.disconnect()
        if not rewards or rewards.get("reward") is None:
            raise UnscoredChildError(
                f"branch child {child.id} produced no verifier reward "
                f"(verify() returned {rewards!r})" + _verifier_error_suffix(rollout)
            )
        return float(rewards["reward"])

    return _runner


def _verifier_error_suffix(rollout: Rollout) -> str:
    """`` — <verifier error>`` when the rollout recorded one, else ``''``.

    The verifier's own diagnostic is the difference between "the agent scored
    nothing" and "the score never reached the host", so it is carried into the
    unscored reason verbatim rather than left in the log.
    """
    error = getattr(rollout, "_verifier_error", None)
    return f" — {error}" if error else ""
