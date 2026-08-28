"""Stage and snapshot policy — what a branch may capture and fork, and when.

The fail-closed half of the branch engine (RFC §3.1–§3.3): every rule that can
be decided *before* anything is quiesced, checkpointed, or run lives here, so a
fork that cannot execute soundly dies with nothing paid for. Three families:

* **layers** — which checkpoint layers exist, that a requested set is
  non-empty (:func:`resolve_layers`), and that the live planes can actually
  take each requested layer (:func:`gate_layers`);
* **stages** — the one capture path for a stage boundary
  (:func:`capture_stage`, called by the lifecycle at the boundaries named in
  ``RolloutConfig.snapshot_stages``) and the resolution of a recorded stage
  into a branch's roll-back point (:func:`recorded_stage_checkpoint`);
* **deltas** — the whole-vector gate (:func:`validate_deltas`) mirroring what
  the delta will need at execution time: the boundary it can execute from, the
  layer that boundary's snapshot must carry, the parent precondition a
  ``skill_mode`` delta cannot see from the delta alone, and the
  caller-supplied-runner conflicts.

Execution lives elsewhere: :mod:`benchflow.branch_transaction` runs the fork
this module admitted, and :mod:`benchflow.branch_children` picks each child's
execution path.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from benchflow._utils.config_override import validate_overlay
from benchflow.branch import StageSnapshot
from benchflow.branch import checkpoint_composed as _checkpoint_composed
from benchflow.branch_children import (
    FRESH_CHILD_LAYER,
    FRESH_CHILD_STAGE,
    SKILL_DELTA_LAYER,
    SKILL_DELTA_STAGE,
    resolve_child_skill_policy,
    resolve_environment_ref_delta,
)
from benchflow.branch_delta import BranchDelta, BranchDeltaNotSupported
from benchflow.branch_lineage import write_stage_snapshots
from benchflow.branch_stage import (
    BranchStageNotCaptured,
    captured_stages,
    validate_stage,
)
from benchflow.skill_policy import SKILL_MODE_NO_SKILL

if TYPE_CHECKING:
    from benchflow.branch_children import ChildRunner
    from benchflow.rollout import Rollout
    from benchflow.trajectories.tree import RolloutNode

logger = logging.getLogger(__name__)

# The checkpoint layers a branch can compose (RFC §3.1). Agent-session is
# layer three and explicitly out of scope for v1.
_SNAPSHOT_LAYERS = frozenset({"environment", "sandbox"})

# The BranchDelta fields the engine executes vs records-only.
# ``injected_prompt`` runs in place on the parent's rollout; ``skill_mode``,
# ``config_override`` and ``environment_ref`` run the child as a fresh rollout
# from the env-ready snapshot (RFC §3.3, gated by
# :func:`_validate_skill_delta` / :func:`_validate_config_delta` /
# :func:`_validate_environment_delta`). Any remaining field exists in the
# schema and provenance so the artifact format is stable, and fails closed
# here. Derived from the schema (all BranchDelta fields minus the executable
# set) so a future BranchDelta field is unsupported-by-default — it fails
# closed here instead of being silently ignored. Sorted for deterministic
# errors; empty today, and load-bearing the day a field is added.
_EXECUTABLE_DELTA_FIELDS = frozenset(
    {"injected_prompt", "skill_mode", "config_override", "environment_ref"}
)
_UNSUPPORTED_DELTA_FIELDS = tuple(
    sorted({f.name for f in dataclasses.fields(BranchDelta)} - _EXECUTABLE_DELTA_FIELDS)
)


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


def resolve_layers(snapshot_layers: Iterable[str], *, subject: str) -> frozenset[str]:
    """Validate a layer set — the one gate every snapshot path uses.

    ``subject`` names the operation in the diagnostic ("branch", "stage
    snapshot 'env-ready'", "branch_at_stage('pre-verify'): the recorded stage
    snapshot").

    Called on *requested* layers by the cursor branch and by
    :func:`capture_stage`, and on the layers a stage branch *derived* from its
    recorded snapshot — the empty set is the same fail-open either way, so it
    is rejected in one place rather than trusted because it came from a
    checkpoint that was already taken.
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
            f"{subject} needs at least one layer — a fork whose checkpoint "
            "captures nothing has no roll-back point, so nothing is restored "
            "between children and every child runs in the world the previous "
            "one left"
        )
    return layers


def gate_layers(
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


def _validate_config_delta(
    overlay: dict[str, Any],
    *,
    index: int,
    at_stage: str | None,
    run_child: ChildRunner | None,
) -> None:
    """Gate a ``config_override`` delta before anything is quiesced (RFC §3.3).

    Three preconditions, each a way the delta would otherwise measure nothing
    (or worse, measure the wrong thing):

    * **the stage** — the C-axis overlay is deep-merged into the task's
      resolved config by ``setup()``, which only a fresh child rollout re-runs;
      by any later boundary the parent's config has already been consumed
      (its timeout enforced, its prompts resolved, its sandbox built), so a
      child there would record the override and run without it.
    * **the allowlist** — the same fail-closed allowlist the run-level overlay
      passes through (#790): scorer-touching sections (``verifier`` /
      ``reward`` / ``solution`` / ``oracle``) are rejected *here*, before the
      parent is quiesced, not at child setup after a snapshot was restored.
    * **the runner** — a caller-supplied ``run_child`` owns the child's
      execution, so the engine cannot run the child under the merged config.
    """
    where = f"stage {at_stage!r}" if at_stage is not None else "the cursor"
    if at_stage != FRESH_CHILD_STAGE:
        raise BranchDeltaNotSupported(
            f"deltas[{index}].config_override executes only at the "
            f"{FRESH_CHILD_STAGE!r} stage boundary, but this branch forks from "
            f"{where}: the overlay is deep-merged into the task's resolved "
            "config by the child's own setup(), which only a fresh child "
            "rollout (use_prebuilt_env) re-runs — by any later boundary the "
            "parent's config has already been consumed. Capture the boundary "
            "with RolloutConfig(snapshot_stages={'env-ready'}, "
            "snapshot_layers={'environment', 'sandbox'}) and fork with "
            "branch_at_stage('env-ready', ...)."
        )
    if run_child is not None:
        raise ValueError(
            f"deltas[{index}].config_override cannot be combined with an "
            "explicit run_child — the caller's runner owns the child's "
            "execution, so the engine cannot run setup() under the merged "
            "config. Drop run_child so the fresh-rollout runner executes the "
            "delta, or apply the override inside your own runner."
        )
    try:
        validate_overlay(overlay)
    except ValueError as exc:
        raise ValueError(f"deltas[{index}].config_override: {exc}") from None


def _validate_environment_delta(
    rollout: Rollout,
    environment_ref: str,
    *,
    index: int,
    at_stage: str | None,
    run_child: ChildRunner | None,
) -> None:
    """Gate an ``environment_ref`` delta before anything is quiesced (RFC §3.3).

    The stage and runner gates mirror the other fresh-child deltas: the child
    manifest binds at *provision* time, and provisioning happens only on the
    fresh-child path a ``env-ready`` fork runs (an in-place child at a later
    boundary would keep the parent's provisioned services no matter what its
    delta records). The delta's own content gates — a manifest-bound parent,
    a resolvable ref, the same image, framework-started services on both
    sides — live in :func:`~benchflow.branch_skill.resolve_environment_ref_delta`,
    which the child runner re-derives its manifest through, so validation and
    execution cannot drift.
    """
    where = f"stage {at_stage!r}" if at_stage is not None else "the cursor"
    if at_stage != FRESH_CHILD_STAGE:
        raise BranchDeltaNotSupported(
            f"deltas[{index}].environment_ref={environment_ref!r} executes "
            f"only at the {FRESH_CHILD_STAGE!r} stage boundary, but this "
            f"branch forks from {where}: the manifest's environment plane is "
            "provisioned over the restored sandbox by the fresh child rollout "
            "(use_prebuilt_env), and at any later boundary the parent's "
            "provisioned services survive the fork — the delta would be "
            "recorded but not enforced. Capture the boundary with "
            "RolloutConfig(snapshot_stages={'env-ready'}, "
            "snapshot_layers={'environment', 'sandbox'}) and fork with "
            "branch_at_stage('env-ready', ...)."
        )
    if run_child is not None:
        raise ValueError(
            f"deltas[{index}].environment_ref cannot be combined with an "
            "explicit run_child — the caller's runner owns the child's "
            "execution, so the engine cannot provision the child manifest's "
            "services. Drop run_child so the fresh-rollout runner executes "
            "the delta, or provision inside your own runner."
        )
    resolve_environment_ref_delta(
        rollout._config.environment_manifest,
        environment_ref,
        subject=f"deltas[{index}].environment_ref",
    )


def validate_deltas(
    rollout: Rollout,
    deltas: Sequence[BranchDelta | None] | None,
    *,
    n: int,
    at_stage: str | None,
    layers: frozenset[str],
    run_child: ChildRunner | None,
) -> None:
    """Gate the whole delta vector before anything runs (RFC §3.3).

    One entry per child, validated in order so a bad or not-yet-executable
    delta fails closed with nothing quiesced, checkpointed, or run — and with
    a diagnostic naming its index. A field outside the executable set fails
    unsupported-by-default; each executable field runs its own gate above; an
    ``injected_prompt`` cannot combine with a caller-supplied ``run_child``
    because that runner owns the child's prompts.
    """
    if deltas is None:
        return
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
        if delta.config_override is not None:
            _validate_config_delta(
                delta.config_override,
                index=index,
                at_stage=at_stage,
                run_child=run_child,
            )
        if delta.environment_ref is not None:
            _validate_environment_delta(
                rollout,
                delta.environment_ref,
                index=index,
                at_stage=at_stage,
                run_child=run_child,
            )
        if delta.skill_mode is not None:
            _validate_skill_delta(
                rollout,
                delta.skill_mode,
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


def runs_fresh_children(at_stage: str | None, run_child: ChildRunner | None) -> bool:
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


def validate_fresh_children(layers: frozenset[str], *, at_stage: str) -> None:
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
    layers = resolve_layers(snapshot_layers, subject=subject)
    snap_env, snap_sandbox = gate_layers(
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


def recorded_stage_checkpoint(
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

    Derived is not the same as trusted: a snapshot carrying *neither* ref
    derives the empty set, which the layer gates below then have nothing to
    check — the fork would restore nothing between children and still record a
    clean stage fork. So the derived set goes through :func:`resolve_layers`,
    the same non-empty gate the cursor branch runs on its requested set, and
    it runs before the disagreement check: an empty capture is broken however
    it is described, and ``snapshot_layers=set()`` would otherwise "agree"
    with it.
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
    layers = resolve_layers(
        (
            layer
            for layer, ref in (
                ("environment", snapshot.environment_ref),
                ("sandbox", snapshot.sandbox_ref),
            )
            if ref is not None
        ),
        subject=f"branch_at_stage({stage!r}): the recorded stage snapshot",
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
