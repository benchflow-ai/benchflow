"""Branch lineage artifacts — ``tree.json``, per-child provenance, stages.

Today ``RolloutTree`` lives and dies in memory; branching must leave the same
quality of evidence as a linear run. This module serializes the tree to a
deterministic ``tree.json`` in the run folder, writes the stage-snapshot
registry (RFC §3.2) as ``stage_snapshots.json``, and builds the
``kind="benchflow-branch"`` source-provenance dict each branch child carries —
the same seam ``benchflow-continue`` uses.

Pure writers only: no engine state, no wall-clock timestamps (determinism is a
test guarantee — goldens pin the output byte-for-byte). Failure isolation is
the caller's job: the engine wraps these writes so an artifact error never
corrupts branch results.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchflow.branch import StageSnapshot
from benchflow.branch_delta import BranchDelta
from benchflow.environment.protocol import StateSnapshot
from benchflow.trajectories.tree import RolloutNode, RolloutTree

_SCHEMA_VERSION = 1
_SNAPSHOT_KEY = "snapshot"
_REWARD_KEY = "reward"
_VALUE_KEY = "value"
_DELTA_KEY = "delta"
_EXECUTION_KEY = "delta_execution"


def _snapshot_refs(snap: Any) -> dict[str, str | None] | None:
    """Serializable per-layer refs of a recorded checkpoint (either shape).

    A legacy bare :class:`StateSnapshot` is an environment-only checkpoint; a
    :class:`StageSnapshot` carries one ref per requested layer. ``None`` in
    and anything unrecognized out map to ``None`` — the serializer records,
    it does not validate.
    """
    if isinstance(snap, StateSnapshot):
        return {"environment": snap.id, "sandbox": None}
    if isinstance(snap, StageSnapshot):
        return {
            "environment": (
                snap.environment_ref.id if snap.environment_ref is not None else None
            ),
            "sandbox": snap.sandbox_ref.ref if snap.sandbox_ref is not None else None,
        }
    return None


def _provenance(delta: BranchDelta | dict[str, Any] | None) -> dict[str, Any]:
    """A delta's provenance dict; ``None`` records as the zero delta.

    Accepts an already-serialized provenance dict verbatim — the branch
    engine records each child's delta provenance on the node at fork time
    (``node.state["delta"]``), and the writers here pass it through.
    """
    if delta is None:
        return BranchDelta().provenance_dict()
    if isinstance(delta, BranchDelta):
        return delta.provenance_dict()
    return delta


def serialize_tree(
    tree: RolloutTree,
    *,
    run_dir: Path,
    cut_point: dict[str, Any] | None = None,
) -> Path:
    """Write ``<run_dir>/tree.json`` — the deterministic lineage artifact.

    Every node serializes from its *own* recorded state: id / parent / stage
    tag (the recorded checkpoint's ``stage``) / snapshot refs, plus
    ``reward``, ``value``, ``delta`` (the provenance dict the engine attached
    at fork time) and ``delta_execution`` (how the engine ran the child —
    absent for the ordinary in-place child) when present on the node. Nothing
    is inferred from position — a tree with several branch points, or several branch
    events at the same parent, serializes each node's provenance exactly as
    recorded. ``cut_point`` is recorded at the top level. Output is
    deterministic — sorted keys, indented, trailing newline, no wall-clock
    timestamps.
    """
    nodes_payload: list[dict[str, Any]] = []
    for node in tree.nodes():
        snap = node.state.get(_SNAPSHOT_KEY)
        entry: dict[str, Any] = {
            "id": node.id,
            "parent": node.parent.id if node.parent is not None else None,
            "stage": getattr(snap, "stage", None),
            "snapshot": _snapshot_refs(snap),
        }
        if _REWARD_KEY in node.state:
            entry["reward"] = float(node.state[_REWARD_KEY])
        if _VALUE_KEY in node.state:
            entry["value"] = float(node.state[_VALUE_KEY])
        if _DELTA_KEY in node.state:
            entry["delta"] = node.state[_DELTA_KEY]
        if _EXECUTION_KEY in node.state:
            entry[_EXECUTION_KEY] = node.state[_EXECUTION_KEY]
        nodes_payload.append(entry)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "cut_point": cut_point,
        "nodes": nodes_payload,
    }
    path = Path(run_dir) / "tree.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path


def stage_snapshots_payload(
    snapshots: Mapping[str, StageSnapshot],
) -> dict[str, dict[str, Any]]:
    """The stage registry as a serializable ``stage -> refs`` mapping (RFC §3.2).

    Each entry records the per-layer roll-back handles and the ``layers`` the
    stage actually captured (sorted) — the set a stage branch restores.
    """
    payload: dict[str, dict[str, Any]] = {}
    for stage, snap in snapshots.items():
        refs = _snapshot_refs(snap) or {"environment": None, "sandbox": None}
        payload[stage] = {
            "environment_ref": refs["environment"],
            "sandbox_ref": refs["sandbox"],
            "layers": sorted(layer for layer, ref in refs.items() if ref is not None),
        }
    return payload


def write_stage_snapshots(
    *, run_dir: Path, snapshots: Mapping[str, StageSnapshot]
) -> Path:
    """Write ``<run_dir>/stage_snapshots.json`` — the recorded stage registry.

    Deterministic like every other lineage artifact: sorted keys, indented,
    trailing newline, no wall-clock timestamps. Rewritten in full on each
    capture so a run that dies mid-lifecycle still leaves the stages it did
    reach. The caller isolates failures.
    """
    path = Path(run_dir) / "stage_snapshots.json"
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "stages": stage_snapshots_payload(snapshots),
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path


def child_provenance(
    parent_ref: str,
    *,
    branch_stage: str,
    snapshot: StageSnapshot | StateSnapshot | None,
    delta: BranchDelta | dict[str, Any] | None,
    cut_point: dict[str, Any] | None = None,
    delta_execution: str | None = None,
) -> dict[str, Any]:
    """The ``kind="benchflow-branch"`` source-provenance dict (RFC §3.4).

    The branch twin of the ``kind="benchflow-continue"`` provenance a
    continued run carries: which rollout the child forked from, at which stage
    boundary, from which snapshot refs, and under which (content-addressed)
    delta — either a :class:`BranchDelta` or the provenance dict the engine
    recorded on the child node at fork time. ``cut_point`` stays ``None``
    until the replay cut-point API lands.

    ``delta_execution`` names how the engine executed the delta when that is
    not the ordinary in-place child — ``"fresh-rollout"`` for a ``skill_mode``
    child, which re-runs installation as its own Rollout over the restored
    snapshot. The key is written only when given, so a child run in place
    keeps the exact dict shape it has always had.
    """
    refs = _snapshot_refs(snapshot) or {"environment": None, "sandbox": None}
    provenance: dict[str, Any] = {
        "kind": "benchflow-branch",
        "parent_rollout": parent_ref,
        "branch_stage": branch_stage,
        "snapshot_ref": {
            "sandbox": refs["sandbox"],
            "environment": refs["environment"],
        },
        "cut_point": cut_point,
        "delta": _provenance(delta),
    }
    if delta_execution is not None:
        provenance[_EXECUTION_KEY] = delta_execution
    return provenance


def branch_child_dir(run_dir: Path, parent_id: str, child_id: str) -> Path:
    """The per-child artifact directory of one branch event (RFC §3.4).

    Namespaced by node id (unique within the tree) so a second branch event —
    at the same parent or a different one — can never overwrite an earlier
    event's artifacts. A child the engine ran as its own Rollout uses this same
    directory as its run directory, so the standard ``config.json`` /
    ``result.json`` / trajectory files land beside the branch's own
    ``provenance.json``.
    """
    return Path(run_dir) / "branches" / parent_id / "children" / child_id


def write_branch_artifacts(
    *,
    run_dir: Path,
    tree: RolloutTree,
    parent: RolloutNode,
    children: Sequence[RolloutNode],
) -> None:
    """Write a completed branch's lineage artifacts under ``run_dir``.

    ``tree.json`` at the top level; per child, :func:`branch_child_dir` holding
    ``provenance.json`` and — when the child recorded a return —
    ``reward.json``. Each child's delta provenance and execution mode are read
    from ``child.state`` (attached by the engine at fork time), never inferred
    positionally. The caller isolates failures: any exception here must be
    caught and logged so an artifact-write error never corrupts branch results.
    """
    snapshot = parent.state.get(_SNAPSHOT_KEY)
    stage = getattr(snapshot, "stage", None)
    branch_stage = stage if stage is not None else f"cursor:{parent.id}"
    serialize_tree(tree, run_dir=run_dir)
    for child in children:
        child_dir = branch_child_dir(run_dir, parent.id, child.id)
        child_dir.mkdir(parents=True, exist_ok=True)
        provenance = child_provenance(
            str(run_dir),
            branch_stage=branch_stage,
            snapshot=snapshot,
            delta=child.state.get(_DELTA_KEY),
            delta_execution=child.state.get(_EXECUTION_KEY),
        )
        (child_dir / "provenance.json").write_text(
            json.dumps(provenance, sort_keys=True, indent=2) + "\n"
        )
        if _REWARD_KEY in child.state:
            (child_dir / "reward.json").write_text(
                json.dumps({"reward": float(child.state[_REWARD_KEY])}) + "\n"
            )
