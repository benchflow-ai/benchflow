"""Branch lineage artifacts — ``tree.json`` and per-child provenance (RFC §3.4).

Today ``RolloutTree`` lives and dies in memory; branching must leave the same
quality of evidence as a linear run. This module serializes the tree to a
deterministic ``tree.json`` in the run folder and builds the
``kind="benchflow-branch"`` source-provenance dict each branch child carries —
the same seam ``benchflow-continue`` uses.

Pure writers only: no engine state, no wall-clock timestamps (determinism is a
test guarantee — goldens pin the output byte-for-byte). Failure isolation is
the caller's job: the engine wraps these writes so an artifact error never
corrupts branch results.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
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
    ``reward``, ``value`` and ``delta`` (the provenance dict the engine
    attached at fork time) when present on the node. Nothing is inferred
    from position — a tree with several branch points, or several branch
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
        nodes_payload.append(entry)

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "cut_point": cut_point,
        "nodes": nodes_payload,
    }
    path = Path(run_dir) / "tree.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path


def child_provenance(
    parent_ref: str,
    *,
    branch_stage: str,
    snapshot: StageSnapshot | StateSnapshot | None,
    delta: BranchDelta | dict[str, Any] | None,
    cut_point: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``kind="benchflow-branch"`` source-provenance dict (RFC §3.4).

    The branch twin of the ``kind="benchflow-continue"`` provenance a
    continued run carries: which rollout the child forked from, at which stage
    boundary, from which snapshot refs, and under which (content-addressed)
    delta — either a :class:`BranchDelta` or the provenance dict the engine
    recorded on the child node at fork time. ``cut_point`` stays ``None``
    until the replay cut-point API lands.
    """
    refs = _snapshot_refs(snapshot) or {"environment": None, "sandbox": None}
    return {
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


def write_branch_artifacts(
    *,
    run_dir: Path,
    tree: RolloutTree,
    parent: RolloutNode,
    children: Sequence[RolloutNode],
) -> None:
    """Write a completed branch's lineage artifacts under ``run_dir``.

    ``tree.json`` at the top level; per child,
    ``branches/<branch-node-id>/children/<child-node-id>/`` holding
    ``provenance.json`` and — when the child recorded a return —
    ``reward.json``. Namespacing by node id (unique within the tree) means a
    second branch event — at the same parent or a different one — can never
    overwrite an earlier event's artifacts. Each child's delta provenance is
    read from ``child.state["delta"]`` (attached by the engine at fork time),
    never inferred positionally. The caller isolates failures: any exception
    here must be caught and logged so an artifact-write error never corrupts
    branch results.
    """
    snapshot = parent.state.get(_SNAPSHOT_KEY)
    stage = getattr(snapshot, "stage", None)
    branch_stage = stage if stage is not None else f"cursor:{parent.id}"
    serialize_tree(tree, run_dir=run_dir)
    for child in children:
        child_dir = Path(run_dir) / "branches" / parent.id / "children" / child.id
        child_dir.mkdir(parents=True, exist_ok=True)
        provenance = child_provenance(
            str(run_dir),
            branch_stage=branch_stage,
            snapshot=snapshot,
            delta=child.state.get(_DELTA_KEY),
        )
        (child_dir / "provenance.json").write_text(
            json.dumps(provenance, sort_keys=True, indent=2) + "\n"
        )
        if _REWARD_KEY in child.state:
            (child_dir / "reward.json").write_text(
                json.dumps({"reward": float(child.state[_REWARD_KEY])}) + "\n"
            )
