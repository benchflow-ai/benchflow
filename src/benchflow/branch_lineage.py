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
from benchflow.trajectories.tree import RolloutNode, RolloutTree, branch_points

_SCHEMA_VERSION = 1
_SNAPSHOT_KEY = "snapshot"
_REWARD_KEY = "reward"
_VALUE_KEY = "value"


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


def _provenance(delta: BranchDelta | None) -> dict[str, Any]:
    """A delta's provenance dict; ``None`` records as the zero delta."""
    return (delta if delta is not None else BranchDelta()).provenance_dict()


def serialize_tree(
    tree: RolloutTree,
    *,
    run_dir: Path,
    stage: str | None = None,
    snapshot: StageSnapshot | StateSnapshot | None = None,
    deltas: Sequence[BranchDelta | None] | None = None,
    cut_point: dict[str, Any] | None = None,
) -> Path:
    """Write ``<run_dir>/tree.json`` — the deterministic lineage artifact.

    Every node serializes as id / parent / stage tag / snapshot refs, plus
    ``reward`` and ``value`` when recorded on the node. The keyword context
    describes the branch just taken: ``snapshot`` identifies the branch node
    (matched by identity against ``node.state["snapshot"]``, falling back to
    the tree's unique branch point), ``stage`` names its boundary when the
    snapshot itself does not, ``deltas`` aligns positionally with that node's
    children (each child entry gains a ``delta`` provenance dict), and
    ``cut_point`` is recorded at the top level. Output is deterministic —
    sorted keys, indented, trailing newline, no wall-clock timestamps.
    """
    branch_node: RolloutNode | None = None
    if snapshot is not None:
        for node in tree.nodes():
            if node.state.get(_SNAPSHOT_KEY) is snapshot:
                branch_node = node
                break
    if branch_node is None and deltas:
        candidates = branch_points(tree)
        if len(candidates) == 1:
            branch_node = candidates[0]

    delta_for: dict[int, dict[str, Any]] = {}
    if branch_node is not None and deltas is not None:
        # Positional alignment, non-strict: a post-branch linear continuation
        # may have grown extra children past the delta'd ones.
        for child, delta in zip(branch_node.children, deltas, strict=False):
            delta_for[id(child)] = _provenance(delta)

    nodes_payload: list[dict[str, Any]] = []
    for node in tree.nodes():
        snap = node.state.get(_SNAPSHOT_KEY)
        node_stage = getattr(snap, "stage", None)
        if node is branch_node and node_stage is None:
            node_stage = stage
        entry: dict[str, Any] = {
            "id": node.id,
            "parent": node.parent.id if node.parent is not None else None,
            "stage": node_stage,
            "snapshot": _snapshot_refs(snap),
        }
        if _REWARD_KEY in node.state:
            entry["reward"] = float(node.state[_REWARD_KEY])
        if _VALUE_KEY in node.state:
            entry["value"] = float(node.state[_VALUE_KEY])
        if id(node) in delta_for:
            entry["delta"] = delta_for[id(node)]
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
    delta: BranchDelta | None,
    cut_point: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``kind="benchflow-branch"`` source-provenance dict (RFC §3.4).

    The branch twin of the ``kind="benchflow-continue"`` provenance a
    continued run carries: which rollout the child forked from, at which stage
    boundary, from which snapshot refs, and under which (content-addressed)
    delta. ``cut_point`` stays ``None`` until the replay cut-point API lands.
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
    deltas: Sequence[BranchDelta | None] | None,
) -> None:
    """Write a completed branch's lineage artifacts under ``run_dir``.

    ``tree.json`` at the top level; per child, ``children/<index>/``
    holding ``provenance.json`` and — when the child recorded a return —
    ``reward.json``. The caller isolates failures: any exception here must be
    caught and logged so an artifact-write error never corrupts branch
    results.
    """
    snapshot = parent.state.get(_SNAPSHOT_KEY)
    stage = getattr(snapshot, "stage", None)
    branch_stage = stage if stage is not None else f"cursor:{parent.id}"
    serialize_tree(
        tree, run_dir=run_dir, stage=stage, snapshot=snapshot, deltas=deltas
    )
    for index, child in enumerate(children):
        child_dir = Path(run_dir) / "children" / str(index)
        child_dir.mkdir(parents=True, exist_ok=True)
        provenance = child_provenance(
            str(run_dir),
            branch_stage=branch_stage,
            snapshot=snapshot,
            delta=deltas[index] if deltas is not None else None,
        )
        (child_dir / "provenance.json").write_text(
            json.dumps(provenance, sort_keys=True, indent=2) + "\n"
        )
        if _REWARD_KEY in child.state:
            (child_dir / "reward.json").write_text(
                json.dumps({"reward": float(child.state[_REWARD_KEY])}) + "\n"
            )
