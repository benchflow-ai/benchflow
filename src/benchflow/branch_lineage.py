"""Allowlisted fork observations, inspired by JeremyJC67 PR #1046.

This display schema is not a checkpoint import format. Never serialize arbitrary
node state, step payloads, provider metadata, configs or exception messages.
"""

from __future__ import annotations

import asyncio
import copy
import math
import re
from pathlib import Path
from typing import Any

from benchflow.branch import StageSnapshot
from benchflow.environment.protocol import StateSnapshot
from benchflow.review.persistence import write_json_atomic

_USAGE_COUNTERS = (
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def child_cost(
    native_usage: Any, timing: Any, sandbox_seconds: float | None = None
) -> dict[str, dict]:
    """Token counters, phase seconds and cost of one child, for tree.json.

    ``cost``: total tokens, USD when the provider reported it (None for
    native-subscription runs, which report no price), and the seconds the
    child held a sandbox (the parent's for an in-place child, its own for an
    isolated one).
    """
    usage = native_usage if isinstance(native_usage, dict) else {}
    phases = timing if isinstance(timing, dict) else {}
    total = usage.get("total_tokens")
    return {
        "cost": {
            "tokens": int(total)
            if isinstance(total, int) and not isinstance(total, bool)
            else 0,
            "usd": _finite(usage.get("cost_usd")),
            "sandbox_seconds": None
            if sandbox_seconds is None
            else round(sandbox_seconds, 3),
        },
        "usage": {
            key: int(usage.get(key) or 0)
            for key in _USAGE_COUNTERS
            if isinstance(usage.get(key) or 0, int)
        },
        "timing_sec": {
            key: round(float(value), 3)
            for key, value in phases.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        },
    }


def fork_cost(
    record: dict[str, Any], *, wall_seconds: float, isolated: bool
) -> dict[str, Any]:
    """A fork's cost: its children's tokens and USD, and sandbox seconds.

    The parent's sandbox is alive for the whole fork (``wall_seconds``); an
    in-place child runs inside it, an isolated child adds its own sandbox.
    ``usd`` is summed only when every child reported one (``usd_known``).
    """
    costs = [child.get("cost") or {} for child in record["children"]]
    usds = [cost.get("usd") for cost in costs]
    children_seconds = (
        sum(cost.get("sandbox_seconds") or 0.0 for cost in costs) if isolated else 0.0
    )
    known = bool(usds) and all(usd is not None for usd in usds)
    wall = round(wall_seconds, 3)
    return {
        "tokens": sum(cost.get("tokens") or 0 for cost in costs),
        "usd": round(math.fsum(u for u in usds if u is not None), 6) if known else None,
        "usd_known": known,
        "wall_seconds": wall,
        "parent_sandbox_seconds": wall,
        "children_sandbox_seconds": round(children_seconds, 3),
        "sandbox_seconds": round(wall + children_seconds, 3),
    }


def cost_totals(forks: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over a trial's forks. A nested fork's parent sandbox is an
    isolated child's, already counted in that child's sandbox seconds."""
    child_nodes = {child["node_id"] for fork in forks for child in fork["children"]}
    costs = [(fork, fork.get("cost") or {}) for fork in forks]
    usds = [cost.get("usd") for _, cost in costs]
    known = bool(usds) and all(usd is not None for usd in usds)
    seconds = 0.0
    for fork, cost in costs:
        seconds += cost.get("children_sandbox_seconds") or 0.0
        if fork.get("rollout") not in child_nodes:
            seconds += cost.get("parent_sandbox_seconds") or 0.0
    return {
        "tokens": sum(cost.get("tokens") or 0 for _, cost in costs),
        "usd": round(math.fsum(u for u in usds if u is not None), 6) if known else None,
        "usd_known": known,
        "sandbox_seconds": round(seconds, 3),
    }


def branch_summary(forks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The ``branches`` block of result.json: forks, children and their cost.

    ``final_metrics`` and ``timing`` in result.json cover the parent's own
    turns only; children's native token usage and phase time are summed here
    so dashboards built on result.json can count them. None when the rollout
    never branched.
    """
    if not forks:
        return None
    children = [child for fork in forks for child in fork["children"]]
    usage = dict.fromkeys(_USAGE_COUNTERS, 0)
    timing: dict[str, float] = {}
    for child in children:
        for key, value in (child.get("usage") or {}).items():
            usage[key] = usage.get(key, 0) + value
        for key, value in (child.get("timing_sec") or {}).items():
            timing[key] = round(timing.get(key, 0.0) + value, 3)
    child_nodes = {child["node_id"] for child in children}
    discarded = any(
        fork.get("parent_restore") == "skipped"
        for fork in forks
        if fork.get("rollout") not in child_nodes
    )
    return {
        "tree": "tree.json",
        # "discarded": the trial's own parent was not restored after a fork
        # (``--parent discard``), so its missing reward is by design.
        "parent": "discarded" if discarded else "kept",
        "forks": [
            {
                "id": fork["id"],
                "status": fork["status"],
                "value": fork["value"],
                "children": len(fork["children"]),
                "scored": sum(c["status"] == "scored" for c in fork["children"]),
                "parent_node": fork["parent_node"],
                "parent_restore": fork["parent_restore"],
                "cost": fork.get("cost"),
                "nodes": [
                    {
                        "index": child["index"],
                        "node_id": child["node_id"],
                        "label": child["intervention"]["label"],
                        "status": child["status"],
                        "reward": child["reward"],
                        "reward_source": child["reward_source"],
                        "path": child["artifacts"]["path"],
                    }
                    for child in fork["children"]
                ],
            }
            for fork in forks
        ],
        "children": len(children),
        "cost": cost_totals(forks),
        "child_usage": usage,
        "child_timing_sec": timing,
    }


class UnscoredChildError(RuntimeError):
    """No canonical verifier observation exists for this child."""


class NonfiniteChildReward(ValueError):
    """A runner returned a non-finite number, which is not a score."""


def error_record(exc: BaseException) -> dict[str, str]:
    code = (
        "missing_verifier_reward"
        if isinstance(exc, UnscoredChildError)
        else "nonfinite_reward"
        if isinstance(exc, NonfiniteChildReward)
        else "cancelled"
        if isinstance(exc, asyncio.CancelledError)
        else "execution_failed"
    )
    return {"type": type(exc).__name__, "code": code}


def child_status(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if isinstance(exc, UnscoredChildError):
        return "unscored"
    return "failed"


class ForkRecord:
    def __init__(
        self,
        rollout: Any,
        run_dir: Path | None,
        fork_id: str,
        parent: Any,
        layers: frozenset[str],
        n: int,
        labels: list[str | None] | None,
        requests: list[str | None] | None = None,
    ):
        self.rollout = rollout
        self.run_dir = run_dir
        # Seconds, rounded to ms: checkpoint, one restore per started child,
        # and the parent restore (None when skipped or not reached).
        self.timing: dict[str, Any] = {
            "checkpoint": None,
            "child_restore": [],
            # Every child, restores and sandbox creation included.
            "children": None,
            "parent_restore": None,
        }
        self.record: dict[str, Any] = {
            "id": fork_id,
            # The rollout that forked: the trial, or an isolated child's
            # sub-rollout (named by its node id) for a nested fork.
            "rollout": getattr(rollout, "_rollout_name", None),
            "parent_node": parent.id,
            "requested_children": n,
            "snapshot": {
                "requested_layers": sorted(layers),
                "captured_layers": [],
                "environment": None,
                "sandbox": None,
                "excluded": [
                    "agent_session",
                    "process_memory",
                    "mounted_contents",
                    "external_services",
                    "physical_state",
                ],
                # fresh: each child starts a new agent session; resumed:
                # children resume the parent's (branch(resume_session=True)).
                "agent_session": "fresh",
                "lifetime": "provider_local",
                "restore_available": None,
                # Sandbox layer only: kept | deleted | deferred | delete_failed.
                "retention": None,
            },
            "status": "running",
            # not_attempted | restored | deferred | failed | skipped
            # (skipped: restore_parent=False; the rollout is discarded).
            "parent_restore": "not_attempted",
            "timing_sec": self.timing,
            "value": None,
            "error": None,
            "parent_restore_error": None,
            "artifact_error": None,
            "children": [
                {
                    "index": index,
                    "node_id": None,
                    "status": "not_started",
                    "reward": None,
                    "reward_source": None,
                    "error": None,
                    "cleanup_error": None,
                    "intervention": {
                        "label": labels[index] if labels else None,
                        "requested": requests[index] if requests else None,
                        # runner: the caller's runner applied the request.
                        "execution": "runner"
                        if requests and requests[index] is not None
                        else "unspecified",
                        "evidence": None,
                    },
                    "artifacts": {"status": "unavailable", "path": None},
                }
                for index in range(n)
            ],
        }
        if not hasattr(rollout, "_branch_forks"):
            rollout._branch_forks = []
        rollout._branch_forks.append(self.record)
        # A sub-rollout's forks also go into its root trial's tree.json.
        self.forks: list[dict[str, Any]] = (
            getattr(rollout, "_lineage_forks", None) or rollout._branch_forks
        )
        if self.forks is not rollout._branch_forks:
            self.forks.append(self.record)

    def captured(self, snapshot: Any) -> None:
        env = (
            snapshot.environment_ref
            if isinstance(snapshot, StageSnapshot)
            else snapshot
        )
        sandbox = snapshot.sandbox_ref if isinstance(snapshot, StageSnapshot) else None
        target = self.record["snapshot"]
        if isinstance(env, StateSnapshot):
            spec = getattr(
                getattr(self.rollout._environment, "_manifest", None), "state", None
            )
            target["environment"] = {
                "id": env.id,
                "kind": getattr(spec, "kind", None),
                "paths": list(getattr(spec, "paths", [])),
            }
            target["captured_layers"].append("environment")
        if sandbox is not None:
            digest = sandbox.meta.get("digest")
            target["sandbox"] = {
                "provider": sandbox.provider,
                "ref": sandbox.ref,
                "digest": digest
                if isinstance(digest, str)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                else None,
            }
            target["captured_layers"].append("sandbox")
        target["captured_layers"].sort()

    def artifacts(self, index: int, directory: Path) -> None:
        """Link a child archive once its observation and custody succeeded."""
        assert self.run_dir is not None
        relative = directory.resolve().relative_to(self.run_dir.resolve())
        self.record["children"][index]["artifacts"] = {
            "status": "available",
            "path": relative.as_posix(),
        }

    def persist(self) -> None:
        if self.run_dir is None:
            return
        document = {
            "schema_version": 1,
            "kind": "benchflow-branch-tree",
            "nodes": [
                {
                    "id": node.id,
                    "parent": node.parent.id if node.parent else None,
                    "step_id": node.step_in.id if node.step_in else None,
                }
                for node in self.rollout._tree.nodes()
            ],
            "forks": copy.deepcopy(self.forks),
        }
        write_json_atomic(self.run_dir / "tree.json", document)
        self._write_labels()

    def _write_labels(self) -> None:
        """``branches/<fork>/labels.json``: child folders are named by node
        id, so index them by label (with status and reward)."""
        assert self.run_dir is not None
        paths = [
            Path(child["artifacts"]["path"])
            for child in self.record["children"]
            if (child.get("artifacts") or {}).get("path")
        ]
        if not paths:
            return
        fork_dir = self.run_dir / paths[0].parent.parent
        if not fork_dir.is_dir():
            return
        write_json_atomic(
            fork_dir / "labels.json",
            {
                "kind": "benchflow-branch-labels",
                "fork_id": self.record["id"],
                "children": [
                    {
                        "label": child["intervention"]["label"],
                        "node_id": child["node_id"],
                        "folder": Path(child["artifacts"]["path"]).name
                        if child["artifacts"].get("path")
                        else None,
                        "status": child["status"],
                        "reward": child["reward"],
                    }
                    for child in self.record["children"]
                ],
            },
        )
