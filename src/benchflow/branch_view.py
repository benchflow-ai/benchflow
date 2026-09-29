"""The branch-view contract: one versioned JSON projection of a trial's branches.

``load_branch_view(trial_dir)`` reads the trial's ``tree.json`` and
``result.json`` and returns a ``benchflow.branch-view/1`` document: per fork
and per child what was requested, tokens, USD, sandbox-seconds, timings,
reuse of the snapshot, and lineage (fork depth, which child a nested fork was
made from, which forks a child made). It is what a viewer or a script should
read instead of the raw files, whose fields grew across releases;
fields an older tree lacks read as null. The document is described by
``BRANCH_VIEW_SCHEMA`` (JSON Schema) and ``docs/reference/branch-view.md``.

Provider snapshot refs are capability handles and never appear; neither do
prompts (a child's ``requested`` is a short description with a digest) or
error messages (errors are ``{type, code}``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCHEMA_ID = "benchflow.branch-view/1"
#: Minor version within ``/1``: fields are only added (changelog in
#: docs/reference/branch-view.md).
SCHEMA_VERSION = "1.1"
KIND = "benchflow.branch-view"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _error(value: Any) -> dict[str, Any] | None:
    value = _obj(value)
    return {"type": value.get("type"), "code": value.get("code")} if value else None


def _archive(rel: Any, file_exists: Callable[[str], bool]) -> dict[str, Any]:
    if not isinstance(rel, str) or not rel:
        return {"path": None, "observation": None, "result": None}
    observation, result = f"{rel}/observation.json", f"{rel}/result.json"
    return {
        "path": rel,
        "observation": observation if file_exists(observation) else None,
        "result": result if file_exists(result) else None,
    }


def _cost_totals(forks: list[dict[str, Any]]) -> dict[str, Any]:
    """Trial totals; a nested fork's parent sandbox (an isolated child's) is
    already in that child's seconds. Same rule as branch_lineage.cost_totals."""
    child_nodes = {
        child.get("node_id")
        for fork in forks
        for child in fork.get("children") or []
        if isinstance(child, dict)
    }
    costs = [(fork, _obj(fork.get("cost"))) for fork in forks]
    usds = [cost.get("usd") for _, cost in costs]
    known = bool(usds) and all(isinstance(usd, int | float) for usd in usds)
    seconds = 0.0
    for fork, cost in costs:
        seconds += cost.get("children_sandbox_seconds") or 0.0
        if fork.get("rollout") not in child_nodes:
            seconds += cost.get("parent_sandbox_seconds") or 0.0
    return {
        "tokens": sum(cost.get("tokens") or 0 for _, cost in costs),
        "usd": round(math.fsum(u for u in usds if isinstance(u, int | float)), 6)
        if known
        else None,
        "usd_known": known,
        "sandbox_seconds": round(seconds, 3),
    }


_USAGE_CLASSES = (
    ("input", "n_input_tokens"),
    ("output", "n_output_tokens"),
    ("cache_read", "n_cache_read_tokens"),
    ("cache_creation", "n_cache_creation_tokens"),
    ("total", "total_tokens"),
)


def _usage_classes(usages: list[Any]) -> dict[str, int] | None:
    """Summed token classes of child ``usage`` records; None when none has one."""
    records = [u for u in usages if isinstance(u, dict)]
    if not records:
        return None
    return {
        name: sum(
            int(u.get(key) or 0)
            for u in records
            if isinstance(u.get(key) or 0, int) and not isinstance(u.get(key), bool)
        )
        for name, key in _USAGE_CLASSES
    }


def _estimate(usage: Any, prices: dict[str, Any] | None) -> float | None:
    """USD at list price from token classes. Zero tokens cost 0 (an oracle
    calls no model); otherwise None without a price or a usage record."""
    classes = _usage_classes([usage])
    if classes is None:
        return None
    if not any(classes[name] for name, _ in _USAGE_CLASSES):
        return 0.0
    per = _obj(_obj(prices).get("usd_per_token"))
    if not per:
        return None
    total = math.fsum(
        classes[name] * float(per.get(name) or per.get("input") or 0.0)
        if name == "cache_creation"
        else classes[name] * float(per.get(name) or 0.0)
        for name in ("input", "output", "cache_read", "cache_creation")
    )
    return round(total, 8)


def _sum_or_none(values: list[float | None]) -> float | None:
    if not values or any(v is None for v in values):
        return None
    return round(math.fsum(v for v in values if v is not None), 8)


_PRICE_TABLE: dict[str, Any] | None = None


def _bundled_table() -> tuple[dict[str, Any], str]:
    """LiteLLM's bundled price table, read as a file (litellm is not
    imported: importing it takes seconds). Empty when litellm is absent."""
    global _PRICE_TABLE
    if _PRICE_TABLE is None:
        import importlib.util
        from importlib import metadata

        table: dict[str, Any] = {}
        source = ""
        try:
            spec = importlib.util.find_spec("litellm")
            roots = list(spec.submodule_search_locations or []) if spec else []
            for root in roots:
                path = Path(root) / "model_prices_and_context_window_backup.json"
                if path.is_file():
                    table = _obj(_json(path))
                    source = (
                        "litellm "
                        + metadata.version("litellm")
                        + " bundled price table"
                    )
                    break
        except (ImportError, ValueError, metadata.PackageNotFoundError):
            table = {}
        _PRICE_TABLE = {"table": table, "source": source}
    return _PRICE_TABLE["table"], _PRICE_TABLE["source"]


def bundled_prices(model: Any) -> dict[str, Any] | None:
    """``pricing`` for ``model`` from LiteLLM's bundled price table, or None
    (no model, unknown model, litellm not installed)."""
    if not isinstance(model, str) or not model:
        return None
    table, source = _bundled_table()
    for name in (model, model.rsplit("/", 1)[-1]):
        entry = _obj(table.get(name))
        if isinstance(entry.get("input_cost_per_token"), int | float):
            return {
                "model": name,
                "source": source,
                "usd_per_token": {
                    "input": entry.get("input_cost_per_token"),
                    "output": entry.get("output_cost_per_token"),
                    "cache_read": entry.get("cache_read_input_token_cost"),
                    "cache_creation": entry.get("cache_creation_input_token_cost"),
                },
            }
    return None


def load_branch_view(
    trial_dir: str | Path, *, prices: dict[str, Any] | None | str = "bundled"
) -> dict[str, Any]:
    """The ``benchflow.branch-view/1`` document of one trial directory.

    ``prices``: ``"bundled"`` (LiteLLM's bundled list prices for the trial's
    model), a ``pricing`` dict, or None for no estimate."""
    trial = Path(trial_dir)
    result = _obj(_json(trial / "result.json"))
    return build_branch_view(
        _obj(_json(trial / "tree.json")),
        result,
        trial_name=trial.name,
        file_exists=lambda rel: (trial / rel).is_file(),
        checkpoint_source=_json(trial / "checkpoint_source.json"),
        prices=bundled_prices(result.get("model"))
        if prices == "bundled"
        else prices
        if isinstance(prices, dict)
        else None,
    )


def build_branch_view(
    tree: dict[str, Any],
    result: dict[str, Any],
    *,
    trial_name: str,
    file_exists: Callable[[str], bool],
    prices: dict[str, Any] | None = None,
    checkpoint_source: Any = None,
) -> dict[str, Any]:
    """The document from parsed ``tree.json`` and ``result.json`` (``{}`` when
    absent); ``file_exists(path relative to the trial)`` answers whether a
    child's observation.json / result.json exists. ``prices`` (``{model,
    source, usd_per_token: {input, output, cache_read, cache_creation}}``,
    e.g. from :func:`bundled_prices`) turns token classes into
    ``usd_estimate``s; ``checkpoint_source`` is the parsed
    ``checkpoint_source.json`` of a trial started from a kept checkpoint.
    Pure: this module imports nothing from benchflow, so a
    viewer can copy it."""
    prices = prices if isinstance(prices, dict) else None
    tree, result = _obj(tree), _obj(result)
    raw_forks = [f for f in tree.get("forks") or [] if isinstance(f, dict)]
    if tree and (
        tree.get("kind") != "benchflow-branch-tree" or tree.get("schema_version") != 1
    ):
        raw_forks = []

    # child node id -> (fork id, label); a fork whose ``rollout`` is a child
    # node was made by that child (nested).
    child_of: dict[str, tuple[str, Any]] = {}
    for fork in raw_forks:
        for child in fork.get("children") or []:
            if isinstance(child, dict) and child.get("node_id"):
                label = _obj(child.get("intervention")).get("label")
                child_of[str(child["node_id"])] = (str(fork.get("id")), label)
    fork_by_id = {str(f.get("id")): f for f in raw_forks}

    def depth(fork: dict[str, Any], seen: frozenset[str] = frozenset()) -> int:
        owner = fork.get("rollout")
        if owner in child_of and fork.get("id") not in seen:
            parent = fork_by_id.get(child_of[owner][0])
            if parent is not None:
                return 1 + depth(parent, seen | {str(fork.get("id"))})
        return 1

    forks = []
    for fork in raw_forks:
        snapshot = _obj(fork.get("snapshot"))
        value = _num(fork.get("value"))
        owner = fork.get("rollout")
        nested_owner = owner if owner in child_of else None
        children = []
        for child in fork.get("children") or []:
            if not isinstance(child, dict):
                continue
            intervention = _obj(child.get("intervention"))
            reward = _num(child.get("reward"))
            node = child.get("node_id")
            children.append(
                {
                    "node_id": node,
                    "index": child.get("index"),
                    "label": intervention.get("label"),
                    "requested": intervention.get("requested"),
                    "execution": intervention.get("execution"),
                    "status": child.get("status"),
                    "reward": reward,
                    "reward_source": child.get("reward_source"),
                    "advantage": round(reward - value, 6)
                    if reward is not None and value is not None
                    else None,
                    "error": _error(child.get("error")),
                    "attempts": child.get("attempts")
                    if isinstance(child.get("attempts"), int)
                    else None,
                    "retried_after": _error(child.get("retried_after")),
                    "cost": child.get("cost")
                    if isinstance(child.get("cost"), dict)
                    else None,
                    "usage": child.get("usage")
                    if isinstance(child.get("usage"), dict)
                    else None,
                    "usd_estimate": _estimate(child.get("usage"), prices),
                    "timing_sec": child.get("timing_sec")
                    if isinstance(child.get("timing_sec"), dict)
                    else None,
                    "snapshot_start": child.get("snapshot_start")
                    if isinstance(child.get("snapshot_start"), dict)
                    else None,
                    "archive": _archive(
                        _obj(child.get("artifacts")).get("path"), file_exists
                    ),
                    "nested_forks": [
                        str(f.get("id")) for f in raw_forks if f.get("rollout") == node
                    ]
                    if node
                    else [],
                }
            )
        forks.append(
            {
                "id": fork.get("id"),
                "kind": fork.get("kind") or "fork",
                "reason": fork.get("reason"),
                "checkpoint": fork.get("checkpoint"),
                "parent_node": fork.get("parent_node"),
                "depth": depth(fork),
                "forked_by": {
                    "rollout": owner,
                    "child_node": nested_owner,
                    "child_label": child_of[nested_owner][1] if nested_owner else None,
                },
                "status": fork.get("status"),
                "value": value,
                "value_stderr": _num(fork.get("value_stderr")),
                "parent_restore": fork.get("parent_restore"),
                "error": _error(fork.get("error")),
                "children_mode": fork.get("children_mode")
                if isinstance(fork.get("children_mode"), dict)
                else None,
                "snapshot": {
                    "layers_requested": list(snapshot.get("requested_layers") or []),
                    "layers_captured": list(snapshot.get("captured_layers") or []),
                    "agent_session": snapshot.get("agent_session"),
                    "retention": snapshot.get("retention"),
                    "provider": _obj(snapshot.get("sandbox")).get("provider"),
                    # 1.1: the fork used an existing image (a kept checkpoint)
                    # instead of taking a snapshot.
                    "reused": snapshot.get("reused") is True,
                },
                "timing_sec": fork.get("timing_sec")
                if isinstance(fork.get("timing_sec"), dict)
                else None,
                "cost": fork.get("cost")
                if isinstance(fork.get("cost"), dict)
                else None,
                "usage": _usage_classes([c["usage"] for c in children]),
                "usd_estimate": _sum_or_none([c["usd_estimate"] for c in children]),
                "children": children,
            }
        )
    branches_cost = _obj(result.get("branches")).get("cost")
    if not isinstance(branches_cost, dict):
        branches_cost = (
            _cost_totals(raw_forks) if any("cost" in f for f in raw_forks) else None
        )
    rewards = _obj(result.get("rewards"))
    retry = result.get("retry")
    parent = _obj(result.get("branches")).get("parent")
    if parent not in ("discarded", "kept"):
        top = [f for f in raw_forks if f.get("rollout") not in child_of]
        parent = (
            None
            if not top
            else "discarded"
            if any(f.get("parent_restore") == "skipped" for f in top)
            else "kept"
        )
    all_children = [c for f in forks for c in f["children"]]
    scored = sum(c["reward"] is not None for c in all_children)
    total_estimate = _sum_or_none([c["usd_estimate"] for c in all_children])
    total_usage = _usage_classes([c["usage"] for c in all_children])
    total_tokens = (
        total_usage["total"]
        if total_usage is not None
        else _obj(branches_cost).get("tokens")
        if isinstance(_obj(branches_cost).get("tokens"), int)
        else None
    )
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_ID,
        "trial": {
            "name": result.get("rollout_name") or trial_name,
            "task": result.get("task_name"),
            "reward": _num(rewards.get("reward")),
            "retry": retry if isinstance(retry, dict) else None,
            "checkpoint_source": {
                "trial": source.get("trial"),
                "checkpoint": source.get("fork_id"),
                "provider": source.get("provider"),
                "prefix_events": source.get("prefix_events")
                if isinstance(source.get("prefix_events"), int)
                else None,
            }
            if (source := _obj(checkpoint_source))
            else None,
            "parent": parent,
            "unscored_by_design": parent == "discarded"
            and _num(rewards.get("reward")) is None,
        },
        "totals": {
            "forks": len(forks),
            "children": len(all_children),
            "cost": branches_cost,
            "usage": total_usage,
            "usd_estimate": total_estimate,
            "per_scored_child": {
                "scored": scored,
                "tokens": total_tokens // scored
                if scored and total_tokens is not None
                else None,
                "usd": round(_obj(branches_cost)["usd"] / scored, 6)
                if scored and isinstance(_obj(branches_cost).get("usd"), int | float)
                else None,
                "usd_estimate": round(total_estimate / scored, 8)
                if scored and total_estimate is not None
                else None,
                "sandbox_seconds": round(
                    _obj(branches_cost)["sandbox_seconds"] / scored, 3
                )
                if scored
                and isinstance(_obj(branches_cost).get("sandbox_seconds"), int | float)
                else None,
            },
        },
        "pricing": prices,
        "forks": forks,
    }


_NULL_STR = {"type": ["string", "null"]}
_NULL_NUM = {"type": ["number", "null"]}
_NULL_OBJ = {"type": ["object", "null"]}
_USAGE = {
    "type": ["object", "null"],
    "properties": {
        name: {"type": "integer"}
        for name in ("input", "output", "cache_read", "cache_creation", "total")
    },
}
_ERROR = {
    "type": ["object", "null"],
    "properties": {"type": _NULL_STR, "code": _NULL_STR},
}

BRANCH_VIEW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SCHEMA_ID,
    "type": "object",
    "required": ["schema", "trial", "totals", "forks"],
    "properties": {
        "kind": {"const": KIND},
        "schema_version": {
            "type": "string",
            "pattern": "^1\\.[0-9]+$",
            "description": "Minor version within /1 (1.0 documents lack it).",
        },
        "schema": {"const": SCHEMA_ID},
        "trial": {
            "type": "object",
            "required": ["name", "task", "reward", "retry"],
            "properties": {
                "name": {"type": "string"},
                "task": _NULL_STR,
                "reward": _NULL_NUM,
                "retry": _NULL_OBJ,
                "parent": {"enum": ["discarded", "kept", None]},
                "checkpoint_source": _NULL_OBJ,
                "unscored_by_design": {"type": "boolean"},
            },
        },
        "totals": {
            "type": "object",
            "required": ["forks", "children", "cost"],
            "properties": {
                "forks": {"type": "integer"},
                "children": {"type": "integer"},
                "cost": _NULL_OBJ,
                "usage": _USAGE,
                "usd_estimate": _NULL_NUM,
                "per_scored_child": {
                    "type": "object",
                    "properties": {
                        "scored": {"type": "integer"},
                        "tokens": {"type": ["integer", "null"]},
                        "usd": _NULL_NUM,
                        "usd_estimate": _NULL_NUM,
                        "sandbox_seconds": _NULL_NUM,
                    },
                },
            },
        },
        "pricing": {
            "type": ["object", "null"],
            "properties": {
                "model": {"type": "string"},
                "source": {"type": "string"},
                "usd_per_token": {"type": "object"},
            },
        },
        "forks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "kind",
                    "reason",
                    "checkpoint",
                    "parent_node",
                    "depth",
                    "forked_by",
                    "status",
                    "value",
                    "parent_restore",
                    "error",
                    "children_mode",
                    "snapshot",
                    "timing_sec",
                    "cost",
                    "children",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "kind": {"enum": ["fork", "retry"]},
                    "reason": _NULL_STR,
                    "checkpoint": _NULL_STR,
                    "parent_node": _NULL_STR,
                    "depth": {"type": "integer", "minimum": 1},
                    "forked_by": {
                        "type": "object",
                        "required": ["rollout", "child_node", "child_label"],
                    },
                    "status": _NULL_STR,
                    "value": _NULL_NUM,
                    "value_stderr": _NULL_NUM,
                    "parent_restore": _NULL_STR,
                    "error": _ERROR,
                    "children_mode": _NULL_OBJ,
                    "snapshot": {
                        "type": "object",
                        "required": [
                            "layers_requested",
                            "layers_captured",
                            "agent_session",
                            "retention",
                            "provider",
                        ],
                    },
                    "timing_sec": _NULL_OBJ,
                    "cost": _NULL_OBJ,
                    "usage": _USAGE,
                    "usd_estimate": _NULL_NUM,
                    "children": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "node_id",
                                "index",
                                "label",
                                "requested",
                                "execution",
                                "status",
                                "reward",
                                "reward_source",
                                "advantage",
                                "error",
                                "cost",
                                "usage",
                                "timing_sec",
                                "snapshot_start",
                                "archive",
                                "nested_forks",
                            ],
                            "properties": {
                                "node_id": _NULL_STR,
                                "label": _NULL_STR,
                                "requested": _NULL_STR,
                                "status": _NULL_STR,
                                "reward": _NULL_NUM,
                                "reward_source": _NULL_STR,
                                "advantage": _NULL_NUM,
                                "error": _ERROR,
                                "attempts": {"type": ["integer", "null"]},
                                "retried_after": _ERROR,
                                "cost": _NULL_OBJ,
                                "usage": _NULL_OBJ,
                                "usd_estimate": _NULL_NUM,
                                "timing_sec": _NULL_OBJ,
                                "snapshot_start": _NULL_OBJ,
                                "archive": {
                                    "type": "object",
                                    "required": ["path", "observation", "result"],
                                },
                                "nested_forks": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}
