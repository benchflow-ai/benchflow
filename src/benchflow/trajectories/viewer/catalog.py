"""Run discovery and sidebar summaries for browse mode."""

import os
from pathlib import Path
from typing import Any

from .models import RunSummary
from .payload import _is_acp_rollout_dir, _load_rollout_metadata


def _runs_cap() -> int:
    """Browse-mode run cap (``BENCHFLOW_VIEWER_MAX_RUNS`` overrides, default 500)."""
    try:
        return max(1, int(os.environ.get("BENCHFLOW_VIEWER_MAX_RUNS", "500")))
    except ValueError:
        return 500


def _discover_rollouts(
    base: Path, max_depth: int = 4, cap: int | None = None
) -> list[str]:
    """Relative paths of ACP rollout dirs under ``base``, sorted, capped.

    The returned ids double as the ``/api/rollout?id=`` whitelist: an id is
    only ever resolved by exact membership here, so crafted ids (``../``
    traversal and the like) can never reach the filesystem. Directory
    symlinks under ``base`` are followed — serving a directory implies
    trusting what it links to.
    """
    if cap is None:
        cap = _runs_cap()
    found: list[str] = []

    def walk(d: Path, depth: int) -> None:
        if len(found) >= cap:
            return
        if _is_acp_rollout_dir(d):
            found.append(d.relative_to(base).as_posix())
            return  # rollout dirs don't nest
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            return
        for child in children:
            if child.name.startswith("."):
                continue
            walk(child, depth + 1)

    try:
        top = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return found
    for child in top:
        if not child.name.startswith("."):
            walk(child, 1)
    return found


def _resolve_browse_rollout(base: Path, rid: str | None) -> Path | None:
    """Resolve an ``/api/rollout`` id strictly by whitelist membership.

    The id is never interpreted as a path unless a fresh scan discovered it,
    so crafted ids (``../`` traversal, absolute paths) return ``None`` even
    when the traversed-to path exists and is a real rollout.
    """
    if rid is None or rid not in set(_discover_rollouts(base)):
        return None
    return base / rid


def _rollout_summary(base: Path, rel_id: str) -> dict[str, Any]:
    """Catalog row for one rollout: identity, verdict, row-level stats."""
    d = base / rel_id
    metadata = _load_rollout_metadata(d)
    return RunSummary(
        id=rel_id,
        name=d.name,
        task_name=metadata.task_name or d.name,
        agent_name=metadata.agent_name,
        model=metadata.model,
        reward=metadata.reward,
        has_error=metadata.has_error,
        skill_mode=metadata.skill_mode,
        duration_sec=metadata.timing.total if metadata.timing is not None else None,
        cost_usd=metadata.usage.cost_usd,
        total_tokens=metadata.usage.total_tokens,
        n_tool_calls=metadata.n_tool_calls,
    ).to_payload()
