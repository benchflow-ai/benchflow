"""Run discovery and sidebar summaries for browse mode."""

import os
from pathlib import Path
from typing import Any

from .models import RunSummary
from .payload import _is_acp_rollout_dir, _load_lineage, _load_rollout_metadata


def _runs_cap() -> int:
    """Browse-mode run cap (``BENCHFLOW_VIEWER_MAX_RUNS`` overrides, default 500)."""
    try:
        return max(1, int(os.environ.get("BENCHFLOW_VIEWER_MAX_RUNS", "500")))
    except ValueError:
        return 500


# Deep enough for a hill-climb run's evals/<version>/<split>/trial-NN/job/<trial>.
_MAX_DEPTH = 8


def _discover_rollouts(
    base: Path, max_depth: int = _MAX_DEPTH, cap: int | None = None
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
    return _rollout_summary_at(base / rel_id, rel_id)


def _rollout_summary_at(d: Path, rel_id: str) -> dict[str, Any]:
    """Catalog row for the rollout folder ``d``, listed under ``rel_id``."""
    metadata = _load_rollout_metadata(d)
    lineage = _load_lineage(d)
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
        status=metadata.status,
        branch_forks=len(lineage.forks) if lineage is not None else None,
        branch_children=sum(len(fork.children) for fork in lineage.forks)
        if lineage is not None
        else None,
    ).to_payload()


def root_labels(paths: list[Path]) -> list[str]:
    """A distinct, path-free label per served job root (its folder name)."""
    labels: list[str] = []
    for path in paths:
        name = path.name or "job"
        label, n = name, 2
        while label in labels:
            label, n = f"{name}-{n}", n + 1
        labels.append(label)
    return labels


class BrowseRoots:
    """One or more served job folders and the ids of the rollouts under them.

    With one root, ids are paths relative to it (the historical ids). With
    several, each id starts with its root's label (:func:`root_labels`), so
    ``/api/rollout`` resolves ids only by membership in a fresh scan, never
    as paths, whatever the number of roots.
    """

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.labels = [""] if len(paths) == 1 else root_labels(paths)

    def pairs(self) -> list[tuple[str, Path]]:
        return list(zip(self.labels, self.paths, strict=True))

    def _prefixed(self, label: str, rel: str) -> str:
        if not label:
            return rel
        return label if rel in ("", ".") else f"{label}/{rel}"

    def scan(self, cap: int) -> dict[str, Path]:
        """Rollout id -> folder, at most ``cap`` of them, in root order."""
        found: dict[str, Path] = {}
        for label, base in self.pairs():
            if len(found) >= cap:
                break
            if label and _is_acp_rollout_dir(base):
                found[label] = base
                continue
            for rel in _discover_rollouts(base, cap=cap - len(found)):
                found[self._prefixed(label, rel)] = base / rel
        return found

    def id_for(self, trial_dir: Path) -> str | None:
        """The id of a trial folder under a root, when it is a rollout."""
        if not _is_acp_rollout_dir(trial_dir):
            return None
        for label, base in self.pairs():
            try:
                rel = trial_dir.relative_to(base).as_posix()
            except ValueError:
                continue
            if not label and rel == ".":
                return None
            return self._prefixed(label, rel)
        return None
