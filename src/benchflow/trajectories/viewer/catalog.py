"""Run discovery and sidebar summaries for browse mode."""

import os
from pathlib import Path
from typing import Any

from .models import NotCounted, RunSummary
from .payload import (
    _is_acp_rollout_dir,
    _load_json,
    _load_lineage,
    _load_rollout_metadata,
)


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


def _not_counted(base: Path, ids: list[str]) -> dict[str, NotCounted]:
    """The runs among ``ids`` that the job does not count as trials, and why.

    The catalog counts the trials ``bf.load_job`` (and so ``bench eval
    metrics`` and summary.json) counts: a run with no result.json is still
    going or was killed (``Job.interrupted``), a reviewer run is not a trial,
    and in an Evaluation job folder the attempts of a retried task are one
    trial, its final attempt (``jobs._final_attempts``). Without this a
    superseded errored attempt, or a run still going, counted as a failed run
    in a group's pass rate.
    """
    from benchflow._utils.result_paths import holds_attempts
    from benchflow.jobs import _final_attempts, load_trial

    reasons: dict[str, NotCounted] = {}
    folders: dict[Path, bool] = {}
    attempts: dict[Path, str] = {}
    trials = []
    for rid in ids:
        d = base / rid
        result = _load_json(d / "result.json")
        if not isinstance(result, dict):
            reasons[rid] = "unfinished"
            continue
        if result.get("purpose") == "reviewer":
            reasons[rid] = "reviewer"
            continue
        if d.parent not in folders:
            folders[d.parent] = holds_attempts(d.parent)
        if not folders[d.parent]:
            continue  # independent samples (bf.run_batch, bf.run): all count
        try:
            trial = load_trial(d)
        except (FileNotFoundError, ValueError, TypeError):
            reasons[rid] = "unfinished"  # bf.load_job skips it too
            continue
        trials.append(trial)
        attempts[trial.path] = rid
    final = {t.path for t in _final_attempts(trials)}
    for path, rid in attempts.items():
        if path not in final:
            reasons[rid] = "superseded"
    return reasons


def _rollout_summaries(base: Path, ids: list[str]) -> list[dict[str, Any]]:
    """Catalog rows for ``ids``, each marked when the job does not count it."""
    reasons = _not_counted(base, ids)
    return [_rollout_summary(base, rid, not_counted=reasons.get(rid)) for rid in ids]


def _rollout_summary(
    base: Path, rel_id: str, *, not_counted: NotCounted | None = None
) -> dict[str, Any]:
    """Catalog row for one rollout: identity, verdict, row-level stats.

    ``not_counted`` comes from :func:`_not_counted` over the whole catalog,
    since whether an attempt is final depends on its siblings.
    """
    d = base / rel_id
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
        not_counted=not_counted,
    ).to_payload()
