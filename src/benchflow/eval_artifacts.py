"""Evaluation artifact helpers for reproducible training/eval workflows."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from benchflow._utils.task_authoring import task_digest
from benchflow.trajectories.export_prime_sft import (
    PrimeSftTrajectoryJsonlError,
    load_llm_trajectory_jsonl,
)

CanonicalizePolicy = Literal["none", "one-healthy-per-task"]
RetryPolicy = Literal["default", "unscored-only"]


@dataclass(frozen=True)
class TaskManifestOptions:
    tasks_dir: Path
    include_tasks: set[str]
    exclude_tasks: set[str]
    source_provenance: dict[str, Any] | None = None
    dataset_name: str | None = None
    dataset_version: str | None = None
    dataset_task_digests: dict[str, str] | None = None


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_task_dir(path: Path) -> bool:
    return (path / "task.md").is_file() or (path / "task.toml").is_file()


def _selected_task_dirs(
    tasks_dir: Path, include_tasks: set[str], exclude_tasks: set[str]
) -> list[Path]:
    if _is_task_dir(tasks_dir):
        candidates = [tasks_dir]
    else:
        candidates = sorted(path for path in tasks_dir.iterdir() if path.is_dir())
        candidates = [path for path in candidates if _is_task_dir(path)]
    selected = []
    for task_dir in candidates:
        name = task_dir.name
        if include_tasks and name not in include_tasks:
            continue
        if name in exclude_tasks:
            continue
        selected.append(task_dir)
    return selected


def build_task_manifest(options: TaskManifestOptions) -> dict[str, Any]:
    tasks = []
    dataset_digests = options.dataset_task_digests or {}
    for task_dir in _selected_task_dirs(
        options.tasks_dir, options.include_tasks, options.exclude_tasks
    ):
        digest = task_digest(task_dir)
        entry: dict[str, Any] = {
            "task_id": task_dir.name,
            "path": str(task_dir),
            "digest": digest,
        }
        registry_digest = dataset_digests.get(task_dir.name)
        if registry_digest is not None:
            entry["registry_digest"] = registry_digest
            entry["registry_digest_match"] = registry_digest == digest
        tasks.append(entry)
    return {
        "schema_version": 1,
        "tasks_dir": str(options.tasks_dir),
        "source": options.source_provenance,
        "dataset_name": options.dataset_name,
        "dataset_version": options.dataset_version,
        "total": len(tasks),
        "tasks": tasks,
    }


def write_task_manifest(path: Path, options: TaskManifestOptions) -> dict[str, Any]:
    manifest = build_task_manifest(options)
    _json_dump(path, manifest)
    return manifest


def _iter_rollouts(job_dir: Path) -> list[Path]:
    if (job_dir / "result.json").is_file():
        return [job_dir]
    return sorted({path.parent for path in job_dir.rglob("result.json")})


def _rollout_dir_from_selection_row(
    row: dict[str, Any], *, job_dir: Path, selection_path: Path
) -> Path:
    rollout_dir = Path(str(row.get("rollout_dir") or ""))
    if not rollout_dir.is_dir() and not rollout_dir.is_absolute():
        rollout_dir = job_dir / rollout_dir
    if not rollout_dir.is_dir() and isinstance(row.get("result_json"), str):
        result_json = Path(row["result_json"])
        if result_json.is_absolute() and not result_json.is_file():
            marker = f"/{selection_path.parent.name}/"
            _, sep, suffix = str(result_json).partition(marker)
            if sep:
                result_json = selection_path.parent / suffix
        rollout_dir = result_json.parent
    return rollout_dir


def _iter_selected_rollouts(selection_path: Path) -> list[Path]:
    selection = _read_json(selection_path)
    if selection is None:
        raise ValueError(f"invalid canonical selection JSON: {selection_path}")
    job_dir = Path(str(selection.get("job_dir") or ""))
    selected = selection.get("selected", selection.get("selection"))
    if not isinstance(selected, list):
        raise ValueError(f"{selection_path}: selected or selection must be a list")
    rollouts = []
    for row in selected:
        if not isinstance(row, dict):
            continue
        rollout_dir = _rollout_dir_from_selection_row(
            row, job_dir=job_dir, selection_path=selection_path
        )
        if not rollout_dir.is_dir():
            raise ValueError(f"selected rollout dir not found: {rollout_dir}")
        rollouts.append(rollout_dir)
    return rollouts


def _reward(result: dict[str, Any]) -> float | None:
    rewards = result.get("rewards")
    if isinstance(rewards, dict):
        value = rewards.get("reward")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    value = result.get("reward")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _tool_call_count(result: dict[str, Any]) -> int:
    value = result.get("n_tool_calls")
    if isinstance(value, int) and value >= 0:
        return value
    agent_result = result.get("agent_result")
    if isinstance(agent_result, dict):
        value = agent_result.get("n_tool_calls")
        if isinstance(value, int) and value >= 0:
            return value
    summary = result.get("trajectory_summary")
    if isinstance(summary, dict):
        value = summary.get("tool_call_steps")
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _llm_trajectory_status(rollout_dir: Path) -> tuple[bool, bool, int]:
    path = rollout_dir / "trajectory" / "llm_trajectory.jsonl"
    if not path.is_file():
        return False, False, 0
    try:
        rows = load_llm_trajectory_jsonl(path, strict=True)
    except PrimeSftTrajectoryJsonlError:
        return True, False, 0
    return True, True, len(rows)


def build_health_summary(
    job_dir: Path, *, canonical_selection: Path | None = None
) -> dict[str, Any]:
    rows = []
    counts = {
        "total_rows": 0,
        "scored_rows": 0,
        "unscored_rows": 0,
        "rows_with_tool_calls": 0,
        "zero_tool_rows": 0,
        "missing_llm_trajectory": 0,
        "malformed_llm_trajectory": 0,
    }
    rollout_dirs = (
        _iter_selected_rollouts(canonical_selection)
        if canonical_selection is not None
        else _iter_rollouts(job_dir)
    )
    for rollout_dir in rollout_dirs:
        result = _read_json(rollout_dir / "result.json")
        if result is None:
            continue
        counts["total_rows"] += 1
        reward = _reward(result)
        scored = reward is not None
        if scored:
            counts["scored_rows"] += 1
        else:
            counts["unscored_rows"] += 1
        tool_calls = _tool_call_count(result)
        if tool_calls > 0:
            counts["rows_with_tool_calls"] += 1
        else:
            counts["zero_tool_rows"] += 1
        has_llm, valid_llm, llm_rows = _llm_trajectory_status(rollout_dir)
        if not has_llm:
            counts["missing_llm_trajectory"] += 1
        elif not valid_llm:
            counts["malformed_llm_trajectory"] += 1
        rows.append(
            {
                "task_id": result.get("task_name") or rollout_dir.name,
                "rollout_dir": str(rollout_dir),
                "reward": reward,
                "scored": scored,
                "tool_calls": tool_calls,
                "has_llm_trajectory": has_llm,
                "valid_llm_trajectory": valid_llm,
                "llm_trajectory_rows": llm_rows,
                "error": result.get("error"),
                "verifier_error": result.get("verifier_error"),
                "error_category": result.get("error_category"),
                "verifier_error_category": result.get("verifier_error_category"),
            }
        )
    return {"schema_version": 1, "job_dir": str(job_dir), **counts, "rows": rows}


def write_health_summary(path: Path, job_dir: Path) -> dict[str, Any]:
    summary = build_health_summary(job_dir)
    _json_dump(path, summary)
    return summary


def _canonical_score(row: dict[str, Any]) -> tuple[int, float, int]:
    scored = 1 if row["scored"] else 0
    reward = row["reward"] if isinstance(row["reward"], float) else float("-inf")
    tool_calls = int(row.get("tool_calls") or 0)
    return scored, reward, tool_calls


def build_canonical_selection(
    job_dir: Path,
    *,
    policy: CanonicalizePolicy,
    expected_tasks: int | None = None,
) -> dict[str, Any]:
    if policy == "none":
        raise ValueError("canonical selection requires a non-none policy")
    if policy != "one-healthy-per-task":
        raise ValueError(f"unknown canonicalization policy: {policy}")

    health = build_health_summary(job_dir)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in health["rows"]:
        grouped.setdefault(str(row["task_id"]), []).append(row)
    selected = []
    rejected = []
    for task_id, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=_canonical_score, reverse=True)
        winner = ranked[0]
        selected.append({**winner, "selection_reason": "best-scored-then-reward"})
        rejected.extend({**row, "rejected_for_task_id": task_id} for row in ranked[1:])

    if expected_tasks is not None and len(selected) != expected_tasks:
        raise ValueError(
            f"canonical selected task count {len(selected)} != expected {expected_tasks}"
        )
    return {
        "schema_version": 1,
        "job_dir": str(job_dir),
        "policy": policy,
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "selected": selected,
        "rejected": rejected,
    }


def write_canonical_selection(
    path: Path,
    job_dir: Path,
    *,
    policy: CanonicalizePolicy,
    expected_tasks: int | None = None,
) -> dict[str, Any]:
    selection = build_canonical_selection(
        job_dir, policy=policy, expected_tasks=expected_tasks
    )
    _json_dump(path, selection)
    return selection


def materialize_canonical_job(selection_path: Path, output_dir: Path) -> None:
    selection = _read_json(selection_path)
    if selection is None:
        raise ValueError(f"invalid canonical selection JSON: {selection_path}")
    selected = selection.get("selected", selection.get("selection"))
    if not isinstance(selected, list):
        raise ValueError(f"{selection_path}: selected or selection must be a list")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for row in selected:
        if not isinstance(row, dict):
            continue
        rollout_dir = _rollout_dir_from_selection_row(
            row,
            job_dir=Path(str(selection.get("job_dir") or "")),
            selection_path=selection_path,
        )
        if not rollout_dir.is_dir():
            raise ValueError(f"selected rollout dir not found: {rollout_dir}")
        dest = output_dir / rollout_dir.name
        shutil.copytree(rollout_dir, dest)


def task_overlap(left_manifest: Path, right_manifest: Path) -> dict[str, Any]:
    left = _read_json(left_manifest)
    right = _read_json(right_manifest)
    if left is None:
        raise ValueError(f"invalid task manifest: {left_manifest}")
    if right is None:
        raise ValueError(f"invalid task manifest: {right_manifest}")
    left_tasks = {
        str(row["task_id"]): row
        for row in left.get("tasks", [])
        if isinstance(row, dict) and row.get("task_id")
    }
    right_tasks = {
        str(row["task_id"]): row
        for row in right.get("tasks", [])
        if isinstance(row, dict) and row.get("task_id")
    }
    task_id_overlap = sorted(set(left_tasks) & set(right_tasks))
    left_digest_to_task = {
        str(row.get("digest")): task_id
        for task_id, row in left_tasks.items()
        if row.get("digest")
    }
    right_digest_to_task = {
        str(row.get("digest")): task_id
        for task_id, row in right_tasks.items()
        if row.get("digest")
    }
    digest_overlap = sorted(set(left_digest_to_task) & set(right_digest_to_task))
    return {
        "schema_version": 1,
        "left": str(left_manifest),
        "right": str(right_manifest),
        "left_count": len(left_tasks),
        "right_count": len(right_tasks),
        "task_id_overlap_count": len(task_id_overlap),
        "task_id_overlap": task_id_overlap,
        "digest_overlap_count": len(digest_overlap),
        "digest_overlap": [
            {
                "digest": digest,
                "left_task_id": left_digest_to_task[digest],
                "right_task_id": right_digest_to_task[digest],
            }
            for digest in digest_overlap
        ],
        "caveat": (
            "Exact task-id/digest disjointness does not prove domain or "
            "generator-family disjointness."
        ),
    }
