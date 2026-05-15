"""Normalize BenchFlow and historical SkillsBench/Harbor result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reward_from_mapping(rewards: Any) -> float | int | None:
    if isinstance(rewards, dict):
        value = rewards.get("reward")
        if isinstance(value, int | float):
            return value
    return None


def normalize_result(
    data: dict[str, Any], *, path: str | None = None
) -> dict[str, Any]:
    """Normalize current BenchFlow and older SkillsBench trajectory result schemas."""
    # Current BenchFlow SDK._build_result schema.
    if "rewards" in data or "n_tool_calls" in data:
        return {
            "schema": "benchflow",
            "path": path,
            "task_name": data.get("task_name"),
            "trial_name": data.get("trial_name"),
            "agent": data.get("agent"),
            "agent_name": data.get("agent_name"),
            "model": data.get("model"),
            "environment": None,
            "reward": _reward_from_mapping(data.get("rewards")),
            "error": data.get("error"),
            "verifier_error": data.get("verifier_error"),
            "n_tool_calls": data.get("n_tool_calls"),
            "n_input_tokens": data.get("n_input_tokens"),
            "n_output_tokens": data.get("n_output_tokens"),
            "n_cache_tokens": data.get("n_cache_tokens"),
            "cost_usd": data.get("cost_usd"),
            "trajectory_source": data.get("trajectory_source"),
            "partial_trajectory": data.get("partial_trajectory"),
            "timing": data.get("timing") or {},
        }

    # Historical SkillsBench / Harbor-like result schema.
    cfg = _as_dict(data.get("config"))
    agent_cfg = _as_dict(cfg.get("agent"))
    env_cfg = _as_dict(cfg.get("environment"))
    agent_result = _as_dict(data.get("agent_result"))
    verifier_result = _as_dict(data.get("verifier_result"))
    exception_info = data.get("exception_info")
    error = None
    if exception_info:
        if isinstance(exception_info, dict):
            error = exception_info.get("message") or exception_info.get("type")
        else:
            error = str(exception_info)

    return {
        "schema": "skillsbench-historical",
        "path": path,
        "task_name": data.get("task_name"),
        "trial_name": data.get("trial_name"),
        "agent": agent_cfg.get("name"),
        "agent_name": (data.get("agent_info") or {}).get("name")
        if isinstance(data.get("agent_info"), dict)
        else agent_cfg.get("name"),
        "model": agent_cfg.get("model_name"),
        "environment": env_cfg.get("type"),
        "reward": _reward_from_mapping(verifier_result.get("rewards")),
        "error": error,
        "verifier_error": None,
        "n_tool_calls": agent_result.get("n_tool_calls"),
        "n_input_tokens": agent_result.get("n_input_tokens"),
        "n_output_tokens": agent_result.get("n_output_tokens"),
        "n_cache_tokens": agent_result.get("n_cache_tokens"),
        "cost_usd": agent_result.get("cost_usd"),
        "trajectory_source": None,
        "partial_trajectory": None,
        "timing": {
            phase: data.get(phase)
            for phase in (
                "environment_setup",
                "agent_setup",
                "agent_execution",
                "verifier",
            )
            if data.get(phase) is not None
        },
    }


def collect_normalized_results(root: str | Path) -> list[dict[str, Any]]:
    """Collect and normalize every result.json under *root*."""
    root = Path(root)
    records = []
    for path in sorted(root.rglob("result.json")):
        try:
            records.append(
                normalize_result(json.loads(path.read_text()), path=str(path))
            )
        except Exception as exc:
            records.append(
                {
                    "schema": "invalid",
                    "path": str(path),
                    "task_name": None,
                    "error": f"invalid result.json: {exc}",
                }
            )
    return records


def _by_task(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task = record.get("task_name")
        if task:
            out.setdefault(task, []).append(record)
    return out


def build_parity_report(
    run_dir: str | Path,
    baseline_dir: str | Path | None = None,
    baseline_error: str | None = None,
) -> dict[str, Any]:
    """Build a broad parity report for current results and optional baselines."""
    run_dir = Path(run_dir)
    current = collect_normalized_results(run_dir)
    baseline = collect_normalized_results(baseline_dir) if baseline_dir else []
    current_by_task = _by_task(current)
    baseline_by_task = _by_task(baseline)

    task_rows = []
    for task in sorted(set(current_by_task) | set(baseline_by_task)):
        cur = current_by_task.get(task, [])
        base = baseline_by_task.get(task, [])
        task_rows.append(
            {
                "task_name": task,
                "current_trials": len(cur),
                "baseline_trials": len(base),
                "current_rewards": [r.get("reward") for r in cur],
                "baseline_rewards": [r.get("reward") for r in base],
                "missing_baseline": not bool(base),
            }
        )

    token_fields_present = any(
        r.get("n_input_tokens") is not None or r.get("n_output_tokens") is not None
        for r in current
    )
    baseline_token_fields_present = any(
        r.get("n_input_tokens") is not None or r.get("n_output_tokens") is not None
        for r in baseline
    )

    return {
        "run_dir": str(run_dir),
        "baseline_dir": str(baseline_dir) if baseline_dir else None,
        "baseline_error": baseline_error,
        "current_count": len(current),
        "baseline_count": len(baseline),
        "token_fields_present": token_fields_present,
        "baseline_token_fields_present": baseline_token_fields_present,
        "tasks": task_rows,
    }


def write_parity_report(
    run_dir: str | Path,
    baseline_dir: str | Path | None = None,
    baseline_error: str | None = None,
) -> dict[str, Any]:
    """Write ``parity_report.json`` into *run_dir* and return the report."""
    run_dir = Path(run_dir)
    report = build_parity_report(run_dir, baseline_dir, baseline_error)
    (run_dir / "parity_report.json").write_text(json.dumps(report, indent=2))
    current = collect_normalized_results(run_dir)
    (run_dir / "normalized_results.jsonl").write_text(
        "".join(json.dumps(record, default=str) + "\n" for record in current)
    )
    if baseline_dir:
        baseline = collect_normalized_results(baseline_dir)
        (run_dir / "normalized_baseline.jsonl").write_text(
            "".join(json.dumps(record, default=str) + "\n" for record in baseline)
        )
    return report
