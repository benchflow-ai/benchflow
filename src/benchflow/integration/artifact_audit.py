"""Deterministic artifact/schema audit for BenchFlow E2E runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_RESULT_FIELDS = {
    "task_name",
    "trial_name",
    "rewards",
    "agent",
    "agent_name",
    "model",
    "n_tool_calls",
    "n_prompts",
    "error",
    "verifier_error",
    "partial_trajectory",
    "trajectory_source",
    "started_at",
    "finished_at",
    "timing",
}

REQUIRED_SIBLINGS = ("config.json", "timing.json", "prompts.json")


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except Exception as exc:
        return None, str(exc)


def _jsonl_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "valid": False, "lines": 0, "error": "missing"}
    if path.stat().st_size == 0:
        return {"exists": True, "valid": True, "lines": 0, "error": None}
    lines = 0
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                json.loads(line)
                lines += 1
        return {"exists": True, "valid": True, "lines": lines, "error": None}
    except Exception as exc:
        return {"exists": True, "valid": False, "lines": lines, "error": str(exc)}


def audit_trial_result(result_path: str | Path) -> dict[str, Any]:
    """Audit one trial directory from its ``result.json`` path."""
    result_path = Path(result_path)
    trial_dir = result_path.parent
    issues: list[dict[str, str]] = []

    data, error = _read_json(result_path)
    if data is None:
        return {
            "trial_dir": str(trial_dir),
            "result_path": str(result_path),
            "task_name": None,
            "agent": None,
            "ok": False,
            "issues": [
                {"severity": "error", "message": f"invalid result.json: {error}"}
            ],
            "files": {},
        }

    missing_fields = sorted(REQUIRED_RESULT_FIELDS - set(data))
    for field in missing_fields:
        issues.append(
            {"severity": "error", "message": f"missing result field: {field}"}
        )

    files: dict[str, Any] = {}
    for rel in REQUIRED_SIBLINGS:
        exists = (trial_dir / rel).exists()
        files[rel] = {"exists": exists}
        if not exists:
            issues.append({"severity": "error", "message": f"missing file: {rel}"})

    trajectory_path = trial_dir / "trajectory" / "acp_trajectory.jsonl"
    files["trajectory/acp_trajectory.jsonl"] = _jsonl_status(trajectory_path)
    if not files["trajectory/acp_trajectory.jsonl"]["exists"]:
        issues.append(
            {
                "severity": "warning",
                "message": "missing file: trajectory/acp_trajectory.jsonl",
            }
        )
    elif not files["trajectory/acp_trajectory.jsonl"]["valid"]:
        issues.append(
            {
                "severity": "error",
                "message": "invalid JSONL: trajectory/acp_trajectory.jsonl",
            }
        )

    install_log = trial_dir / "agent" / "install-stdout.txt"
    files["agent/install-stdout.txt"] = {"exists": install_log.exists()}
    if data.get("agent") != "oracle" and not install_log.exists():
        issues.append(
            {"severity": "warning", "message": "missing file: agent/install-stdout.txt"}
        )

    timing = data.get("timing")
    if not isinstance(timing, dict):
        issues.append(
            {"severity": "error", "message": "result timing is not an object"}
        )
    elif "total" not in timing:
        issues.append({"severity": "warning", "message": "timing.total missing"})

    return {
        "trial_dir": str(trial_dir),
        "result_path": str(result_path),
        "task_name": data.get("task_name"),
        "agent": data.get("agent"),
        "model": data.get("model"),
        "ok": not any(i["severity"] == "error" for i in issues),
        "issues": issues,
        "files": files,
    }


def audit_run(run_dir: str | Path) -> dict[str, Any]:
    """Audit every ``result.json`` under an E2E run directory."""
    run_dir = Path(run_dir)
    trials = [
        audit_trial_result(path)
        for path in sorted(run_dir.rglob("result.json"))
        if path.name == "result.json"
    ]
    error_count = sum(
        1 for t in trials for issue in t["issues"] if issue.get("severity") == "error"
    )
    warning_count = sum(
        1 for t in trials for issue in t["issues"] if issue.get("severity") == "warning"
    )
    return {
        "run_dir": str(run_dir),
        "n_trials": len(trials),
        "error_count": error_count,
        "warning_count": warning_count,
        "ok": error_count == 0,
        "trials": trials,
    }


def write_artifact_audit(run_dir: str | Path) -> dict[str, Any]:
    """Write ``artifact_audit.json`` into *run_dir* and return the report."""
    run_dir = Path(run_dir)
    report = audit_run(run_dir)
    (run_dir / "artifact_audit.json").write_text(json.dumps(report, indent=2))
    return report
