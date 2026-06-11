"""Report generation and decision helpers for acceptance-live validation.

This module owns the acceptance-live report plane: writing the declared report
(and its ``.sha256`` sidecar), summarizing runs, deriving leaderboard
suitability, and the reward/flake expectation checks that gate a case. None of
these functions touch the patched rollout seams, so they extract cleanly while
the orchestration plane stays in the ``benchflow.task.acceptance_live`` façade.
The façade re-exports every name defined here.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from benchflow._utils.scoring import contains_verifier_dep_install_marker
from benchflow.task.acceptance_live_model import (
    _DEP_INSTALL_FLAKE_HINT,
    _LEADERBOARD_CALIBRATION_TYPES,
    LiveAcceptanceCase,
    LiveAcceptanceExpectation,
    LiveAcceptanceRunResult,
    LiveAcceptanceSpec,
)
from benchflow.task.task import Task


def _write_live_acceptance_report(
    task_dir: Path,
    *,
    sandbox_type: str,
    spec: LiveAcceptanceSpec,
    records: list[dict[str, Any]],
    staged_worktree: Path | None,
    leaderboard_suitability: dict[str, Any],
) -> None:
    if spec.report_path is None:
        return
    report_path = _live_report_output_path(task_dir, spec.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "acceptance-live-report",
        "schema_version": "1.0",
        "benchflow_version": _benchflow_version(),
        "generated_at": datetime.now(UTC).isoformat(),
        "sandbox": sandbox_type,
        "task": {
            "path": task_dir.name,
            "task_md_sha256": _file_sha256(task_dir / "task.md"),
            "oracle_sha256": _file_sha256(Task(task_dir).paths.solve_path),
            "verifier_sha256": _file_sha256(Task(task_dir).paths.test_path),
        },
        "workspace": {
            "source": spec.workspace.source,
            "target": spec.workspace.target,
            "staged_tree_sha256": (
                _tree_sha256(staged_worktree) if staged_worktree is not None else None
            ),
        },
        "spec_sha256": _spec_sha256(spec),
        "cases": [
            {
                "name": case.name,
                "type": case.case_type,
                "source": case.source,
                "command": case.command,
                "reruns": case.reruns,
                "expect": _expectation_dict(case.expect),
            }
            for case in spec.cases
        ],
        "case_summaries": [
            _case_summary(case=case, records=records) for case in spec.cases
        ],
        "leaderboard_suitability": leaderboard_suitability,
        "summary": _report_summary(records),
        "runs": records,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    digest = sha256(report_path.read_bytes()).hexdigest()
    sidecar = report_path.with_suffix(report_path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {spec.report_path.as_posix()}\n")


def _live_report_output_path(task_dir: Path, report_path: Path) -> Path:
    if report_path.is_absolute():
        return report_path
    return task_dir / report_path


def _benchflow_version() -> str:
    with contextlib.suppress(Exception):
        from benchflow import __version__

        return __version__
    return "0+unknown"


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    entries: list[str] = []
    for path in sorted(child for child in root.rglob("*") if child.is_file()):
        rel = path.relative_to(root).as_posix()
        entries.append(f"{rel}\0{sha256(path.read_bytes()).hexdigest()}")
    return sha256("\n".join(entries).encode()).hexdigest()


def _spec_sha256(spec: LiveAcceptanceSpec) -> str:
    payload = {
        "workspace": {
            "source": spec.workspace.source,
            "target": spec.workspace.target,
        },
        "cases": [
            {
                "name": case.name,
                "type": case.case_type,
                "source": case.source,
                "command": case.command,
                "reruns": case.reruns,
                "expect": _expectation_dict(case.expect),
            }
            for case in spec.cases
        ],
        "leaderboard": {
            "required": spec.leaderboard.required,
            "max_flake_rate": spec.leaderboard.max_flake_rate,
        },
        "report": spec.report_path.as_posix() if spec.report_path else None,
    }
    return _canonical_sha256(payload)


def _run_record(
    *,
    case: LiveAcceptanceCase,
    run_index: int,
    result: LiveAcceptanceRunResult,
    expectation_issues: list[str],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case": case.name,
        "type": case.case_type,
        "source": case.source,
        "run_index": run_index,
        "reward": result.reward,
        "status": "failed"
        if result.error is not None or result.reward is None or expectation_issues
        else "passed",
        "error": result.error,
        "verifier_error_category": result.verifier_error_category,
        "diagnostic_code": result.diagnostic_code,
        "artifact_hint": result.artifact_hint,
        "expectation_issues": expectation_issues,
    }
    record["sha256"] = _canonical_sha256(record)
    return record


def _report_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    failed = sum(1 for record in records if record.get("status") != "passed")
    rewards = [
        float(record["reward"])
        for record in records
        if isinstance(record.get("reward"), int | float)
        and not isinstance(record.get("reward"), bool)
    ]
    return {
        "total_runs": total,
        "passed_runs": total - failed,
        "failed_runs": failed,
        "flake_rate": (failed / total) if total else 0.0,
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
    }


def _leaderboard_suitability(
    *,
    spec: LiveAcceptanceSpec,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _report_summary(records)
    total_runs = int(summary["total_runs"])
    failed_runs = int(summary["failed_runs"])
    flake_rate = float(summary["flake_rate"])
    passed_records = [record for record in records if record.get("status") == "passed"]
    generated_types = {
        str(record.get("type"))
        for record in passed_records
        if record.get("source") == "calibration-report"
        and record.get("type") in _LEADERBOARD_CALIBRATION_TYPES
    }
    missing_generated_types = sorted(_LEADERBOARD_CALIBRATION_TYPES - generated_types)
    has_oracle = any(record.get("type") == "oracle" for record in passed_records)
    has_reference = any(record.get("type") == "reference" for record in passed_records)
    checks = {
        "has_live_runs": total_runs > 0,
        "all_runs_passed": total_runs > 0 and failed_runs == 0,
        "flake_rate_within_limit": flake_rate <= spec.leaderboard.max_flake_rate + 1e-9,
        "has_oracle_proof": has_oracle,
        "has_reference_proof": has_reference,
        "has_generated_calibration_coverage": not missing_generated_types,
    }
    issues: list[str] = []
    if not checks["has_live_runs"]:
        issues.append("requires at least one live run")
    if not checks["all_runs_passed"]:
        issues.append("requires all live runs to pass")
    if not checks["flake_rate_within_limit"]:
        issues.append(
            f"flake_rate {flake_rate:.6g} exceeds max_flake_rate "
            f"{spec.leaderboard.max_flake_rate:.6g}"
        )
    if not has_oracle:
        issues.append("requires a passed oracle live case")
    if not has_reference:
        issues.append("requires a passed reference live case")
    if missing_generated_types:
        issues.append(
            "missing generated calibration live case types: "
            + ", ".join(missing_generated_types)
        )
    return {
        "status": "suitable" if not issues else "insufficient",
        "required": spec.leaderboard.required,
        "max_flake_rate": spec.leaderboard.max_flake_rate,
        "required_generated_calibration_types": sorted(_LEADERBOARD_CALIBRATION_TYPES),
        "observed_generated_calibration_types": sorted(generated_types),
        "checks": checks,
        "issues": issues,
    }


def _expectation_dict(expect: LiveAcceptanceExpectation) -> dict[str, Any]:
    return {
        "reward_min": expect.reward_min,
        "reward_max": expect.reward_max,
        "reward_range": list(expect.reward_range) if expect.reward_range else None,
        "reward_equals": expect.reward_equals,
        "flake_rate_max": expect.flake_rate_max,
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode()).hexdigest()


def _case_summary(
    *,
    case: LiveAcceptanceCase,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    case_records = [record for record in records if record.get("case") == case.name]
    total = len(case_records)
    failed = sum(1 for record in case_records if record.get("status") != "passed")
    rewards = [
        float(record["reward"])
        for record in case_records
        if isinstance(record.get("reward"), int | float)
        and not isinstance(record.get("reward"), bool)
    ]
    flake_rate = (failed / total) if total else 0.0
    threshold = case.expect.flake_rate_max
    status = (
        "passed"
        if (threshold is not None and flake_rate <= threshold)
        or (threshold is None and failed == 0)
        else "failed"
    )
    return {
        "case": case.name,
        "type": case.case_type,
        "source": case.source,
        "total_runs": total,
        "passed_runs": total - failed,
        "failed_runs": failed,
        "flake_rate": flake_rate,
        "flake_rate_max": threshold,
        "min_reward": min(rewards) if rewards else None,
        "max_reward": max(rewards) if rewards else None,
        "status": status,
    }


def _check_case_flake_expectation(
    *,
    case: LiveAcceptanceCase,
    records: list[dict[str, Any]],
) -> list[str]:
    threshold = case.expect.flake_rate_max
    if threshold is None:
        return []
    summary = _case_summary(case=case, records=records)
    flake_rate = summary["flake_rate"]
    if not isinstance(flake_rate, int | float):
        return [f"acceptance-live case {case.name!r} did not produce flake rate"]
    if float(flake_rate) - 1e-9 > threshold:
        issue = (
            f"acceptance-live case {case.name!r} flake_rate "
            f"{float(flake_rate):.6g} exceeds flake_rate_max {threshold:.6g}"
        )
        hint = _case_failure_hint(records)
        if hint:
            issue += f"; {hint}"
        return [issue]
    return []


def _case_failure_hint(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        if record.get("status") == "passed":
            continue
        error = record.get("error")
        if isinstance(error, str) and contains_verifier_dep_install_marker(error):
            return _DEP_INSTALL_FLAKE_HINT
    return None


def _check_reward_expectation(
    prefix: str,
    reward: float,
    expect: LiveAcceptanceExpectation,
) -> list[str]:
    issues: list[str] = []
    if expect.reward_min is not None and reward + 1e-9 < expect.reward_min:
        issues.append(
            f"{prefix} reward {reward:.6g} is below reward_min {expect.reward_min:.6g}"
        )
    if expect.reward_max is not None and reward - 1e-9 > expect.reward_max:
        issues.append(
            f"{prefix} reward {reward:.6g} is above reward_max {expect.reward_max:.6g}"
        )
    if expect.reward_range is not None:
        low, high = expect.reward_range
        if reward + 1e-9 < low or reward - 1e-9 > high:
            issues.append(
                f"{prefix} reward {reward:.6g} is outside reward_range "
                f"[{low:.6g}, {high:.6g}]"
            )
    if expect.reward_equals is not None and abs(reward - expect.reward_equals) > 1e-9:
        issues.append(
            f"{prefix} reward {reward:.6g} does not equal reward_equals "
            f"{expect.reward_equals:.6g}"
        )
    return issues
