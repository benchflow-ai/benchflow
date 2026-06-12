"""Conventional acceptance-evidence discovery for ``task.md`` documents.

This filesystem/JSON subsystem inspects ``<task_dir>/evidence/acceptance/`` for
oracle, verifier-stability, review, calibration, and acceptance-live reports and
folds the discovered evidence into normalized frontmatter. It owns its own
``_read_json`` / ``_first_number`` / ``_max_case_reward`` helpers and reuses the
shared dict-merge helpers from the normalization layer.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from benchflow.task._document_normalize import (
    _ensure_mapping,
    _mapping,
    _merge_missing,
)


def _apply_conventional_evidence(
    normalized: dict[str, Any],
    *,
    task_dir: Path,
    profiles: list[str],
) -> None:
    if not {"acceptance-live", "leaderboard-local"}.intersection(profiles):
        return
    discovered = _discover_conventional_evidence(task_dir)
    if not discovered:
        return
    benchflow = _ensure_mapping(normalized, "benchflow")
    existing = _mapping(benchflow.get("evidence"), "benchflow.evidence", default={})
    benchflow["evidence"] = _merge_missing(existing, discovered)


def _discover_conventional_evidence(task_dir: Path) -> dict[str, Any]:
    root = task_dir / "evidence" / "acceptance"
    if not root.is_dir():
        return {}

    evidence: dict[str, Any] = {}
    artifact_paths: set[str] = set()
    trajectory_paths: set[str] = set()

    oracle = _read_json(root / "oracle-run.json")
    if isinstance(oracle, dict):
        oracle_map = cast(dict[str, Any], oracle)
        required_reward = _first_number(
            oracle_map.get("required_reward"),
            oracle_map.get("expected_reward"),
            oracle_map.get("reward"),
        )
        evidence["oracle_runs"] = {
            "required_reward": required_reward if required_reward is not None else 1.0,
            "artifact": "evidence/acceptance/oracle-run.json",
        }
        artifact_paths.add("evidence/acceptance/oracle-run.json")

    verifier = _read_json(root / "verifier-stability-report.json")
    if isinstance(verifier, dict):
        verifier_map = cast(dict[str, Any], verifier)
        evidence["verifier"] = {
            "reruns": verifier_map.get("reruns", 3),
            "flake_rate": verifier_map.get("flake_rate", 0.0),
            "report": "evidence/acceptance/verifier-stability-report.json",
        }
        artifact_paths.add("evidence/acceptance/verifier-stability-report.json")

    review = _read_json(root / "review.json")
    if isinstance(review, dict):
        review_map = cast(dict[str, Any], review)
        review_evidence = {
            "anti_cheat": review_map.get("anti_cheat", "passed"),
            "instruction_alignment": review_map.get("instruction_alignment", "passed"),
            "artifact": "evidence/acceptance/review.json",
        }
        if isinstance(review_map.get("reviewer"), str):
            review_evidence["reviewer"] = review_map["reviewer"]
        evidence["review"] = review_evidence
        artifact_paths.add("evidence/acceptance/review.json")

    calibration = _read_json(root / "calibration-report.json")
    if isinstance(calibration, dict):
        calibration_map = cast(dict[str, Any], calibration)
        calibration_evidence = _calibration_evidence_from_report(
            calibration_map,
            root=root,
        )
        if calibration_evidence:
            evidence["calibration"] = calibration_evidence
            artifact_paths.add("evidence/acceptance/calibration-report.json")
            gold_artifact = calibration_evidence.get("human_or_reference_examples", [])
            for example in gold_artifact:
                if isinstance(example, dict) and isinstance(
                    example.get("artifact"), str
                ):
                    artifact_paths.add(example["artifact"])

    live_report = root / "live-report.json"
    if live_report.is_file():
        acceptance_live = _acceptance_live_evidence_from_report(live_report)
        if acceptance_live:
            evidence["acceptance_live"] = acceptance_live
            artifact_paths.add("evidence/acceptance/live-report.json")

    gold_trajectory = root / "gold-trajectory.jsonl"
    if gold_trajectory.is_file():
        trajectory_paths.add("evidence/acceptance/gold-trajectory.jsonl")

    artifacts = _pin_existing_files(task_dir, artifact_paths)
    trajectories = _pin_existing_files(task_dir, trajectory_paths)
    if artifacts:
        evidence["artifacts"] = artifacts
    if trajectories:
        evidence["trajectories"] = trajectories
    return evidence


def _calibration_evidence_from_report(
    report: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        return {}
    by_type: dict[str, list[dict[str, Any]]] = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            continue
        case_type = raw_case.get("type")
        if isinstance(case_type, str):
            by_type.setdefault(case_type, []).append(raw_case)

    no_op = _max_case_reward(by_type.get("no-op", []))
    known_bad = _max_case_reward(by_type.get("known-bad", []))
    partial_rewards = [
        reward
        for case in by_type.get("partial", [])
        if (reward := _first_number(case.get("reward"))) is not None
    ]
    reference_examples = []
    gold_result = root / "gold-result.json"
    for case in by_type.get("reference", []):
        reward = _first_number(case.get("reward"))
        if reward is None:
            continue
        example = {
            "name": str(case.get("name") or "reference"),
            "expected_reward": reward,
        }
        if gold_result.is_file():
            example["artifact"] = "evidence/acceptance/gold-result.json"
        reference_examples.append(example)

    if no_op is None or known_bad is None or not partial_rewards:
        return {}
    return {
        "no_op_reward_max": no_op,
        "known_bad_reward_max": known_bad,
        "partial_solution_range": [min(partial_rewards), max(partial_rewards)],
        "report": "evidence/acceptance/calibration-report.json",
        "human_or_reference_examples": reference_examples,
    }


def _acceptance_live_evidence_from_report(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    evidence: dict[str, Any] = {
        "report": "evidence/acceptance/live-report.json",
        "workspace": {
            "source": "current-worktree",
            "target": "/repo",
        },
        "calibration": {
            "from": "calibration.report",
            "reruns": 1,
            "flake_rate_max": 0.0,
        },
    }
    if not isinstance(report, dict):
        return evidence
    report_map = cast(dict[str, Any], report)

    cases = []
    raw_cases = report_map.get("cases", [])
    if not isinstance(raw_cases, list):
        raw_cases = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict) or raw_case.get("source") != "declared":
            continue
        case_map = cast(dict[str, Any], raw_case)
        case = {
            "name": case_map.get("name"),
            "type": case_map.get("type"),
            "reruns": case_map.get("reruns", 1),
            "expect": case_map.get("expect", {}),
        }
        if isinstance(case["name"], str) and isinstance(case["type"], str):
            cases.append(case)
    if cases:
        evidence["cases"] = cases
    leaderboard = report_map.get("leaderboard_suitability")
    if isinstance(leaderboard, dict):
        evidence["leaderboard"] = {
            "required": True,
            "max_flake_rate": 0.0,
        }
    return evidence


def _pin_existing_files(task_dir: Path, paths: set[str]) -> list[dict[str, str]]:
    pinned = []
    for rel in sorted(paths):
        path = task_dir / rel
        if path.is_file():
            pinned.append(
                {"path": rel, "sha256": sha256(path.read_bytes()).hexdigest()}
            )
    return pinned


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _first_number(*values: object) -> float | None:
    for value in values:
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return None


def _max_case_reward(cases: list[dict[str, Any]]) -> float | None:
    rewards = [
        reward
        for case in cases
        if (reward := _first_number(case.get("reward"))) is not None
    ]
    return max(rewards) if rewards else None
