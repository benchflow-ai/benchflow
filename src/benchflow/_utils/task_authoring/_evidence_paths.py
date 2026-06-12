"""Leaf helpers shared by the task-authoring structural and acceptance gates.

Pure stdlib, no ``benchflow`` imports — so both the structural-check and
acceptance-evidence clusters can depend on it without introducing an import
cycle with the package façade.
"""

import json
from hashlib import sha256
from pathlib import Path
from typing import cast

TASK_DOCUMENT_FILE = "task.md"

# Placeholder marker written by init_task — must be replaced before the task
# is considered authored. Catching this in check_task prevents a freshly
# scaffolded task from being mistaken for a real benchmark (#360).
_PLACEHOLDER_MARKER = "[REPLACE:"


def _declared_evidence_path_key(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = _safe_relative_path(value)
    return path.as_posix() if path is not None else None


def _check_declared_evidence_file(
    value: object,
    *,
    task_dir: Path,
    source: str,
    expected_sha256: object | None = None,
) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{source} must be a non-empty relative path"]
    path = _safe_relative_path(value)
    if path is None:
        return [f"{source} must be a safe relative path"]
    local_path = task_dir / path
    if not local_path.is_file():
        return [f"{source} references missing file: {value}"]
    if expected_sha256 is None:
        return []
    if not isinstance(expected_sha256, str) or not expected_sha256.strip():
        return [f"{source} sha256 must be a non-empty string when declared"]
    digest = sha256(local_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        return [f"{source} sha256 mismatch for {value}"]
    return []


def _load_declared_evidence_json(
    value: object,
    *,
    task_dir: Path,
    source: str,
) -> tuple[object | None, list[str]]:
    issues = _check_declared_evidence_file(value, task_dir=task_dir, source=source)
    if issues:
        return None, issues
    assert isinstance(value, str)
    path = _safe_relative_path(value)
    assert path is not None
    try:
        return json.loads((task_dir / path).read_text()), []
    except json.JSONDecodeError as exc:
        return None, [f"{source} is not valid JSON: {exc}"]
    except OSError as exc:
        return None, [f"{source} cannot be read: {exc}"]


def _has_regular_file(root: Path) -> bool:
    return any(path.is_file() for path in root.rglob("*"))


def _number_value(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _safe_relative_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _check_primary_evidence_pins(evidence: dict[str, object]) -> list[str]:
    pinned_paths = _pinned_evidence_paths(evidence)
    required = _primary_evidence_paths(evidence)
    return [
        f"{source} must be listed in benchflow.evidence.artifacts or "
        "benchflow.evidence.trajectories with sha256"
        for source, path in required
        if path not in pinned_paths
    ]


def _pinned_evidence_paths(evidence: dict[str, object]) -> set[str]:
    pinned_paths: set[str] = set()
    for list_key in ("trajectories", "artifacts"):
        items = evidence.get(list_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            item_mapping = cast(dict[str, object], item)
            if (
                not isinstance(item_mapping.get("sha256"), str)
                or not str(item_mapping.get("sha256")).strip()
            ):
                continue
            path = _declared_evidence_path_key(item_mapping.get("path"))
            if path is not None:
                pinned_paths.add(path)
    return pinned_paths


def _primary_evidence_paths(evidence: dict[str, object]) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []

    oracle_runs = evidence.get("oracle_runs")
    if isinstance(oracle_runs, dict):
        oracle_mapping = cast(dict[str, object], oracle_runs)
        _append_declared_evidence_path(
            paths,
            "acceptance oracle_runs.artifact",
            oracle_mapping.get("artifact"),
        )

    verifier = evidence.get("verifier")
    if isinstance(verifier, dict):
        verifier_mapping = cast(dict[str, object], verifier)
        _append_declared_evidence_path(
            paths,
            "acceptance verifier.report",
            verifier_mapping.get("report"),
        )

    review = evidence.get("review")
    if isinstance(review, dict):
        review_mapping = cast(dict[str, object], review)
        _append_declared_evidence_path(
            paths,
            "acceptance review.artifact",
            review_mapping.get("artifact"),
        )

    calibration = evidence.get("calibration")
    if isinstance(calibration, dict):
        calibration_mapping = cast(dict[str, object], calibration)
        _append_declared_evidence_path(
            paths,
            "acceptance calibration.report",
            calibration_mapping.get("report"),
        )
        examples = calibration_mapping.get("human_or_reference_examples")
        if isinstance(examples, list):
            for index, example in enumerate(examples):
                if not isinstance(example, dict):
                    continue
                example_mapping = cast(dict[str, object], example)
                _append_declared_evidence_path(
                    paths,
                    (
                        "acceptance calibration.human_or_reference_examples"
                        f"[{index}].artifact"
                    ),
                    example_mapping.get("artifact"),
                )

    return paths


def _append_declared_evidence_path(
    paths: list[tuple[str, str]],
    source: str,
    value: object,
) -> None:
    path = _declared_evidence_path_key(value)
    if path is not None:
        paths.append((source, path))
