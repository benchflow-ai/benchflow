"""Inbound recognition for iOSWorld benchmark sources.

iOSWorld is a native iOS Simulator benchmark. BenchFlow can recognize its
repository/task shapes now, but it cannot honestly translate them into a
runnable task until a macOS/iOS Simulator sandbox provider exists.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchflow.adapters.inbound import (
    InboundSupportReport,
    InboundTask,
    UnsupportedInboundTaskError,
)

IOSWORLD_TASK_FILE = "iosworld-task.json"

_TASK_ID_INVALID = re.compile(r"[^a-z0-9._-]+")
_UNSUPPORTED_REASON = "iOSWorld tasks require a macOS/iOS Simulator provider mapping"
_REQUIRED_PROVIDER = "macos-ios-simulator"
_REQUIRED_CAPABILITIES = [
    "macos",
    "xcode-26",
    "ios-26-simulator-runtime",
    "appium-xcuitest",
    "iosworld-app-bootstrap",
]


class IOSWorldAdapter:
    """Recognize iOSWorld sources and report provider-honest blockers."""

    source = "iosworld"

    @classmethod
    def is_task_dir(cls, task_dir: Path | str) -> bool:
        return cls.support_report(task_dir) is not None

    @classmethod
    def support_report(cls, task_dir: Path | str) -> InboundSupportReport | None:
        root = Path(task_dir)
        if (root / IOSWORLD_TASK_FILE).is_file():
            return _task_slice_report(root)
        if _looks_like_iosworld_repo(root):
            return _repo_report(root)
        return None

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        report = cls.support_report(task_dir)
        if report is None:
            root = Path(task_dir)
            raise FileNotFoundError(
                "iOSWorld task is missing iosworld-task.json or repository "
                f"signatures: {root}"
            )
        raise UnsupportedInboundTaskError(report)


def _task_slice_report(root: Path) -> InboundSupportReport:
    path = root / IOSWORLD_TASK_FILE
    try:
        raw = _load_json_object(path)
    except ValueError as exc:
        return InboundSupportReport(
            source=IOSWorldAdapter.source,
            supported=False,
            task_id=root.name,
            dataset="iosworld",
            reason=f"invalid {IOSWORLD_TASK_FILE}: {exc}",
            details={
                "issue": "invalid-iosworld-task-json",
                "required_provider": _REQUIRED_PROVIDER,
                "required_capabilities": _REQUIRED_CAPABILITIES,
            },
        )

    task_id = _task_id(raw, fallback=root.name)
    return InboundSupportReport(
        source=IOSWorldAdapter.source,
        supported=False,
        task_id=task_id,
        dataset="iosworld",
        reason=_UNSUPPORTED_REASON,
        details={
            "issue": "macos-ios-simulator-provider-required",
            "required_provider": _REQUIRED_PROVIDER,
            "shape": "task-slice",
            "apps": _string_list(raw.get("apps")),
            "category": raw.get("category"),
            "difficulty": raw.get("difficulty"),
            "rubric_count": _rubric_count(raw.get("rubric")),
            "required_capabilities": _REQUIRED_CAPABILITIES,
        },
    )


def _repo_report(root: Path) -> InboundSupportReport:
    tasks_path = root / "tasks.json"
    try:
        tasks = _load_json_list(tasks_path)
    except ValueError as exc:
        return InboundSupportReport(
            source=IOSWorldAdapter.source,
            supported=False,
            dataset="iosworld",
            reason=f"invalid iOSWorld tasks.json: {exc}",
            details={
                "issue": "invalid-iosworld-tasks-json",
                "shape": "repository",
                "required_provider": _REQUIRED_PROVIDER,
                "required_capabilities": _REQUIRED_CAPABILITIES,
            },
        )

    return InboundSupportReport(
        source=IOSWorldAdapter.source,
        supported=False,
        dataset="iosworld",
        reason=_UNSUPPORTED_REASON,
        details={
            "issue": "macos-ios-simulator-provider-required",
            "required_provider": _REQUIRED_PROVIDER,
            "shape": "repository",
            "task_count": len(tasks),
            "categories": _counts(
                str(task.get("category") or "unknown")
                for task in tasks
                if isinstance(task, dict)
            ),
            "apps": sorted(
                {
                    app
                    for task in tasks
                    if isinstance(task, dict)
                    for app in _string_list(task.get("apps"))
                }
            ),
            "required_capabilities": _REQUIRED_CAPABILITIES,
        },
    )


def _looks_like_iosworld_repo(root: Path) -> bool:
    return (
        (root / "tasks.json").is_file()
        and (root / "scripts" / "run_task_by_id.sh").is_file()
        and (root / "iphone" / "bootstrap" / "bootstrap_ios_apps.sh").is_file()
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object")
    return raw


def _load_json_list(path: Path) -> list[Any]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(raw, list):
        raise ValueError("expected a JSON array")
    return raw


def _task_id(raw: dict[str, Any], *, fallback: str) -> str:
    for key in ("name", "id", "task_id"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return _task_slug(value)
    return _task_slug(fallback)


def _task_slug(task_id: str) -> str:
    slug = _TASK_ID_INVALID.sub("-", task_id.strip().lower()).strip("-._")
    if not slug:
        slug = "task"
    if not slug[0].isalnum():
        slug = f"task-{slug}"
    return slug


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _rubric_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def from_iosworld_task(task_dir: Path | str) -> InboundTask:
    """Convenience function: translate an iOSWorld task when supported."""
    return IOSWorldAdapter.from_task_dir(task_dir)
