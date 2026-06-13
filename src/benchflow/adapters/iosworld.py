"""Inbound recognition + translation for iOSWorld benchmark sources.

iOSWorld is a native iOS Simulator benchmark: each task drives one or more of
26 SwiftUI apps and is scored by an LLM judge against a list of rubric
criteria. BenchFlow runs it on the ``macos-ios-simulator`` sandbox provider
(``benchflow.sandbox.macos_ios_simulator``), which boots a real iOS Simulator
device via ``xcrun simctl``.

This adapter is provider-honest in both directions:

* When the host advertises the iOS-Simulator capabilities the provider needs
  (macOS + Xcode 26 + an iOS runtime + Appium/xcuitest), it recognizes an
  ``iosworld-task.json`` slice and translates it into a native
  :class:`~benchflow.adapters.inbound.InboundTask` — the goal becomes the
  instruction and the rubric becomes an LLM-judge verifier, mirroring how the
  Browser Use / computer-use adapters map a rubric to a reward descriptor.
* When those prerequisites are missing (the common case off a Mac, and CI), it
  reports a structured *unsupported* result naming exactly which provider and
  capabilities are required, so ``tasks check`` stays honest about the blocker.

The host-capability check is delegated to
:func:`~benchflow.sandbox.macos_ios_simulator.detect_ios_simulator_capabilities`
so the adapter and the sandbox provider can never disagree about whether a host
can run iOSWorld.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from benchflow.adapters.inbound import (
    InboundCompatibility,
    InboundSupportReport,
    InboundTask,
    UnsupportedInboundTaskError,
    manifest_from_task_config,
)
from benchflow.task.config import TaskConfig

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

# The capabilities the host can actually advertise (a subset of
# _REQUIRED_CAPABILITIES). ``iosworld-app-bootstrap`` — building/installing the
# 26 SwiftUI apps — is a follow-up milestone, not a host fact this provider
# detects, so it is reported as a pending follow-up on the supported path
# rather than gating support.
_HOST_CAPABILITIES = [
    "macos",
    "xcode-26",
    "ios-26-simulator-runtime",
    "appium-xcuitest",
]

# The rubric is scored by an LLM judge; the criteria are written to this file
# (relative to the materialized task dir) and referenced by the verifier.
_RUBRIC_REL = "tests/rubric.md"
_DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


class IOSWorldAdapter:
    """Recognize iOSWorld sources and translate them when the host can run them."""

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
        if not report.supported:
            raise UnsupportedInboundTaskError(report)
        # Supported path: only a single-task slice can be translated to one
        # InboundTask (the repository shape is a whole suite, not one task).
        root = Path(task_dir)
        return _translate_task_slice(root)


def _host_can_run_iosworld() -> bool:
    """Whether the host advertises every detectable iOS-Simulator capability."""
    from benchflow.sandbox.macos_ios_simulator import (
        detect_ios_simulator_capabilities,
    )

    caps = detect_ios_simulator_capabilities()
    return all(caps.get(name, False) for name in _HOST_CAPABILITIES)


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
    if _host_can_run_iosworld():
        return InboundSupportReport(
            source=IOSWorldAdapter.source,
            supported=True,
            task_id=task_id,
            dataset="iosworld",
            reason=None,
            details={
                "provider": _REQUIRED_PROVIDER,
                "shape": "task-slice",
                "apps": _string_list(raw.get("apps")),
                "category": raw.get("category"),
                "difficulty": raw.get("difficulty"),
                "rubric_count": _rubric_count(raw.get("rubric")),
                # The host substrate is present; bootstrap of the 26 SwiftUI
                # apps is a follow-up step that runs at task setup time, not a
                # host capability — surface it as pending so the report stays
                # honest about what is and is not yet wired.
                "pending_capabilities": ["iosworld-app-bootstrap"],
            },
        )
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

    host_ready = _host_can_run_iosworld()
    details: dict[str, Any] = {
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
    }
    if host_ready:
        # The host can run iOSWorld, but a whole-repository source is a suite,
        # not a single translatable task. Report it as a not-yet-supported
        # shape (the per-task slice is the supported unit) rather than claiming
        # the suite as one task.
        details["issue"] = "iosworld-repository-suite-not-a-single-task"
        details["provider"] = _REQUIRED_PROVIDER
        details["pending_capabilities"] = ["iosworld-app-bootstrap"]
        return InboundSupportReport(
            source=IOSWorldAdapter.source,
            supported=False,
            dataset="iosworld",
            reason=(
                "iOSWorld repository is a task suite; split it into per-task "
                "iosworld-task.json slices to translate individual tasks"
            ),
            details=details,
        )
    details["issue"] = "macos-ios-simulator-provider-required"
    details["required_capabilities"] = _REQUIRED_CAPABILITIES
    return InboundSupportReport(
        source=IOSWorldAdapter.source,
        supported=False,
        dataset="iosworld",
        reason=_UNSUPPORTED_REASON,
        details=details,
    )


def _translate_task_slice(root: Path) -> InboundTask:
    """Translate an ``iosworld-task.json`` slice into a native ``InboundTask``."""
    path = root / IOSWORLD_TASK_FILE
    raw = _load_json_object(path)
    task_id = _task_id(raw, fallback=root.name)
    goal = _goal(raw, path)
    apps = _string_list(raw.get("apps"))
    rubric_items = _rubric_criteria(raw.get("rubric"))

    name = f"iosworld/{task_id}"
    instruction = goal + "\n"
    config = _build_config(name=name, task_id=task_id, raw=raw, apps=apps)
    assert config.task is not None
    manifest = manifest_from_task_config(name=config.task.name, config=config)
    rubric_md = _render_rubric(goal=goal, apps=apps, criteria=rubric_items)

    return InboundTask(
        name=task_id,
        source=IOSWorldAdapter.source,
        instruction=instruction,
        manifest=manifest,
        config=config,
        files={},
        generated_files={_RUBRIC_REL: rubric_md},
        compatibility=InboundCompatibility(
            source=IOSWorldAdapter.source,
            config_extra=_compat_metadata(raw),
            config_extra_paths=(IOSWORLD_TASK_FILE,),
        ),
    )


def _build_config(
    *, name: str, task_id: str, raw: dict[str, Any], apps: list[str]
) -> TaskConfig:
    keywords = ["iosworld", "ios", "simulator", "external-eval", *apps]
    config_payload: dict[str, Any] = {
        "schema_version": "1.3",
        "task": {
            "name": name,
            "description": f"iOSWorld benchmark task {task_id}",
            "keywords": keywords,
        },
        "metadata": {
            "benchmark": "iosworld",
            "iosworld": {
                key: raw[key]
                for key in ("name", "category", "difficulty", "apps")
                if key in raw and raw[key] is not None
            },
        },
        "source": IOSWorldAdapter.source,
        # The rubric is scored by an LLM judge against the agent's
        # deliverables — the same reward shape the Browser Use / computer-use
        # adapters use for criteria-style benchmarks.
        "verifier": {
            "type": "llm-judge",
            "judge": {
                "model": _DEFAULT_JUDGE_MODEL,
                "rubric_path": _RUBRIC_REL,
                "input_type": "deliverables",
            },
        },
        # The macos-ios-simulator provider runs on the macOS host.
        "environment": {"os": "macos"},
    }
    return TaskConfig.model_validate(config_payload)


def _render_rubric(*, goal: str, apps: list[str], criteria: list[str]) -> str:
    lines = [
        "# iOSWorld task rubric",
        "",
        f"Goal: {goal}",
    ]
    if apps:
        lines.append(f"Apps: {', '.join(apps)}")
    lines.extend(
        [
            "",
            "Award full reward only if the agent satisfies every criterion "
            "below; otherwise score the fraction completed.",
            "",
        ]
    )
    if criteria:
        lines.extend(f"- {criterion}" for criterion in criteria)
    else:
        lines.append("- Complete the stated goal.")
    return "\n".join(lines) + "\n"


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


def _goal(raw: dict[str, Any], path: Path) -> str:
    for key in ("goal", "instruction", "task"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"iOSWorld task JSON must define a non-empty 'goal': {path}")


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.strip()]


def _rubric_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _rubric_criteria(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    criteria: list[str] = []
    for item in value:
        if isinstance(item, dict):
            criterion = cast("dict[str, Any]", item).get("criterion")
            if isinstance(criterion, str) and criterion.strip():
                criteria.append(criterion.strip())
        elif isinstance(item, str) and item.strip():
            criteria.append(item.strip())
    return criteria


def _compat_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    keys = ("name", "goal", "apps", "category", "difficulty", "rubric")
    return {key: raw[key] for key in keys if key in raw}


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def from_iosworld_task(task_dir: Path | str) -> InboundTask:
    """Convenience function: translate an iOSWorld task when supported."""
    return IOSWorldAdapter.from_task_dir(task_dir)
