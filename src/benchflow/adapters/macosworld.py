"""Inbound recognition + translation for macOSWorld benchmark sources.

macOSWorld (``github.com/showlab/macosworld``) is an interactive macOS GUI
benchmark: 202 multilingual tasks across 30 macOS applications, scored by
running ``osascript``/shell *grading commands* inside a live macOS environment
that reports ``True``/``False`` per check. Each task ships multilingual
instruction text (``en``/``zh``/``ar``/``ja``/``ru``), a per-language
environment snapshot, optional setup ``pre_command`` / ``in_process`` steps,
and a list of grading commands. BenchFlow runs macOSWorld on the **cua**
sandbox provider (``benchflow.sandbox.cua``) backed by a macOS VM
(``TaskOS.MACOS`` → ``Image.macos(kind="vm")``), which on Apple Silicon boots
locally via Hypervisor.framework through lume.

This adapter is provider-honest in both directions, mirroring
:mod:`benchflow.adapters.iosworld`:

* When the host can run a cua macOS VM (Apple-Silicon macOS host with the
  optional ``cua`` SDK installed), it recognizes a ``macosworld-task.json``
  slice and translates it into a native
  :class:`~benchflow.adapters.inbound.InboundTask` — the English goal becomes
  the instruction and the grading commands become an LLM-judge rubric, the
  same reward shape the iOSWorld / Browser Use / computer-use adapters use for
  criteria-style benchmarks.
* When those prerequisites are missing (the common case off an Apple-Silicon
  Mac, and CI), it reports a structured *unsupported* result naming exactly
  which provider and capabilities are required, so ``tasks check`` stays honest
  about the blocker.

The ``macos-vm-exec`` capability — running grading commands *inside* the macOS
VM through cua's exec bridge — is a pending follow-up (the macOS exec bridge,
upstream task #19). Even when the macOS VM image exists, full execution may be
pending, so this adapter reports ``macos-vm-exec`` as a *pending* follow-up on
the supported path rather than claiming full end-to-end support prematurely.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchflow.adapters.inbound import (
    InboundCompatibility,
    InboundSupportReport,
    InboundTask,
    UnsupportedInboundTaskError,
    manifest_from_task_config,
)
from benchflow.task.config import TaskConfig

MACOSWORLD_TASK_FILE = "macosworld-task.json"

_TASK_ID_INVALID = re.compile(r"[^a-z0-9._-]+")
_UNSUPPORTED_REASON = "macOSWorld tasks require a cua macOS VM provider mapping"
_REQUIRED_PROVIDER = "cua"
_REQUIRED_CAPABILITIES = [
    "macos",
    "cua-macos-vm",
    "macos-vm-exec",
]

# The capabilities the host can actually advertise today (a subset of
# _REQUIRED_CAPABILITIES). ``macos-vm-exec`` — running the grading commands
# inside the macOS VM via cua's exec bridge — is a pending follow-up
# (the macOS exec bridge, #19), not a host fact this provider can yet detect,
# so it is reported as a pending follow-up on the supported path rather than
# gating recognition.
_HOST_CAPABILITIES = [
    "macos",
    "cua-macos-vm",
]

# The instruction is offered in several languages; the supported path
# translates the English variant, the one variant every task provides.
_DEFAULT_LANGUAGE = "en"

# The grading commands are scored by an LLM judge against the agent's
# deliverables; the rubric is written to this file (relative to the
# materialized task dir) and referenced by the verifier.
_RUBRIC_REL = "tests/rubric.md"
_DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


class MacOSWorldAdapter:
    """Recognize macOSWorld sources and translate them when the host can run them."""

    source = "macosworld"

    @classmethod
    def is_task_dir(cls, task_dir: Path | str) -> bool:
        return cls.support_report(task_dir) is not None

    @classmethod
    def support_report(cls, task_dir: Path | str) -> InboundSupportReport | None:
        root = Path(task_dir)
        if (root / MACOSWORLD_TASK_FILE).is_file():
            return _task_slice_report(root)
        if _looks_like_macosworld_repo(root):
            return _repo_report(root)
        return None

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        report = cls.support_report(task_dir)
        if report is None:
            root = Path(task_dir)
            raise FileNotFoundError(
                "macOSWorld task is missing macosworld-task.json or repository "
                f"signatures: {root}"
            )
        if not report.supported:
            raise UnsupportedInboundTaskError(report)
        # Supported path: only a single-task slice can be translated to one
        # InboundTask (the repository shape is a whole suite, not one task).
        root = Path(task_dir)
        return _translate_task_slice(root)


def _host_can_run_macosworld() -> bool:
    """Whether the host advertises every detectable cua macOS-VM capability."""
    caps = detect_cua_macos_capabilities()
    return all(caps.get(name, False) for name in _HOST_CAPABILITIES)


def detect_cua_macos_capabilities() -> dict[str, bool]:
    """Probe the host for the cua macOS-VM capabilities the adapter gates on.

    Returns a capability -> present mapping. macOSWorld runs on a macOS VM
    booted by the cua provider; on Apple Silicon that VM runs locally via
    Hypervisor.framework, so the host facts this checks are: a Darwin/arm64
    host and an importable ``cua`` SDK.

    Set ``BENCHFLOW_MACOSWORLD_FORCE_UNSUPPORTED=1`` to force every capability
    to ``False`` regardless of host — used by tests (and operators) that want
    the provider-honest *unsupported* path on a host that happens to be capable.
    """
    if _force_unsupported():
        return {"macos": False, "cua-macos-vm": False}

    import platform

    is_macos = platform.system() == "Darwin"
    # macOS VMs are VM-backed and on Apple Silicon run locally via
    # Hypervisor.framework; an Intel host cannot host the local macOS VM.
    is_apple_silicon = is_macos and platform.machine() in {"arm64", "aarch64"}
    return {
        "macos": is_macos,
        "cua-macos-vm": is_apple_silicon and _cua_sdk_available(),
    }


def _force_unsupported() -> bool:
    import os

    return os.environ.get(
        "BENCHFLOW_MACOSWORLD_FORCE_UNSUPPORTED", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _cua_sdk_available() -> bool:
    """Whether the optional cua SDK can be imported on this host."""
    import importlib.util

    return (
        importlib.util.find_spec("cua_sandbox") is not None
        or importlib.util.find_spec("cua") is not None
    )


def _task_slice_report(root: Path) -> InboundSupportReport:
    path = root / MACOSWORLD_TASK_FILE
    try:
        raw = _load_json_object(path)
    except ValueError as exc:
        return InboundSupportReport(
            source=MacOSWorldAdapter.source,
            supported=False,
            task_id=root.name,
            dataset="macosworld",
            reason=f"invalid {MACOSWORLD_TASK_FILE}: {exc}",
            details={
                "issue": "invalid-macosworld-task-json",
                "required_provider": _REQUIRED_PROVIDER,
                "required_capabilities": _REQUIRED_CAPABILITIES,
            },
        )

    task_id = _task_id(raw, fallback=root.name)
    if _host_can_run_macosworld():
        return InboundSupportReport(
            source=MacOSWorldAdapter.source,
            supported=True,
            task_id=task_id,
            dataset="macosworld",
            reason=None,
            details={
                "provider": _REQUIRED_PROVIDER,
                "shape": "task-slice",
                "languages": _languages(raw.get("task")),
                "grading_command_count": _grading_count(raw.get("grading_command")),
                # The macOS VM substrate is present; running the grading
                # commands inside the VM through cua's exec bridge (#19) is a
                # pending follow-up, not a host capability — surface it as
                # pending so the report stays honest about what is not yet wired.
                "pending_capabilities": ["macos-vm-exec"],
            },
        )
    return InboundSupportReport(
        source=MacOSWorldAdapter.source,
        supported=False,
        task_id=task_id,
        dataset="macosworld",
        reason=_UNSUPPORTED_REASON,
        details={
            "issue": "cua-macos-vm-provider-required",
            "required_provider": _REQUIRED_PROVIDER,
            "shape": "task-slice",
            "languages": _languages(raw.get("task")),
            "grading_command_count": _grading_count(raw.get("grading_command")),
            "required_capabilities": _REQUIRED_CAPABILITIES,
        },
    )


def _repo_report(root: Path) -> InboundSupportReport:
    tasks_dir = root / "tasks"
    # Count the task JSON files per category directory — the meaningful task
    # count, not the bare directory presence.
    categories: dict[str, int] = {}
    task_count = 0
    for category_dir in sorted(p for p in tasks_dir.iterdir() if p.is_dir()):
        count = sum(
            1 for p in category_dir.glob("*.json") if not p.name.startswith(".")
        )
        categories[category_dir.name] = count
        task_count += count
    host_ready = _host_can_run_macosworld()
    details: dict[str, Any] = {
        "required_provider": _REQUIRED_PROVIDER,
        "shape": "repository",
        "task_count": task_count,
        "categories": categories,
    }
    if host_ready:
        # The host can run macOSWorld, but a whole-repository source is a
        # suite, not a single translatable task. Report it as a not-yet-
        # supported shape (the per-task slice is the supported unit) rather
        # than claiming the suite as one task.
        details["issue"] = "macosworld-repository-suite-not-a-single-task"
        details["provider"] = _REQUIRED_PROVIDER
        details["pending_capabilities"] = ["macos-vm-exec"]
        return InboundSupportReport(
            source=MacOSWorldAdapter.source,
            supported=False,
            dataset="macosworld",
            reason=(
                "macOSWorld repository is a task suite; split it into per-task "
                "macosworld-task.json slices to translate individual tasks"
            ),
            details=details,
        )
    details["issue"] = "cua-macos-vm-provider-required"
    details["required_capabilities"] = _REQUIRED_CAPABILITIES
    return InboundSupportReport(
        source=MacOSWorldAdapter.source,
        supported=False,
        dataset="macosworld",
        reason=_UNSUPPORTED_REASON,
        details=details,
    )


def _translate_task_slice(root: Path) -> InboundTask:
    """Translate a ``macosworld-task.json`` slice into a native ``InboundTask``."""
    path = root / MACOSWORLD_TASK_FILE
    raw = _load_json_object(path)
    task_id = _task_id(raw, fallback=root.name)
    goal = _goal(raw, path)
    languages = _languages(raw.get("task"))
    grading = _grading_criteria(raw.get("grading_command"))

    name = f"macosworld/{task_id}"
    instruction = goal + "\n"
    config = _build_config(name=name, task_id=task_id, raw=raw, languages=languages)
    assert config.task is not None
    manifest = manifest_from_task_config(name=config.task.name, config=config)
    rubric_md = _render_rubric(goal=goal, criteria=grading)

    return InboundTask(
        name=task_id,
        source=MacOSWorldAdapter.source,
        instruction=instruction,
        manifest=manifest,
        config=config,
        files={},
        generated_files={_RUBRIC_REL: rubric_md},
        compatibility=InboundCompatibility(
            source=MacOSWorldAdapter.source,
            config_extra=_compat_metadata(raw),
            config_extra_paths=(MACOSWORLD_TASK_FILE,),
        ),
    )


def _build_config(
    *, name: str, task_id: str, raw: dict[str, Any], languages: list[str]
) -> TaskConfig:
    keywords = ["macosworld", "macos", "cua", "external-eval"]
    config_payload: dict[str, Any] = {
        "schema_version": "1.3",
        "task": {
            "name": name,
            "description": f"macOSWorld benchmark task {task_id}",
            "keywords": keywords,
        },
        "metadata": {
            "benchmark": "macosworld",
            "macosworld": {
                key: raw[key]
                for key in ("id", "languages", "snapshot")
                if key in raw and raw[key] is not None
            }
            | ({"languages": languages} if languages else {}),
        },
        "source": MacOSWorldAdapter.source,
        # The grading commands are scored by an LLM judge against the agent's
        # deliverables — the same reward shape the iOSWorld / Browser Use /
        # computer-use adapters use for criteria-style benchmarks.
        "verifier": {
            "type": "llm-judge",
            "judge": {
                "model": _DEFAULT_JUDGE_MODEL,
                "rubric_path": _RUBRIC_REL,
                "input_type": "deliverables",
            },
        },
        # The cua macOS VM provider runs on the macOS host.
        "environment": {"os": "macos"},
    }
    return TaskConfig.model_validate(config_payload)


def _render_rubric(*, goal: str, criteria: list[str]) -> str:
    lines = [
        "# macOSWorld task rubric",
        "",
        f"Goal: {goal}",
        "",
        "Award full reward only if the agent satisfies every criterion "
        "below; otherwise score the fraction completed.",
        "",
    ]
    if criteria:
        lines.extend(f"- {criterion}" for criterion in criteria)
    else:
        lines.append("- Complete the stated goal.")
    return "\n".join(lines) + "\n"


def _looks_like_macosworld_repo(root: Path) -> bool:
    return (
        (root / "testbench.py").is_file()
        and (root / "constants.py").is_file()
        and (root / "tasks").is_dir()
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object")
    return raw


def _task_id(raw: dict[str, Any], *, fallback: str) -> str:
    for key in ("id", "name", "task_id"):
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
    """The instruction text, preferring the English variant.

    macOSWorld's ``task`` is a per-language mapping (``en`` is always present);
    a plain-string ``task`` is also accepted for forward compatibility.
    """
    task = raw.get("task")
    if isinstance(task, str) and task.strip():
        return task.strip()
    if isinstance(task, dict):
        value = task.get(_DEFAULT_LANGUAGE)
        if isinstance(value, str) and value.strip():
            return value.strip()
        for candidate in task.values():
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    raise ValueError(
        f"macOSWorld task JSON must define a non-empty 'task' instruction: {path}"
    )


def _languages(task: object) -> list[str]:
    if not isinstance(task, dict):
        return []
    return sorted(
        key
        for key, value in task.items()
        if isinstance(key, str) and isinstance(value, str) and value.strip()
    )


def _grading_count(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _grading_criteria(value: object) -> list[str]:
    """Render macOSWorld grading commands as human-readable rubric criteria.

    Each grading entry is a ``[command, timeout]`` pair (occasionally with
    extra elements); the shell/``osascript`` command is the verifiable check.
    """
    if not isinstance(value, list):
        return []
    criteria: list[str] = []
    for item in value:
        command = _grading_command_text(item)
        if command:
            criteria.append(f"Grading check passes: {command}")
    return criteria


def _grading_command_text(item: object) -> str:
    if isinstance(item, str) and item.strip():
        return item.strip()
    if isinstance(item, list) and item:
        first = item[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return ""


def _compat_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "task",
        "snapshot",
        "grading_command",
        "in_process",
        "pre_command",
        "before_action_delay_seconds",
        "before_grading_delay_seconds",
        "force_snapshot_recovery",
        "force_ec2",
        "force_error_free_prep",
    )
    return {key: raw[key] for key in keys if key in raw}


def from_macosworld_task(task_dir: Path | str) -> InboundTask:
    """Convenience function: translate a macOSWorld task when supported."""
    return MacOSWorldAdapter.from_task_dir(task_dir)
