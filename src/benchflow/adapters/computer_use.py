"""Inbound adapter for computer-use benchmark task slices.

Computer-use benchmarks separate the desktop substrate from the agent loop and
task importer. This adapter handles the task side of that split: it translates
a computer-use-shaped task directory with a ``computer-use-task.json``
descriptor into BenchFlow's native ``InboundTask`` contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchflow.adapters.inbound import (
    InboundCompatibility,
    InboundTask,
    carry_native_subtrees,
    manifest_from_task_config,
)
from benchflow.environment.manifest import EnvironmentManifest
from benchflow.task.config import TaskConfig

COMPUTER_USE_TASK_FILE = "computer-use-task.json"

_TASK_ID_INVALID = re.compile(r"[^a-z0-9._-]+")
_INSTRUCTION_FIELDS = ("instruction", "confirmed_task", "task")
_COMPAT_KEYS = (
    "task_id",
    "benchmark",
    "category",
    "instruction",
    "expected_result",
    "expected_file",
    "roundtrip_file",
    "screenshot_required",
)


class ComputerUseAdapter:
    """Translate a computer-use task directory into an ``InboundTask``."""

    source = "computer-use-benchmark"

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        """Translate a computer-use-shaped task directory.

        The directory must contain ``computer-use-task.json``. Native-shaped
        ``environment/``, ``tests/``, and ``solution/`` subtrees are carried
        through unchanged for materialization.
        """
        root = Path(task_dir)
        descriptor_path = root / COMPUTER_USE_TASK_FILE
        if not descriptor_path.is_file():
            raise FileNotFoundError(
                f"Computer Use task is missing {COMPUTER_USE_TASK_FILE}: "
                f"{descriptor_path}"
            )

        raw = _load_descriptor(descriptor_path)
        task_id = _required_string(raw, "task_id", descriptor_path)
        short_name = _task_slug(task_id)
        instruction = _instruction(raw, descriptor_path)
        config = cls._build_config(name=f"computer-use/{short_name}", raw=raw)
        assert config.task is not None
        manifest = cls._load_manifest(root, name=config.task.name, config=config)
        files = cls._build_file_map(root)

        return InboundTask(
            name=short_name,
            source=cls.source,
            instruction=instruction,
            manifest=manifest,
            config=config,
            files=files,
            compatibility=InboundCompatibility(
                source=cls.source,
                config_extra=_compat_metadata(raw),
                config_extra_paths=(COMPUTER_USE_TASK_FILE,),
            ),
        )

    @staticmethod
    def _build_config(name: str, raw: dict[str, Any]) -> TaskConfig:
        timeout = _optional_positive_float(raw, "timeout_sec")
        agent_timeout = _optional_positive_float(raw, "agent_timeout_sec") or timeout
        verifier_timeout = (
            _optional_positive_float(raw, "verifier_timeout_sec") or timeout
        )
        task_id = _required_string(raw, "task_id", Path(COMPUTER_USE_TASK_FILE))

        metadata: dict[str, Any] = {
            "benchmark": raw.get("benchmark") or "computer-use",
            "computer_use": {
                key: raw[key]
                for key in (
                    "task_id",
                    "category",
                    "expected_result",
                    "expected_file",
                    "roundtrip_file",
                    "screenshot_required",
                )
                if key in raw and raw[key] is not None
            },
        }

        config_payload: dict[str, Any] = {
            "schema_version": "1.3",
            "task": {
                "name": name,
                "description": f"Computer-use benchmark task {task_id}",
                "keywords": ["computer-use", "desktop", "external-eval"],
            },
            "metadata": metadata,
            "source": ComputerUseAdapter.source,
        }
        if agent_timeout is not None:
            config_payload["agent"] = {"timeout_sec": agent_timeout}
        if verifier_timeout is not None:
            config_payload["verifier"] = {"timeout_sec": verifier_timeout}

        environment: dict[str, Any] = {}
        if raw.get("docker_image"):
            environment["docker_image"] = str(raw["docker_image"])
        if raw.get("workdir"):
            environment["workdir"] = str(raw["workdir"])
        if environment:
            config_payload["environment"] = environment

        return TaskConfig.model_validate(config_payload)

    @staticmethod
    def _load_manifest(
        root: Path, *, name: str, config: TaskConfig
    ) -> EnvironmentManifest:
        manifest_path = root / "environment.toml"
        if manifest_path.is_file():
            return EnvironmentManifest.model_validate_toml(manifest_path.read_text())
        return manifest_from_task_config(name=name, config=config)

    @staticmethod
    def _build_file_map(root: Path) -> dict[str, Path]:
        files: dict[str, Path] = {}

        def _place(native: str, src: Path) -> None:
            existing = files.get(native)
            if existing is not None and existing != src:
                raise ValueError(
                    "Computer Use task file map collision for "
                    f"{native!r}: {existing} vs {src}"
                )
            files[native] = src

        carry_native_subtrees(root, _place)
        return files


def _load_descriptor(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Computer Use task JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Computer Use task JSON must be an object: {path}")
    return raw


def _required_string(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Computer Use task JSON must define a non-empty {key!r}: {path}"
        )
    return value.strip()


def _instruction(raw: dict[str, Any], path: Path) -> str:
    for key in _INSTRUCTION_FIELDS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip() + "\n"
    fields = ", ".join(repr(field) for field in _INSTRUCTION_FIELDS)
    raise ValueError(
        f"Computer Use task JSON must define one of {fields} as the instruction: {path}"
    )


def _task_slug(task_id: str) -> str:
    slug = _TASK_ID_INVALID.sub("-", task_id.strip().lower()).strip("-._")
    if not slug:
        slug = "task"
    if not slug[0].isalnum():
        slug = f"task-{slug}"
    return slug


def _optional_positive_float(raw: dict[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Computer Use task {key!r} must be numeric") from exc
    if number <= 0:
        raise ValueError(f"Computer Use task {key!r} must be positive")
    return number


def _compat_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in _COMPAT_KEYS if key in raw}


def from_computer_use_task(task_dir: Path | str) -> InboundTask:
    """Convenience function: translate a computer-use task directory."""
    return ComputerUseAdapter.from_task_dir(task_dir)
