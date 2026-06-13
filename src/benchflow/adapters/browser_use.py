"""Inbound adapter for Browser Use benchmark task slices.

The Browser Use benchmark separates benchmark tasks from framework/browser
runners. This adapter handles the task side of that split: it translates a
Browser Use-shaped task directory with a ``browser-use-task.json`` descriptor
into BenchFlow's native ``InboundTask`` contract.
"""

from __future__ import annotations

import base64
import hashlib
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

BROWSER_USE_TASK_FILE = "browser-use-task.json"

_TASK_ID_INVALID = re.compile(r"[^a-z0-9._-]+")
_INSTRUCTION_FIELDS = ("confirmed_task", "instruction", "task")
_COMPAT_KEYS = (
    "task_id",
    "benchmark",
    "category",
    "confirmed_task",
    "ground_truth",
    "answer",
    "expected_result",
    "url",
    "start_url",
    "upstream_task_index",
)


class BrowserUseAdapter:
    """Translate a Browser Use benchmark task directory into an ``InboundTask``."""

    source = "browser-use-benchmark"

    @classmethod
    def from_task_dir(cls, task_dir: Path | str) -> InboundTask:
        """Translate a Browser Use-shaped task directory.

        The directory must contain ``browser-use-task.json``. Native-shaped
        ``environment/``, ``tests/``, and ``solution/`` subtrees are carried
        through unchanged for materialization.
        """
        root = Path(task_dir)
        descriptor_path = root / BROWSER_USE_TASK_FILE
        if not descriptor_path.is_file():
            raise FileNotFoundError(
                f"Browser Use task is missing {BROWSER_USE_TASK_FILE}: "
                f"{descriptor_path}"
            )

        raw = _load_descriptor(descriptor_path)
        task_id = _required_string(raw, "task_id", descriptor_path)
        short_name = _task_slug(task_id)
        instruction = _instruction(raw, descriptor_path)
        config = cls._build_config(name=f"browser-use/{short_name}", raw=raw)
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
                config_extra_paths=(BROWSER_USE_TASK_FILE,),
            ),
        )

    @staticmethod
    def _build_config(name: str, raw: dict[str, Any]) -> TaskConfig:
        timeout = _optional_positive_float(raw, "timeout_sec")
        agent_timeout = _optional_positive_float(raw, "agent_timeout_sec") or timeout
        verifier_timeout = (
            _optional_positive_float(raw, "verifier_timeout_sec") or timeout
        )
        task_id = _required_string(raw, "task_id", Path(BROWSER_USE_TASK_FILE))

        metadata: dict[str, Any] = {
            "benchmark": raw.get("benchmark") or "browser-use",
            "browser_use": {
                key: raw[key]
                for key in (
                    "task_id",
                    "category",
                    "ground_truth",
                    "expected_result",
                    "url",
                    "start_url",
                )
                if key in raw and raw[key] is not None
            },
        }

        config_payload: dict[str, Any] = {
            "schema_version": "1.3",
            "task": {
                "name": name,
                "description": f"Browser Use benchmark task {task_id}",
                "keywords": ["browser-use", "browser", "external-eval"],
            },
            "metadata": metadata,
            "source": BrowserUseAdapter.source,
        }
        if agent_timeout is not None:
            config_payload["agent"] = {"timeout_sec": agent_timeout}
        verifier = _verifier_config(raw, verifier_timeout=verifier_timeout)
        if verifier:
            config_payload["verifier"] = verifier
        if raw.get("docker_image"):
            config_payload["environment"] = {"docker_image": str(raw["docker_image"])}

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
                    "Browser Use task file map collision for "
                    f"{native!r}: {existing} vs {src}"
                )
            files[native] = src

        carry_native_subtrees(root, _place)
        return files


def _load_descriptor(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Browser Use task JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Browser Use task JSON must be an object: {path}")
    return raw


def load_encrypted_benchmark_tasks(
    encrypted_file: Path | str,
    *,
    benchmark: str | None = None,
    interleave: bool = True,
) -> list[dict[str, Any]]:
    """Load Browser Use's encrypted public benchmark task list in memory.

    Browser Use publishes task suites such as ``BU_Bench_V1.enc`` encrypted
    with a deterministic key derived from the benchmark name. This helper
    mirrors their public runner without writing plaintext tasks to the repo.
    """
    path = Path(encrypted_file)
    benchmark_name = benchmark or path.stem
    try:
        from cryptography.fernet import Fernet
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cryptography is required to read Browser Use encrypted suites"
        ) from exc

    key = base64.urlsafe_b64encode(hashlib.sha256(benchmark_name.encode()).digest())
    encrypted = base64.b64decode(path.read_text())
    raw = json.loads(Fernet(key).decrypt(encrypted))
    if not isinstance(raw, list):
        raise ValueError(f"Browser Use encrypted suite must decode to a list: {path}")
    tasks = [task for task in raw if isinstance(task, dict)]
    if len(tasks) != len(raw):
        raise ValueError(
            f"Browser Use encrypted suite contains non-object tasks: {path}"
        )
    return interleave_benchmark_tasks(tasks) if interleave else tasks


def interleave_benchmark_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match Browser Use's public distributed runner task ordering."""
    if len(tasks) != 100:
        return list(tasks)
    return [tasks[d * 20 + i] for i in range(20) for d in range(5)]


def official_task_descriptor(
    task: dict[str, Any],
    *,
    benchmark: str,
    task_index: int | None = None,
    judge_model: str = "gemini-2.5-flash",
    judge_env_key: str = "GEMINI_API_KEY",
    judge_input_dir: str = "/logs/artifacts",
    rubric_path: str = "tests/rubric.toml",
    timeout_sec: int = 1800,
) -> dict[str, Any]:
    """Convert one official Browser Use task object to our task descriptor."""
    task_id = _required_string(task, "task_id", Path(benchmark))
    descriptor: dict[str, Any] = {
        "task_id": task_id,
        "benchmark": benchmark,
        "category": str(task.get("category") or "browser-use"),
        "confirmed_task": _required_string(task, "confirmed_task", Path(benchmark)),
        "ground_truth": task.get("answer"),
        "answer": task.get("answer"),
        "timeout_sec": timeout_sec,
        "verifier": {
            "type": "llm-judge",
            "timeout_sec": timeout_sec,
            "judge": {
                "model": judge_model,
                "rubric_path": rubric_path,
                "input_dir": judge_input_dir,
                "input_type": "deliverables",
            },
            "env": {
                judge_env_key: f"${{{judge_env_key}}}",
            },
        },
    }
    if task_index is not None:
        descriptor["upstream_task_index"] = task_index
    return descriptor


def _required_string(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Browser Use task JSON must define a non-empty {key!r}: {path}"
        )
    return value.strip()


def _instruction(raw: dict[str, Any], path: Path) -> str:
    for key in _INSTRUCTION_FIELDS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip() + "\n"
    fields = ", ".join(repr(field) for field in _INSTRUCTION_FIELDS)
    raise ValueError(
        f"Browser Use task JSON must define one of {fields} as the instruction: {path}"
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
        raise ValueError(f"Browser Use task {key!r} must be numeric") from exc
    if number <= 0:
        raise ValueError(f"Browser Use task {key!r} must be positive")
    return number


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _verifier_config(
    raw: dict[str, Any],
    *,
    verifier_timeout: float | None,
) -> dict[str, Any]:
    verifier: dict[str, Any] = {}
    if verifier_timeout is not None:
        verifier["timeout_sec"] = verifier_timeout

    raw_verifier = raw.get("verifier")
    if not isinstance(raw_verifier, dict):
        return verifier

    verifier_type = _optional_string(raw_verifier.get("type"))
    if verifier_type is not None:
        verifier["type"] = verifier_type

    raw_timeout = _optional_positive_float(raw_verifier, "timeout_sec")
    if raw_timeout is not None:
        verifier["timeout_sec"] = raw_timeout

    judge = raw_verifier.get("judge")
    if isinstance(judge, dict):
        judge_payload: dict[str, Any] = {}
        for key in ("model", "rubric_path", "input_dir", "input_type", "context"):
            value = _optional_string(judge.get(key))
            if value is not None:
                judge_payload[key] = value
        if judge_payload:
            verifier["judge"] = judge_payload

    env = raw_verifier.get("env")
    if isinstance(env, dict):
        verifier["env"] = {
            str(key): str(value)
            for key, value in env.items()
            if isinstance(key, str) and key
        }

    return verifier


def _compat_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in _COMPAT_KEYS if key in raw}


def from_browser_use_task(task_dir: Path | str) -> InboundTask:
    """Convenience function: translate a Browser Use task directory."""
    return BrowserUseAdapter.from_task_dir(task_dir)
