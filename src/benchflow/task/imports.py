"""Compatibility imports for foreign task configuration files."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from benchflow.task.config import TaskConfig


@dataclass(frozen=True)
class TaskConfigImportReport:
    """Report for a foreign ``task.toml`` import.

    Native ``TaskConfig`` remains strict. Foreign adapters use this report to
    preserve unknown upstream keys in an explicit compatibility envelope rather
    than accepting them as first-class BenchFlow schema.
    """

    source: str
    status: Literal["strict", "preserved-extra"]
    extra: dict[str, Any] = field(default_factory=dict)
    extra_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["extra_paths"] = list(self.extra_paths)
        return data


@dataclass(frozen=True)
class ImportedTaskConfig:
    """A validated task config plus any preserved foreign extensions.

    ``declared`` is the parsed source mapping restricted to natively supported
    keys (compat extras removed). It records what the author actually wrote,
    so emitters can stay minimal instead of materializing model defaults.
    """

    config: TaskConfig
    report: TaskConfigImportReport
    declared: dict[str, Any]


def import_task_config_toml(
    toml_data: str,
    *,
    source: str,
) -> ImportedTaskConfig:
    """Validate foreign TOML while preserving unknown extension keys.

    This is deliberately separate from :meth:`TaskConfig.model_validate_toml`.
    The native parser keeps rejecting unknown keys; compatibility importers can
    opt into this two-pass parse when their job is to ingest foreign tasks.
    """

    raw = tomllib.loads(toml_data)
    try:
        config = TaskConfig.model_validate(copy.deepcopy(raw))
    except ValidationError as exc:
        extra_errors = [
            error for error in exc.errors() if error.get("type") == "extra_forbidden"
        ]
        if not extra_errors:
            raise

        sanitized = copy.deepcopy(raw)
        extra: dict[str, Any] = {}
        for error in extra_errors:
            path = tuple(error["loc"])
            value = _pop_path(sanitized, path)
            _set_path(extra, path, value)

        declared = copy.deepcopy(sanitized)
        try:
            config = TaskConfig.model_validate(sanitized)
        except ValidationError as sanitized_exc:
            raise exc from sanitized_exc

        paths = tuple(sorted(_format_path(path) for path in _leaf_paths(extra)))
        return ImportedTaskConfig(
            config=config,
            report=TaskConfigImportReport(
                source=source,
                status="preserved-extra",
                extra=extra,
                extra_paths=paths,
            ),
            declared=declared,
        )

    return ImportedTaskConfig(
        config=config,
        report=TaskConfigImportReport(source=source, status="strict"),
        declared=raw,
    )


def merge_compat_extra(
    base: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Return ``base`` with preserved foreign keys restored.

    ``extra`` comes from a compatibility envelope and must not overwrite a
    supported native key. A collision means the native schema learned that key
    after import, so the native value is authoritative.
    """

    merged = copy.deepcopy(base)
    _merge_missing(merged, extra)
    return merged


def _pop_path(data: dict[str, Any], path: tuple[str | int, ...]) -> Any:
    current: Any = data
    for part in path[:-1]:
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if isinstance(current, list) and isinstance(part, int):
            if part >= len(current):
                return None
            current = current[part]
            continue
        return None
    final_part = path[-1]
    if isinstance(current, dict) and isinstance(final_part, str):
        return current.pop(final_part, None)
    if isinstance(current, list) and isinstance(final_part, int):
        if final_part >= len(current):
            return None
        value = current[final_part]
        current[final_part] = None
        return value
    return None


def _set_path(data: dict[str, Any], path: tuple[str | int, ...], value: Any) -> None:
    current: Any = data
    for index, part in enumerate(path[:-1]):
        next_part = path[index + 1]
        child = _get_child(current, part)
        if not _container_matches(child, next_part):
            child = [] if isinstance(next_part, int) else {}
            _assign_child(current, part, child)
        current = child
    _assign_child(current, path[-1], value)


def _get_child(container: Any, part: str | int) -> Any:
    if isinstance(container, dict):
        return container.get(part)
    if isinstance(container, list) and isinstance(part, int) and part < len(container):
        return container[part]
    return None


def _assign_child(container: Any, part: str | int, value: Any) -> None:
    if isinstance(container, dict):
        container[part] = value
        return
    if isinstance(container, list) and isinstance(part, int):
        while len(container) <= part:
            container.append(None)
        container[part] = value
        return
    raise TypeError(f"cannot assign compatibility path segment {part!r}")


def _container_matches(value: Any, next_part: str | int) -> bool:
    return (
        isinstance(value, list)
        if isinstance(next_part, int)
        else isinstance(value, dict)
    )


def _leaf_paths(
    data: dict[str, Any] | list[Any],
    prefix: tuple[str | int, ...] = (),
) -> list[tuple[str | int, ...]]:
    paths: list[tuple[str | int, ...]] = []
    items = enumerate(data) if isinstance(data, list) else data.items()
    for key, value in items:
        path = (*prefix, key)
        if isinstance(value, dict | list):
            child_paths = _leaf_paths(value, path)
            paths.extend(child_paths or [path])
        elif value is not None:
            paths.append(path)
    return paths


def _format_path(path: tuple[str | int, ...]) -> str:
    rendered = ""
    for part in path:
        if isinstance(part, int):
            rendered += f"[{part}]"
        elif rendered:
            rendered += f".{part}"
        else:
            rendered = part
    return rendered


def _merge_missing(
    target: dict[str, Any] | list[Any], extra: dict[str, Any] | list[Any]
) -> None:
    if isinstance(target, list) and isinstance(extra, list):
        for index, value in enumerate(extra):
            if value is None:
                continue
            while len(target) <= index:
                target.append(None)
            if target[index] is None:
                target[index] = copy.deepcopy(value)
            elif isinstance(target[index], dict | list) and isinstance(
                value, dict | list
            ):
                _merge_missing(target[index], value)
        return

    if not isinstance(target, dict) or not isinstance(extra, dict):
        return

    for key, value in extra.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict | list) and isinstance(value, dict | list):
            _merge_missing(target[key], value)
