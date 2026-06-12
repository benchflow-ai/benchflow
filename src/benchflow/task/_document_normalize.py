"""Frontmatter normalization for ``task.md`` documents.

This layer expands authoring shorthands (``name``, ``image``, ``verifier``,
``oracle``) and profile presets into canonical frontmatter. It owns the shared
dict-merge helpers (``_deep_merge``, ``_merge_missing``, ``_ensure_mapping``,
``_mapping``) that the evidence and parse layers also consume, keeping the
import graph acyclic.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from benchflow.task._document_profiles import _PROFILE_KEYS, _TASK_AUTHORING_PROFILES


class TaskDocumentParseError(ValueError):
    """Raised when a ``task.md`` document cannot be parsed."""


def normalize_task_document_frontmatter(
    frontmatter: dict[str, Any],
    *,
    task_dir: Path | None = None,
) -> dict[str, Any]:
    """Expand authoring shorthands and profiles into canonical frontmatter."""

    raw = deepcopy(frontmatter)
    profiles = _parse_authoring_profiles(raw)
    profile_defaults: dict[str, Any] = {}
    for profile in profiles:
        profile_defaults = _deep_merge(
            profile_defaults,
            deepcopy(_TASK_AUTHORING_PROFILES[profile]),
        )

    shorthand_name = raw.pop("name", None)
    shorthand_image = raw.pop("image", None)
    verifier_path = _pop_path_shorthand(raw, "verifier")
    oracle_path = _pop_path_shorthand(raw, "oracle")
    for key in _PROFILE_KEYS:
        raw.pop(key, None)

    normalized = _deep_merge(profile_defaults, raw)
    _apply_name_shorthand(
        normalized,
        shorthand_name,
        canonical_was_explicit=_has_nested(raw, ("task", "name")),
    )
    _apply_image_shorthand(
        normalized,
        shorthand_image,
        canonical_was_explicit=_has_nested(raw, ("environment", "docker_image")),
    )
    _apply_path_shorthand(normalized, "verifier", verifier_path)
    _apply_path_shorthand(normalized, "oracle", oracle_path)
    _record_applied_profiles(normalized, profiles)
    if task_dir is not None:
        from benchflow.task._document_evidence import _apply_conventional_evidence

        _apply_conventional_evidence(normalized, task_dir=task_dir, profiles=profiles)
    return normalized


def _parse_authoring_profiles(frontmatter: dict[str, Any]) -> list[str]:
    profiles: list[str] = []
    for key in _PROFILE_KEYS:
        raw_value = frontmatter.get(key)
        if raw_value is None:
            continue
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise TaskDocumentParseError(f"{key} entries must be profile names")
            profile = value.strip()
            if profile not in _TASK_AUTHORING_PROFILES:
                known = ", ".join(sorted(_TASK_AUTHORING_PROFILES))
                raise TaskDocumentParseError(
                    f"unknown task.md profile {profile!r}; known profiles: {known}"
                )
            if profile not in profiles:
                profiles.append(profile)
    return profiles


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _merge_missing(base: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = deepcopy(value)
        elif isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_missing(merged[key], value)
    return merged


def _has_nested(mapping: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        child: dict[str, Any] = {}
        parent[key] = child
        return child
    if not isinstance(value, dict):
        raise TaskDocumentParseError(f"{key} must be a mapping after normalization")
    return value


def _apply_name_shorthand(
    normalized: dict[str, Any],
    value: Any,
    *,
    canonical_was_explicit: bool,
) -> None:
    if value is None or canonical_was_explicit:
        return
    if not isinstance(value, str) or not value.strip():
        raise TaskDocumentParseError("name must be a non-empty string")
    task = _ensure_mapping(normalized, "task")
    name = value.strip()
    task["name"] = name if "/" in name else f"benchflow/{name}"


def _apply_image_shorthand(
    normalized: dict[str, Any],
    value: Any,
    *,
    canonical_was_explicit: bool,
) -> None:
    if value is None or canonical_was_explicit:
        return
    if not isinstance(value, str) or not value.strip():
        raise TaskDocumentParseError("image must be a non-empty string")
    environment = _ensure_mapping(normalized, "environment")
    environment["docker_image"] = value.strip()


def _pop_path_shorthand(frontmatter: dict[str, Any], key: str) -> str | None:
    value = frontmatter.get(key)
    if value is None or isinstance(value, dict):
        return None
    if not isinstance(value, str) or not value.strip():
        raise TaskDocumentParseError(f"{key} must be a mapping or non-empty path")
    frontmatter.pop(key)
    return value.strip()


def _apply_path_shorthand(
    normalized: dict[str, Any],
    key: str,
    path: str | None,
) -> None:
    if path is None:
        return
    safe_path = _safe_relative_posix_path(path, source=key)
    benchflow = _ensure_mapping(normalized, "benchflow")
    if key == "verifier":
        verifier = _ensure_mapping(benchflow, "verifier")
        base = safe_path.rstrip("/")
        verifier.setdefault("path", base + "/")
        verifier.setdefault("spec", f"{base}/verifier.md")
        verifier.setdefault("entrypoint", f"{base}/test.sh")
    else:
        oracle = _ensure_mapping(benchflow, "oracle")
        oracle.setdefault("path", safe_path.rstrip("/") + "/")


def _safe_relative_posix_path(value: str, *, source: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise TaskDocumentParseError(f"{source} path must be safe and relative")
    return path.as_posix()


def _record_applied_profiles(
    normalized: dict[str, Any],
    profiles: list[str],
) -> None:
    if not profiles:
        return
    benchflow = _ensure_mapping(normalized, "benchflow")
    authoring = _ensure_mapping(benchflow, "authoring")
    authoring.setdefault("profiles", profiles)
    authoring.setdefault("normalized", True)


def _mapping(
    value: Any, source: str, *, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    if value is None:
        return {} if default is None else dict(default)
    if not isinstance(value, dict):
        raise TaskDocumentParseError(f"{source} must be a mapping")
    return value
