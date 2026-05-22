"""Source provenance validation helpers for benchmark artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

SOURCE_REQUIRED = {
    "type",
    "repo",
    "requested_ref",
    "resolved_sha",
    "path",
    "local_path",
    "dirty",
    "file_hashes",
}


def _is_hex(value: str) -> bool:
    return all(ch in "0123456789abcdefABCDEF" for ch in value)


def is_git_hash(value: object) -> bool:
    """Return True for SHA-1 or SHA-256 git object IDs."""
    return isinstance(value, str) and len(value) in {40, 64} and _is_hex(value)


def is_sha256_digest(value: object) -> bool:
    """Return True for artifact file digests in ``sha256:<hex>`` form."""
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and _is_hex(digest)


def source_issues(
    source: object,
    label: str,
    *,
    require_file_hashes: bool,
    require_clean: bool = True,
) -> list[str]:
    """Return audit issues for one source provenance block."""
    if not isinstance(source, dict):
        return [f"{label}: missing source provenance"]

    source_dict = cast(dict[str, Any], source)
    issues: list[str] = []
    missing = SOURCE_REQUIRED - set(source_dict.keys())
    if missing:
        issues.append(f"{label}: source missing {missing}")
        return issues
    if source_dict.get("type") != "github":
        issues.append(f"{label}: source.type must be 'github'")
    repo = source_dict.get("repo")
    if not isinstance(repo, str) or "/" not in repo:
        issues.append(f"{label}: source.repo must be org/repo")
    if not is_git_hash(source_dict.get("resolved_sha")):
        issues.append(f"{label}: source.resolved_sha must be a git hash")
    dirty = source_dict.get("dirty")
    if not isinstance(dirty, bool):
        issues.append(f"{label}: source.dirty must be a boolean")
    elif require_clean and dirty:
        issues.append(f"{label}: source.dirty must be false for validation evidence")
    file_hashes = source_dict.get("file_hashes")
    if not isinstance(file_hashes, dict):
        issues.append(f"{label}: source.file_hashes must be an object")
    elif require_file_hashes and not file_hashes:
        issues.append(f"{label}: source.file_hashes must include task files")
    elif require_file_hashes and "task.toml" not in file_hashes:
        issues.append(f"{label}: source.file_hashes must include task.toml")
    elif isinstance(file_hashes, dict):
        bad_hashes = [
            name
            for name, digest in file_hashes.items()
            if not isinstance(name, str) or not is_sha256_digest(digest)
        ]
        if bad_hashes:
            issues.append(f"{label}: invalid source.file_hashes entries: {bad_hashes}")
    return issues


def source_matches_parent(
    result_source: dict[str, Any] | None, parent_source: dict[str, Any]
) -> bool:
    """Return True when a task source is covered by a parent source."""
    if not isinstance(result_source, dict):
        return False
    for key in ("type", "repo", "requested_ref", "resolved_sha", "dirty"):
        if result_source.get(key) != parent_source.get(key):
            return False
    parent_path = str(parent_source.get("path") or "").strip("/")
    result_path = str(result_source.get("path") or "").strip("/")
    if (
        parent_path
        and result_path != parent_path
        and not result_path.startswith(f"{parent_path}/")
    ):
        return False
    parent_local = parent_source.get("local_path")
    result_local = result_source.get("local_path")
    if isinstance(parent_local, str) and parent_local:
        if not isinstance(result_local, str) or not result_local:
            return False
        parent_local_path = Path(parent_local).resolve(strict=False)
        result_local_path = Path(result_local).resolve(strict=False)
        if result_local_path != parent_local_path and not result_local_path.is_relative_to(
            parent_local_path
        ):
            return False
    return True


def summary_source_fields(
    parent_source: dict[str, Any] | None, results: dict[str, dict]
) -> dict[str, Any]:
    """Return safe source fields for summary.json."""
    if not parent_source:
        return {}
    mismatches = [
        task
        for task, result in sorted(results.items())
        if not source_matches_parent(result.get("source"), parent_source)
    ]
    if mismatches:
        return {"source_mismatch_tasks": mismatches}
    return {"source": parent_source}
