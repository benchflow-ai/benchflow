"""Pure-local staging for trajectory contributions."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Never

from benchflow import __version__
from benchflow.publish.redact import (
    redact_value,
    redact_value_to_stability,
    redaction_breakdown,
)

MAX_FILE_BYTES = 128 * 1024**2
MAX_ARTIFACTS = 8
MAX_JSONL_CAPTURE_BYTES = 2 * MAX_FILE_BYTES
# One optional workspace snapshot may ride along with the JSONL trajectory.
# Zips above this cap are skipped locally (never created) so contributors'
# disks and uplinks are not burdened by runaway workspaces.
MAX_ATTACHMENT_BYTES = 1024**3
MAX_ATTACHMENT_FILES = 50_000
MAX_TOTAL_ARTIFACTS = MAX_ARTIFACTS + 1
MAX_CAPTURE_BYTES = MAX_JSONL_CAPTURE_BYTES + MAX_ATTACHMENT_BYTES
MAX_MANIFEST_BYTES = 1024**2
MAX_RUN_METADATA_BYTES = 1024**2
MAX_UPLOADED_BY_LENGTH = 256
MAX_GITHUB_ID_LENGTH = 39
MAX_EMAIL_LENGTH = 254
MAX_JSONL_RECORD_BYTES = 8 * 1024**2
MAX_JSON_NESTING = 100
MAX_JSON_NODES = 100_000
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
ARTIFACT_NAME_PATTERN = re.compile(
    r"^(?:trajectory/[A-Za-z0-9._-]{1,128}\.jsonl"
    r"|workspace/[A-Za-z0-9._-]{1,128}\.zip)$"
)
TRAJECTORY_ARTIFACT_PREFIX = "trajectory/"
WORKSPACE_ARTIFACT_PREFIX = "workspace/"
# Directories that never belong in a workspace snapshot: VCS internals,
# dependency trees, and caches. ``.git`` also keeps credential-bearing
# remotes and hooks out of the archive.
WORKSPACE_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".uv-cache",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        ".DS_Store",
    }
)
# Filename shapes that overwhelmingly carry credentials. Workspace archives
# are uploaded as-is (no content redaction), so obvious secret carriers stay
# on the contributor's machine.
WORKSPACE_EXCLUDED_FILES = (
    # A linked git worktree's ``.git`` is a pointer FILE (not a directory)
    # carrying a local absolute path; the directory case is excluded above.
    re.compile(r"^\.git$", re.IGNORECASE),
    re.compile(r"^\.env(\..+)?$", re.IGNORECASE),
    re.compile(r".*\.(pem|key|p12|pfx|keystore)$", re.IGNORECASE),
    re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)(\..+)?$", re.IGNORECASE),
    re.compile(r"^\.netrc$", re.IGNORECASE),
    re.compile(r"^\.npmrc$", re.IGNORECASE),
    re.compile(r"^(credentials|secrets?)(\..+)?$", re.IGNORECASE),
    re.compile(r"^\.DS_Store$", re.IGNORECASE),
)
GITHUB_ID_PATTERN = re.compile(
    rf"^[A-Za-z0-9](?:[A-Za-z0-9-]{{0,{MAX_GITHUB_ID_LENGTH - 2}}}[A-Za-z0-9])?$"
)
EMAIL_LOCAL_PATTERN = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+$")
EMAIL_DOMAIN_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, item in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON object key: {key}")
        parsed[key] = item
    return parsed


def _reject_non_finite_json(constant: str) -> Never:
    raise ValueError(f"non-finite JSON number: {constant}")


def strict_json_loads(value: str | bytes) -> Any:
    """Parse standards-compliant JSON while rejecting ambiguous object keys."""
    return json.loads(
        value,
        object_pairs_hook=_reject_duplicate_json_keys,
        parse_constant=_reject_non_finite_json,
    )


@dataclass(frozen=True)
class StagedFile:
    relname: str
    local_path: Path
    sha256: str
    size_bytes: int
    content_type: str
    created_at: datetime | None = None


@dataclass(frozen=True)
class StagedTrajectoryArtifacts:
    """Validated, redacted artifacts awaiting contributor finalization."""

    source_id: str
    traj_digest: str
    files: tuple[StagedFile, ...]
    ignored: tuple[str, ...]
    redaction_replacements: int
    _metadata_dir: Path | None
    _staging_dir: Path
    _redact: bool
    # Per-category artifact replacement counts in display order (display-only:
    # the manifest/server contract stays untouched). Sums to
    # ``redaction_replacements``.
    redaction_categories: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class StagedCapture:
    source_id: str
    traj_digest: str
    files: tuple[StagedFile, ...]
    manifest: dict[str, Any]
    ignored: tuple[str, ...]
    artifact_redaction_replacements: int
    redaction_replacements: int
    # Display-only per-category counts for the artifact replacements.
    artifact_redaction_categories: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class _ResolvedInput:
    files: tuple[Path, ...]
    metadata_dir: Path | None
    ignored: tuple[str, ...]


def default_source_id(path: Path) -> str:
    """Derive a safe source id from a trajectory path."""
    resolved = path.expanduser().resolve()
    raw = resolved.stem if resolved.is_file() else resolved.name
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")
    return validate_source_id(sanitized or "trajectory")


def validate_source_id(source_id: str) -> str:
    """Validate and normalize a source id used in object names."""
    normalized = source_id.strip().strip("/")
    invalid_segment = any(segment in {".", ".."} for segment in normalized.split("/"))
    if (
        not SOURCE_ID_PATTERN.fullmatch(normalized)
        or "//" in normalized
        or invalid_segment
    ):
        raise ValueError(
            "invalid source id; use --source-id with 1-128 letters, numbers, "
            "dots, underscores, hyphens, or single path separators"
        )
    return normalized


def validate_github_id(github_id: str) -> str:
    """Validate a self-asserted GitHub username stored as contributor provenance."""
    normalized = github_id.strip()
    if not GITHUB_ID_PATTERN.fullmatch(normalized) or "--" in normalized:
        raise ValueError(
            "invalid GitHub ID; use the 1-39 character username without '@'"
        )
    return normalized


def validate_email(email: str) -> str:
    """Validate a bounded ASCII contributor email address."""
    normalized = email.strip()
    if len(normalized) > MAX_EMAIL_LENGTH or normalized.count("@") != 1:
        raise ValueError("invalid contributor email address")
    local, domain = normalized.rsplit("@", 1)
    labels = domain.split(".")
    if (
        not local
        or len(local) > 64
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or not EMAIL_LOCAL_PATTERN.fullmatch(local)
        or len(labels) < 2
        or any(
            not label
            or len(label) > 63
            or not EMAIL_DOMAIN_LABEL_PATTERN.fullmatch(label)
            for label in labels
        )
    ):
        raise ValueError("invalid contributor email address")
    return normalized


def validate_artifact_name(name: str) -> str:
    """Require the canonical public capture object namespaces."""
    if not ARTIFACT_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "artifact name is outside trajectory/*.jsonl and workspace/*.zip"
        )
    _, replacements = redact_value(name)
    if replacements:
        raise ValueError("artifact filename resembles a secret")
    return name


def max_artifact_bytes(name: str) -> int:
    """Per-artifact byte ceiling for an artifact's namespace."""
    if name.startswith(WORKSPACE_ARTIFACT_PREFIX):
        return MAX_ATTACHMENT_BYTES
    return MAX_FILE_BYTES


@contextmanager
def stage_trajectory_artifacts(
    path: Path,
    *,
    source_id: str,
    redact: bool = True,
) -> Iterator[StagedTrajectoryArtifacts]:
    """Validate and redact JSONL artifacts without building their manifest."""
    source_id = validate_source_id(source_id)
    resolved = _resolve_input(path)
    if len(resolved.files) > MAX_ARTIFACTS:
        raise ValueError(
            f"trajectory capture exceeds {MAX_ARTIFACTS} artifact files: "
            f"{len(resolved.files)} files"
        )
    capture_bytes = sum(source.stat().st_size for source in resolved.files)
    if capture_bytes > MAX_JSONL_CAPTURE_BYTES:
        raise ValueError(
            f"trajectory capture exceeds {MAX_JSONL_CAPTURE_BYTES} bytes: "
            f"{capture_bytes} bytes"
        )
    for source in resolved.files:
        _validate_jsonl(source)

    with tempfile.TemporaryDirectory(prefix="benchflow-traj-") as temp_name:
        staging_dir = Path(temp_name)
        payloads: list[StagedFile] = []
        replacement_count = 0
        replacement_categories: Counter[str] = Counter()
        for source in resolved.files:
            relname = validate_artifact_name(f"trajectory/{source.name}")
            target = staging_dir / relname
            target.parent.mkdir(parents=True, exist_ok=True)
            if redact:
                replacements = _redact_jsonl(
                    source, target, categories=replacement_categories
                )
                replacement_count += replacements
            else:
                shutil.copyfile(source, target)
            payloads.append(
                _staged_file(
                    target,
                    relname,
                    "application/jsonl",
                    created_at=_source_created_at(source),
                )
            )

        payloads.sort(key=lambda item: item.relname)
        for payload in payloads:
            if payload.size_bytes > MAX_FILE_BYTES:
                raise ValueError(
                    f"staged trajectory file exceeds {MAX_FILE_BYTES} bytes: "
                    f"{payload.relname} ({payload.size_bytes} bytes)"
                )
        staged_capture_bytes = sum(payload.size_bytes for payload in payloads)
        if staged_capture_bytes > MAX_JSONL_CAPTURE_BYTES:
            raise ValueError(
                f"staged trajectory capture exceeds {MAX_JSONL_CAPTURE_BYTES} "
                f"bytes: {staged_capture_bytes} bytes"
            )
        traj_digest = _trajectory_digest(payloads)
        yield StagedTrajectoryArtifacts(
            source_id=source_id,
            traj_digest=traj_digest,
            files=tuple(payloads),
            ignored=resolved.ignored,
            redaction_replacements=replacement_count,
            _metadata_dir=resolved.metadata_dir,
            _staging_dir=staging_dir,
            _redact=redact,
            redaction_categories=redaction_breakdown(replacement_categories),
        )


@dataclass(frozen=True)
class WorkspaceAttachResult:
    """Outcome of trying to attach a workspace snapshot to staged artifacts."""

    artifacts: StagedTrajectoryArtifacts
    attached: StagedFile | None
    file_count: int
    excluded_count: int
    skipped_reason: str | None


def _workspace_archive_name(folder: Path) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", folder.name).strip("-._")
    return f"workspace/{sanitized[:120] or 'workspace'}.zip"


def _workspace_file_excluded(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in WORKSPACE_EXCLUDED_FILES)


def _collect_workspace_files(
    folder: Path, *, max_bytes: int
) -> tuple[list[Path], int, int] | str:
    """Walk the workspace, honoring exclusions; return a reason string to skip.

    Aborts the walk as soon as the included size passes ``max_bytes`` so a
    runaway workspace never costs a full scan, let alone a zip.
    """
    import os

    included: list[Path] = []
    total_bytes = 0
    excluded = 0
    for root, dirnames, filenames in os.walk(folder, followlinks=False):
        kept_dirs = []
        for dirname in dirnames:
            if dirname in WORKSPACE_EXCLUDED_DIRS or (Path(root, dirname).is_symlink()):
                excluded += 1
            else:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in filenames:
            candidate = Path(root, filename)
            if candidate.is_symlink() or _workspace_file_excluded(filename):
                excluded += 1
                continue
            try:
                size = candidate.stat().st_size
            except OSError:
                excluded += 1
                continue
            total_bytes += size
            if total_bytes > max_bytes:
                return f"workspace folder exceeds {max_bytes} bytes before compression"
            included.append(candidate)
            if len(included) > MAX_ATTACHMENT_FILES:
                return f"workspace folder exceeds {MAX_ATTACHMENT_FILES} files"
    if not included:
        return "workspace folder has no files to archive"
    return included, total_bytes, excluded


def attach_workspace_archive(
    artifacts: StagedTrajectoryArtifacts,
    folder: Path,
    *,
    max_bytes: int | None = None,
) -> WorkspaceAttachResult:
    """Zip a workspace folder into the staging area as ``workspace/*.zip``.

    The archive is written inside the staging temporary directory, so it is
    always deleted when staging exits — success, cancel, or crash alike.
    Contents are archived as-is (no redaction); callers exclude VCS internals,
    dependency trees, and secret-shaped filenames via the module exclusion
    lists and must surface that trade-off to the contributor.
    """
    import dataclasses
    import zipfile

    if max_bytes is None:
        max_bytes = MAX_ATTACHMENT_BYTES

    def skipped(reason: str) -> WorkspaceAttachResult:
        return WorkspaceAttachResult(
            artifacts=artifacts,
            attached=None,
            file_count=0,
            excluded_count=0,
            skipped_reason=reason,
        )

    resolved = folder.expanduser()
    if resolved.is_symlink() or not resolved.is_dir():
        return skipped(f"workspace folder not found: {resolved}")
    resolved = resolved.resolve()
    collected = _collect_workspace_files(resolved, max_bytes=max_bytes)
    if isinstance(collected, str):
        return skipped(collected)
    included, _total_bytes, excluded = collected

    relname = validate_artifact_name(_workspace_archive_name(resolved))
    target = artifacts._staging_dir / relname
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(included):
                archive.write(source, arcname=str(source.relative_to(resolved)))
    except (OSError, ValueError) as exc:
        target.unlink(missing_ok=True)
        return skipped(f"could not archive workspace folder: {exc}")
    if target.stat().st_size > max_bytes:
        target.unlink(missing_ok=True)
        return skipped(f"workspace archive exceeds {max_bytes} bytes")

    attached = _staged_file(target, relname, "application/zip")
    payloads = sorted((*artifacts.files, attached), key=lambda item: item.relname)
    updated = dataclasses.replace(
        artifacts,
        files=tuple(payloads),
        traj_digest=_trajectory_digest(payloads),
    )
    return WorkspaceAttachResult(
        artifacts=updated,
        attached=attached,
        file_count=len(included),
        excluded_count=excluded,
        skipped_reason=None,
    )


def finalize_trajectory_capture(
    artifacts: StagedTrajectoryArtifacts,
    *,
    uploaded_by: str | None = None,
    github_id: str | None = None,
    email: str | None = None,
    trajectory_report: Mapping[str, Any] | None = None,
) -> StagedCapture:
    """Bind contributor metadata and write the manifest for staged artifacts."""
    if uploaded_by is not None and len(uploaded_by) > MAX_UPLOADED_BY_LENGTH:
        raise ValueError(
            f"trajectory contributor label exceeds {MAX_UPLOADED_BY_LENGTH} characters"
        )
    contributor = _contributor(github_id, email)
    if trajectory_report is not None and contributor is None:
        raise ValueError("trajectory report metadata requires contributor metadata")
    payloads = list(artifacts.files)
    replacement_count = artifacts.redaction_replacements
    manifest = _build_manifest(
        source_id=artifacts.source_id,
        traj_digest=artifacts.traj_digest,
        payloads=payloads,
        metadata_dir=artifacts._metadata_dir,
        uploaded_by=uploaded_by,
        contributor=contributor,
        trajectory_report=trajectory_report,
        redact=artifacts._redact,
        replacement_count=replacement_count,
    )
    if artifacts._redact:
        redacted_manifest, manifest_replacements = redact_value_to_stability(manifest)
        if redacted_manifest["source_id"] != manifest["source_id"]:
            raise ValueError("source id contains a secret-like value")
        if redacted_manifest.get("contributor") != manifest.get("contributor"):
            raise ValueError("contributor metadata resembles a secret-like value")
        replacement_count += manifest_replacements
        redacted_manifest["redaction"]["replacements"] = replacement_count
        manifest = redacted_manifest
    manifest_path = artifacts._staging_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_size = manifest_path.stat().st_size
    if manifest_size > MAX_MANIFEST_BYTES:
        raise ValueError(
            f"trajectory manifest exceeds {MAX_MANIFEST_BYTES} bytes: "
            f"{manifest_size} bytes"
        )
    manifest_file = _staged_file(
        manifest_path,
        "manifest.json",
        "application/json",
    )
    return StagedCapture(
        source_id=artifacts.source_id,
        traj_digest=artifacts.traj_digest,
        files=(*artifacts.files, manifest_file),
        manifest=manifest,
        ignored=artifacts.ignored,
        artifact_redaction_replacements=artifacts.redaction_replacements,
        redaction_replacements=replacement_count,
        artifact_redaction_categories=artifacts.redaction_categories,
    )


@contextmanager
def stage_trajectory_capture(
    path: Path,
    *,
    source_id: str,
    redact: bool = True,
    uploaded_by: str | None = None,
    github_id: str | None = None,
    email: str | None = None,
    trajectory_report: Mapping[str, Any] | None = None,
) -> Iterator[StagedCapture]:
    """Validate and stage a complete capture without mutating its source."""
    with stage_trajectory_artifacts(
        path,
        source_id=source_id,
        redact=redact,
    ) as artifacts:
        yield finalize_trajectory_capture(
            artifacts,
            uploaded_by=uploaded_by,
            github_id=github_id,
            email=email,
            trajectory_report=trajectory_report,
        )


def _contributor(github_id: str | None, email: str | None) -> dict[str, str] | None:
    if github_id is None and email is None:
        return None
    if github_id is None or email is None:
        raise ValueError("GitHub ID and email must be provided together")
    return {
        "github_id": validate_github_id(github_id),
        "email": validate_email(email),
    }


def _resolve_input(path: Path) -> _ResolvedInput:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"trajectory path must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if resolved.is_file():
        if resolved.suffix.casefold() != ".jsonl":
            raise ValueError(f"trajectory file must end in .jsonl: {resolved}")
        return _ResolvedInput((resolved,), resolved.parent, ())
    if not resolved.is_dir():
        raise ValueError(f"trajectory path not found: {resolved}")

    trajectory_dir = resolved / "trajectory"
    if trajectory_dir.is_symlink():
        raise ValueError(
            f"trajectory directory must not be a symlink: {trajectory_dir}"
        )
    trial_dir = resolved if trajectory_dir.is_dir() else None
    payload_dir = trajectory_dir if trial_dir is not None else resolved
    entries = sorted(payload_dir.iterdir(), key=lambda item: item.name)
    for item in entries:
        if item.is_symlink() and item.suffix.casefold() == ".jsonl":
            raise ValueError(f"trajectory file must not be a symlink: {item}")
    files = tuple(
        item
        for item in entries
        if not item.is_symlink()
        and item.is_file()
        and item.suffix.casefold() == ".jsonl"
    )
    ignored = tuple(
        item.name
        for item in entries
        if item.is_file() and item.suffix.casefold() != ".jsonl"
    )
    if not files:
        raise ValueError(f"no .jsonl trajectory files found in {payload_dir}")
    return _ResolvedInput(files, trial_dir or resolved, ignored)


def _validate_jsonl(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(
            f"trajectory file exceeds {MAX_FILE_BYTES} bytes: {path} ({size} bytes)"
        )
    if size == 0:
        raise ValueError(f"trajectory JSONL is empty: {path}")
    records = 0
    with path.open("rb") as stream:
        line_number = 0
        while line_bytes := stream.readline(MAX_JSONL_RECORD_BYTES + 1):
            line_number += 1
            if len(line_bytes) > MAX_JSONL_RECORD_BYTES:
                raise ValueError(
                    f"{path}: line {line_number}: JSONL record exceeds "
                    f"{MAX_JSONL_RECORD_BYTES} bytes"
                )
            try:
                line = line_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{path}: trajectory JSONL must be UTF-8: {exc}"
                ) from exc
            if not line.strip():
                continue
            try:
                value = strict_json_loads(line)
            except (RecursionError, ValueError) as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}: line {line_number}: top-level record must be an object"
                )
            try:
                validate_json_complexity(value)
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_number}: {exc}") from exc
            records += 1
    if records == 0:
        raise ValueError(f"trajectory JSONL has no records: {path}")


def _redact_jsonl(
    source: Path, target: Path, *, categories: Counter[str] | None = None
) -> int:
    replacements = 0
    with (
        source.open(encoding="utf-8", newline="") as input_stream,
        target.open("w", encoding="utf-8", newline="") as output_stream,
    ):
        for line_number, line in enumerate(input_stream, start=1):
            body, newline = _split_newline(line)
            if not body.strip():
                output_stream.write(line)
                continue
            value = strict_json_loads(body)
            try:
                redacted, count = redact_value_to_stability(
                    value, categories=categories
                )
            except RecursionError as exc:  # defense in depth after complexity gate
                raise ValueError(
                    f"{source}: line {line_number}: trajectory JSONL nesting "
                    "exceeds the limit"
                ) from exc
            except ValueError as exc:
                raise ValueError(f"{source}: line {line_number}: {exc}") from exc
            # Redaction can normalize an already-safe legacy placeholder to the
            # BenchFlow marker without counting a newly discovered secret. Keep
            # the staged artifact and its manifest preview on the same canonical
            # representation even when the replacement count remains zero.
            if count or redacted != value:
                output_stream.write(
                    json.dumps(redacted, separators=(",", ":"), ensure_ascii=False)
                    + newline
                )
            else:
                output_stream.write(line)
            replacements += count
    return replacements


def _split_newline(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1]
    return line, ""


def validate_json_complexity(value: Any) -> None:
    """Bound container depth and nodes before recursive redaction."""
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError(f"JSON record exceeds {MAX_JSON_NODES} values")
        if isinstance(item, Mapping):
            if depth >= MAX_JSON_NESTING:
                raise ValueError(f"JSON nesting exceeds {MAX_JSON_NESTING} levels")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            if depth >= MAX_JSON_NESTING:
                raise ValueError(f"JSON nesting exceeds {MAX_JSON_NESTING} levels")
            stack.extend((child, depth + 1) for child in item)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _staged_file(
    path: Path,
    relname: str,
    content_type: str,
    *,
    created_at: datetime | None = None,
) -> StagedFile:
    return StagedFile(
        relname=relname,
        local_path=path,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        content_type=content_type,
        created_at=created_at,
    )


def _source_created_at(path: Path) -> datetime:
    stat = path.stat()
    timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
    return datetime.fromtimestamp(timestamp, UTC)


def _trajectory_digest(payloads: list[StagedFile]) -> str:
    digest_input = "\n".join(
        f"{item.relname}\t{item.sha256}"
        for item in sorted(payloads, key=lambda f: f.relname)
    )
    return hashlib.sha256(digest_input.encode()).hexdigest()


def _build_manifest(
    *,
    source_id: str,
    traj_digest: str,
    payloads: list[StagedFile],
    metadata_dir: Path | None,
    uploaded_by: str | None,
    contributor: dict[str, str] | None,
    trajectory_report: Mapping[str, Any] | None,
    redact: bool,
    replacement_count: int,
) -> dict[str, Any]:
    if trajectory_report is not None:
        schema_version = "1.2.0"
    elif contributor is not None:
        schema_version = "1.1.0"
    else:
        schema_version = "1.0.0"
    manifest = {
        "schema_version": schema_version,
        "kind": "bronze.trajectory",
        "created_at": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_id": source_id,
        "traj_digest": f"sha256:{traj_digest}",
        "uploaded_by": uploaded_by,
        "tool": {"name": "benchflow", "version": __version__},
        "run": _load_run_metadata(metadata_dir),
        "artifacts": [
            {"name": item.relname, "sha256": item.sha256, "bytes": item.size_bytes}
            for item in payloads
        ],
        "redaction": {"applied": redact, "replacements": replacement_count},
    }
    if contributor is not None:
        manifest["contributor"] = contributor
    if trajectory_report is not None:
        manifest["trajectory_report"] = dict(trajectory_report)
    return manifest


def _load_run_metadata(metadata_dir: Path | None) -> dict[str, Any]:
    result = _read_object(metadata_dir / "result.json") if metadata_dir else {}
    config = _read_object(metadata_dir / "config.json") if metadata_dir else {}
    raw_rewards = result.get("rewards")
    rewards: Mapping[str, Any] = raw_rewards if isinstance(raw_rewards, Mapping) else {}
    return {
        "agent": _first_scalar(result.get("agent"), config.get("agent")),
        "model": _first_scalar(result.get("model"), config.get("model")),
        "harness": _first_scalar(result.get("harness"), config.get("harness")),
        "skill_mode": _first_scalar(result.get("skill_mode"), config.get("skill_mode")),
        "task_id": _first_scalar(
            result.get("task_id"),
            result.get("task"),
            config.get("task_id"),
            config.get("task"),
        ),
        "reward": _first_scalar(result.get("reward"), rewards.get("reward")),
    }


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        return {}
    try:
        with path.open("rb") as stream:
            payload = stream.read(MAX_RUN_METADATA_BYTES + 1)
        if len(payload) > MAX_RUN_METADATA_BYTES:
            return {}
        value = strict_json_loads(payload)
    except (OSError, RecursionError, UnicodeDecodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_scalar(*values: Any) -> str | int | float | bool | None:
    for value in values:
        if value is None or isinstance(value, (dict, list)):
            continue
        if isinstance(value, (str, int, float, bool)):
            return value
    return None
