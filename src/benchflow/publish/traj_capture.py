"""Pure-local staging for trajectory contributions."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow import __version__
from benchflow.publish.redact import redact_value
from benchflow.trajectories.export_prime_sft import (
    PrimeSftTrajectoryJsonlError,
    load_llm_trajectory_jsonl,
)

MAX_FILE_BYTES = 1024**3
SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


@dataclass(frozen=True)
class StagedFile:
    relname: str
    local_path: Path
    sha256: str
    size_bytes: int
    content_type: str


@dataclass(frozen=True)
class StagedCapture:
    source_id: str
    traj_digest: str
    files: tuple[StagedFile, ...]
    manifest: dict[str, Any]
    ignored: tuple[str, ...]
    redaction_replacements: int


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
    invalid_segment = any(
        segment in {".", ".."} for segment in normalized.split("/")
    )
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


@contextmanager
def stage_trajectory_capture(
    path: Path,
    *,
    source_id: str,
    redact: bool = True,
    uploaded_by: str | None = None,
) -> Iterator[StagedCapture]:
    """Validate and stage a trajectory capture without mutating its source."""
    source_id = validate_source_id(source_id)
    resolved = _resolve_input(path)
    for source in resolved.files:
        _validate_jsonl(source)

    with tempfile.TemporaryDirectory(prefix="benchflow-traj-") as temp_name:
        staging_dir = Path(temp_name)
        payloads: list[StagedFile] = []
        replacement_count = 0
        for source in resolved.files:
            relname = f"trajectory/{source.name}"
            target = staging_dir / relname
            target.parent.mkdir(parents=True, exist_ok=True)
            if redact:
                replacements = _redact_jsonl(source, target)
                replacement_count += replacements
            else:
                shutil.copyfile(source, target)
            payloads.append(_staged_file(target, relname, "application/jsonl"))

        payloads.sort(key=lambda item: item.relname)
        traj_digest = _trajectory_digest(payloads)
        manifest = _build_manifest(
            source_id=source_id,
            traj_digest=traj_digest,
            payloads=payloads,
            metadata_dir=resolved.metadata_dir,
            uploaded_by=uploaded_by,
            redact=redact,
            replacement_count=replacement_count,
        )
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_file = _staged_file(
            manifest_path,
            "manifest.json",
            "application/json",
        )
        yield StagedCapture(
            source_id=source_id,
            traj_digest=traj_digest,
            files=(*payloads, manifest_file),
            manifest=manifest,
            ignored=resolved.ignored,
            redaction_replacements=replacement_count,
        )


def _resolve_input(path: Path) -> _ResolvedInput:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.suffix.casefold() != ".jsonl":
            raise ValueError(f"trajectory file must end in .jsonl: {resolved}")
        return _ResolvedInput((resolved,), resolved.parent, ())
    if not resolved.is_dir():
        raise ValueError(f"trajectory path not found: {resolved}")

    trial_dir = resolved if (resolved / "trajectory").is_dir() else None
    payload_dir = resolved / "trajectory" if trial_dir is not None else resolved
    entries = sorted(payload_dir.iterdir(), key=lambda item: item.name)
    files = tuple(
        item
        for item in entries
        if item.is_file() and item.suffix.casefold() == ".jsonl"
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
    if path.name == "llm_trajectory.jsonl":
        try:
            records = load_llm_trajectory_jsonl(path, strict=True)
        except PrimeSftTrajectoryJsonlError as exc:
            raise ValueError(str(exc)) from exc
        if not records:
            raise ValueError(f"trajectory JSONL has no records: {path}")
        return

    records = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}: line {line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path}: line {line_number}: top-level record must be an object"
                    )
                records += 1
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: trajectory JSONL must be UTF-8: {exc}") from exc
    if records == 0:
        raise ValueError(f"trajectory JSONL has no records: {path}")


def _redact_jsonl(source: Path, target: Path) -> int:
    replacements = 0
    with (
        source.open(encoding="utf-8", newline="") as input_stream,
        target.open("w", encoding="utf-8", newline="") as output_stream,
    ):
        for line in input_stream:
            body, newline = _split_newline(line)
            if not body.strip():
                output_stream.write(line)
                continue
            value = json.loads(body)
            redacted, count = redact_value(value)
            if count:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _staged_file(path: Path, relname: str, content_type: str) -> StagedFile:
    return StagedFile(
        relname=relname,
        local_path=path,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        content_type=content_type,
    )


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
    redact: bool,
    replacement_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
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


def _load_run_metadata(metadata_dir: Path | None) -> dict[str, Any]:
    result = _read_object(metadata_dir / "result.json") if metadata_dir else {}
    config = _read_object(metadata_dir / "config.json") if metadata_dir else {}
    raw_rewards = result.get("rewards")
    rewards: Mapping[str, Any] = (
        raw_rewards if isinstance(raw_rewards, Mapping) else {}
    )
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
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _first_scalar(*values: Any) -> str | int | float | bool | None:
    for value in values:
        if value is None or isinstance(value, (dict, list)):
            continue
        if isinstance(value, (str, int, float, bool)):
            return value
    return None
