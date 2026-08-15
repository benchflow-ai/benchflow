"""Fail-closed validation of quarantined trajectory contributions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from benchflow.publish.redact import redact_value
from services.trajectory_upload.contract import (
    MAX_MANIFEST_BYTES,
    ContributionManifest,
)


class CaptureRejected(ValueError):
    """A quarantined capture failed an integrity or format gate."""


@dataclass(frozen=True)
class ValidatedCapture:
    manifest: ContributionManifest
    artifact_paths: dict[str, Path]
    manifest_bytes: bytes


def validate_local_capture(
    manifest_bytes: bytes,
    artifact_paths: dict[str, Path],
) -> ValidatedCapture:
    """Validate manifest, digests, strict JSONL shape, and secret redaction."""
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise CaptureRejected("manifest exceeds the 1 MiB limit")
    try:
        raw_manifest = json.loads(manifest_bytes)
        manifest = ContributionManifest.model_validate(raw_manifest)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise CaptureRejected(f"invalid manifest: {exc}") from exc

    expected_names = {artifact.name for artifact in manifest.artifacts}
    if set(artifact_paths) != expected_names:
        raise CaptureRejected("downloaded artifacts do not match the manifest")

    for artifact in manifest.artifacts:
        path = artifact_paths[artifact.name]
        if path.stat().st_size != artifact.bytes:
            raise CaptureRejected(f"size mismatch for {artifact.name}")
        if _sha256(path) != artifact.sha256:
            raise CaptureRejected(f"sha256 mismatch for {artifact.name}")
        _validate_and_scan_jsonl(path, artifact.name)
    return ValidatedCapture(
        manifest=manifest,
        artifact_paths=artifact_paths,
        manifest_bytes=manifest_bytes,
    )


def _validate_and_scan_jsonl(path: Path, relname: str) -> None:
    records = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CaptureRejected(
                        f"{relname}: line {line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise CaptureRejected(
                        f"{relname}: line {line_number}: top-level record must be an object"
                    )
                _reject_secrets(record, relname, line_number)
                records += 1
    except UnicodeDecodeError as exc:
        raise CaptureRejected(f"{relname}: trajectory JSONL must be UTF-8") from exc
    if records == 0:
        raise CaptureRejected(f"{relname}: trajectory JSONL has no records")


def _reject_secrets(record: dict, relname: str, line_number: int) -> None:
    _, replacements = redact_value(record)
    if replacements:
        raise CaptureRejected(
            f"{relname}: line {line_number}: secret-like value survived client redaction"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
