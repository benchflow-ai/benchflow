"""Saved-attempt references: hash the evidence once, re-verify it at every boundary.

:func:`attempt_ref` reads a completed rollout's ``result.json``, the selected
output (a directory or one file) and the artifact a reviewer sees, and
returns a ``benchflow.saved-attempt-ref.v1``: every byte hashed, every
locator recorded. :func:`verify_attempt` hashes the same files again and
refuses a changed, missing, extra or substituted file. The bridge calls it
when an assessment is created, when its collector assignment is bound and
when a vote is imported, so evidence that changes in between stops the
assessment instead of being reviewed.

Nothing here writes: the rollout, its ``result.json`` and its outputs stay
exactly as they are.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benchflow.assessment import records as r
from benchflow.assessment.errors import EvidenceRejected, InvalidRecord

__all__ = [
    "attempt_ref",
    "frozen_file",
    "parse_runtime_ref",
    "read_regular",
    "verify_attempt",
    "verify_frozen_file",
]


def _regular(path: Path, what: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError:
        raise EvidenceRejected(f"{what} {path} is missing") from None
    if not stat.S_ISREG(info.st_mode):
        raise EvidenceRejected(
            f"{what} {path} is not a regular file (links are refused)"
        )
    return info


def file_sha256(path: Path, what: str) -> tuple[str, int]:
    """SHA-256 and size of a regular file, read through a no-follow handle."""
    info = _regular(path, what)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise EvidenceRejected(
            f"{what} {path} cannot be read: {exc.strerror}"
        ) from None
    with os.fdopen(fd, "rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        size = os.fstat(handle.fileno()).st_size
    if size != info.st_size:
        raise EvidenceRejected(f"{what} {path} changed while it was hashed")
    return digest, size


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(path))


def _members(output: Path) -> tuple[str, list[dict[str, Any]]]:
    """The output's kind and its members, sorted by UTF-8 path bytes."""
    try:
        info = output.lstat()
    except FileNotFoundError:
        raise EvidenceRejected(f"output {output} is missing") from None
    if stat.S_ISREG(info.st_mode):
        name = output.name
        _check_name(name, output)
        digest, size = file_sha256(output, "output file")
        return "file", [{"path": name, "sha256": digest, "bytes": size}]
    if not stat.S_ISDIR(info.st_mode):
        raise EvidenceRejected(f"output {output} is neither a file nor a directory")
    members: list[dict[str, Any]] = []
    for relative in _walk(output, ""):
        digest, size = file_sha256(output / relative, "output file")
        members.append({"path": relative, "sha256": digest, "bytes": size})
    if not members:
        raise EvidenceRejected(f"output directory {output} holds no files")
    members.sort(key=lambda m: r.utf8_sort_key(m["path"]))
    return "directory", members


def _check_name(name: str, where: Path) -> None:
    try:
        os.fsencode(name).decode("utf-8")
    except UnicodeDecodeError:
        raise EvidenceRejected(f"{where}: a file name is not valid UTF-8") from None


def _walk(root: Path, prefix: str) -> Iterable[str]:
    directory = root / prefix if prefix else root
    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            _check_name(entry.name, directory)
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_symlink():
                raise EvidenceRejected(f"output member {relative} is a symbolic link")
            if entry.is_dir(follow_symlinks=False):
                yield from _walk(root, relative)
            elif entry.is_file(follow_symlinks=False):
                yield relative
            else:
                raise EvidenceRejected(
                    f"output member {relative} is not a regular file"
                )


def _load_result(data: bytes, path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceRejected(f"{path} repeats the member {key!r}")
            result[key] = value
        return result

    try:
        result = json.loads(data, object_pairs_hook=unique)
    except EvidenceRejected:
        raise
    except (ValueError, RecursionError):
        raise EvidenceRejected(f"{path} is not readable JSON") from None
    if not isinstance(result, dict):
        raise EvidenceRejected(f"{path} is not a JSON object")
    return result


def read_regular(path: Path, what: str) -> bytes:
    info = _regular(path, what)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise EvidenceRejected(
            f"{what} {path} cannot be read: {exc.strerror}"
        ) from None
    with os.fdopen(fd, "rb") as handle:
        data = handle.read()
    if len(data) != info.st_size:
        raise EvidenceRejected(f"{what} {path} changed while it was read")
    return data


def _result_identity(path: Path) -> tuple[str, int, dict[str, Any]]:
    data = read_regular(path, "result")
    digest, size = hashlib.sha256(data).hexdigest(), len(data)
    result = _load_result(data, path)
    rollout = result.get("rollout_name")
    task_digest = result.get("task_digest")
    if not isinstance(rollout, str) or not rollout:
        raise EvidenceRejected(f"{path} names no rollout (rollout_name)")
    if not isinstance(task_digest, str):
        raise EvidenceRejected(
            f"{path} has no task_digest, so the attempt cannot be bound to a task"
        )
    try:
        r.prefixed_digest(task_digest, "task_digest")
    except InvalidRecord:
        raise EvidenceRejected(
            f"{path}: task_digest is not 'sha256:<64 hex>'"
        ) from None
    if not result.get("finished_at"):
        raise EvidenceRejected(f"{path} has no finished_at: not a completed rollout")
    return digest, size, result


def parse_runtime_ref(value: str) -> dict[str, str]:
    """``PROVIDER:RUNTIME_ID[:ROLE]`` (role solver, verifier or shared; default solver)."""
    provider, sep, rest = value.partition(":")
    if not sep or not rest:
        raise InvalidRecord(f"runtime {value!r} must be PROVIDER:RUNTIME_ID[:ROLE]")
    runtime_id, sep, role = rest.rpartition(":")
    if not sep or role not in r.RUNTIME_ROLES:
        runtime_id, role = rest, "solver"
    ref = {
        "provider": provider,
        "runtime_id": runtime_id,
        "role": role,
        "source": "declared",
    }
    r.name(ref["provider"], "runtime.provider")
    r.text(ref["runtime_id"], "runtime.runtime_id")
    return ref


def _verify_result_runtime(
    result: Mapping[str, Any], refs: Iterable[Mapping[str, Any]], rollout: str
) -> None:
    """A supplied reference must retain the sandbox named by its hashed result."""
    sandbox = result.get("sandbox_id")
    recorded = [ref for ref in refs if ref["source"] == "result.sandbox_id"]
    if sandbox is None or sandbox == "":
        valid = not recorded
    else:
        valid = (
            isinstance(sandbox, str)
            and len(recorded) == 1
            and recorded[0]["runtime_id"] == sandbox
        )
    if not valid:
        raise EvidenceRejected(
            f"{rollout}: runtime_refs do not retain exactly the result.sandbox_id "
            "recorded in result.json; declared runtimes cannot replace it"
        )


def attempt_ref(
    rollout_dir: str | Path,
    output: str | Path,
    review_artifact: str | Path,
    *,
    sandbox_provider: str | None = None,
    sandbox_role: str = "solver",
    runtimes: Iterable[Mapping[str, str]] = (),
    media_type: str | None = None,
) -> dict[str, Any]:
    """Hash one completed rollout into a ``benchflow.saved-attempt-ref.v1``.

    ``result.json`` must name the rollout and its ``task_digest`` and carry
    ``finished_at``. Its ``sandbox_id``, when present, becomes a runtime of
    ``sandbox_provider`` (required then); ``runtimes`` adds the others
    (verifier sandboxes, other attempts' allocations). An attempt with no
    runtime at all is refused: the release before human review must name
    every allocation.
    """
    rollout = _absolute(rollout_dir)
    result_path = rollout / "result.json"
    result_sha256, result_bytes, result = _result_identity(result_path)
    output_path = _absolute(output)
    kind, members = _members(output_path)
    review_path = _absolute(review_artifact)
    review_sha256, review_bytes = file_sha256(review_path, "review artifact")
    refs: list[dict[str, Any]] = []
    sandbox = result.get("sandbox_id")
    if isinstance(sandbox, str) and sandbox:
        if sandbox_provider is None:
            raise InvalidRecord(
                f"{result_path} records sandbox {sandbox}; name its provider (daytona, docker, ...)"
            )
        r.name(sandbox_provider, "sandbox_provider")
        r.enum(sandbox_role, "sandbox_role", r.RUNTIME_ROLES)
        refs.append(
            {
                "provider": sandbox_provider,
                "runtime_id": sandbox,
                "role": sandbox_role,
                "source": "result.sandbox_id",
            }
        )
    for runtime in runtimes:
        refs.append(dict(runtime))
    if not refs:
        raise EvidenceRejected(
            f"{result_path} records no sandbox and no runtime was declared; name every "
            "execution allocation so its release can be proven"
        )
    manifest = r.output_manifest_sha256(kind, members)
    review: dict[str, Any] = {
        "sha256": review_sha256,
        "bytes": review_bytes,
        "locator": str(review_path),
    }
    if media_type is not None:
        review["media_type"] = media_type
    ref = {
        "schema_version": r.ATTEMPT_SCHEMA,
        "attempt_id": r.attempt_id(
            result["rollout_name"], result["task_digest"], result_sha256, manifest
        ),
        "rollout_id": result["rollout_name"],
        "task_digest": result["task_digest"],
        "result_sha256": result_sha256,
        "output_manifest_sha256": manifest,
        "result": {"locator": str(result_path), "bytes": result_bytes},
        "output": {"kind": kind, "locator": str(output_path), "members": members},
        "review_artifact": review,
        "runtime_refs": refs,
    }
    r.check_attempt(ref, "attempt")
    _verify_result_runtime(result, refs, result["rollout_name"])
    return ref


def verify_attempt(ref: Mapping[str, Any]) -> None:
    """Hash the attempt's files again; refuse any change since the ref was made."""
    rollout = ref["rollout_id"]
    result_path = Path(ref["result"]["locator"])
    digest, size, result = _result_identity(result_path)
    if digest != ref["result_sha256"] or size != ref["result"]["bytes"]:
        raise EvidenceRejected(f"{rollout}: result.json changed ({result_path})")
    if (
        result.get("rollout_name") != rollout
        or result.get("task_digest") != ref["task_digest"]
    ):
        raise EvidenceRejected(f"{rollout}: result.json names another rollout or task")
    _verify_result_runtime(result, ref["runtime_refs"], rollout)
    output = Path(ref["output"]["locator"])
    kind, members = _members(output)
    recorded = ref["output"]["members"]
    if kind != ref["output"]["kind"] or members != recorded:
        found = {m["path"]: m["sha256"] for m in members}
        wanted = {m["path"]: m["sha256"] for m in recorded}
        missing = sorted(set(wanted) - set(found))
        extra = sorted(set(found) - set(wanted))
        changed = sorted(p for p in set(found) & set(wanted) if found[p] != wanted[p])
        detail = "; ".join(
            f"{label}: {', '.join(paths[:5])}{' ...' if len(paths) > 5 else ''}"
            for label, paths in (
                ("missing", missing),
                ("extra", extra),
                ("changed", changed),
            )
            if paths
        )
        raise EvidenceRejected(
            f"{rollout}: output changed ({detail or 'kind differs'})"
        )
    if r.output_manifest_sha256(kind, members) != ref["output_manifest_sha256"]:
        raise EvidenceRejected(f"{rollout}: output manifest digest differs")
    review = ref["review_artifact"]
    review_digest, review_size = file_sha256(Path(review["locator"]), "review artifact")
    if review_digest != review["sha256"] or review_size != review["bytes"]:
        raise EvidenceRejected(
            f"{rollout}: review artifact changed ({review['locator']})"
        )


def frozen_file(path: str | Path, *, file_id: str, version: str) -> dict[str, Any]:
    """``{id, version, sha256, bytes, locator}`` of a frozen rubric or protocol file."""
    absolute = _absolute(path)
    digest, size = file_sha256(absolute, "frozen file")
    ref = {
        "id": file_id,
        "version": version,
        "sha256": digest,
        "bytes": size,
        "locator": str(absolute),
    }
    return r.check_frozen_file(ref, "frozen_file")


def verify_frozen_file(ref: Mapping[str, Any], what: str) -> None:
    digest, size = file_sha256(Path(ref["locator"]), what)
    if digest != ref["sha256"] or size != ref["bytes"]:
        raise EvidenceRejected(f"the frozen {what} {ref['locator']} changed")
