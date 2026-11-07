"""Bounded recovery of task-declared submission files (GH #948).

The task declares the absolute paths of its submission files in
``[verifier].submission_files``; the contract kind is ``submission-files-v1``.
The caller must stop solver writers before capture and bind the returned
manifest digest to its immutable solver/task/config receipt. This module never
captures a whole directory, only the declared regular files. It is stdlib-only
so its source can also run inside a recovery container.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import uuid
from pathlib import Path
from typing import Any, cast

CONTRACT_KIND = "submission-files-v1"
MAX_SUBMISSION_FILES = 8
MAX_PATH_CHARS = 512
# Verifier-owned and kernel paths a submission may never replace.
RESERVED_ROOTS = ("/dev", "/logs", "/proc", "/solution", "/sys", "/tests")
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 16384
_CHUNK = 65536
_GLOB_CHARS = frozenset("*?[]{}")


class SubmissionError(ValueError):
    """Submission evidence cannot be captured or admitted safely."""


def _validate_path(path: object) -> str:
    if not isinstance(path, str) or not path:
        raise SubmissionError("Submission paths must be non-empty strings")
    if len(path) > MAX_PATH_CHARS:
        raise SubmissionError(f"Submission path exceeds {MAX_PATH_CHARS} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        raise SubmissionError(f"Submission path has control characters: {path!r}")
    if _GLOB_CHARS.intersection(path):
        raise SubmissionError(f"Submission paths are literal, not globs: {path!r}")
    if not path.startswith("/"):
        raise SubmissionError(f"Submission path must be absolute: {path!r}")
    parts = path.split("/")[1:]
    if any(part in ("", ".", "..") for part in parts):
        raise SubmissionError(f"Submission path must be canonical: {path!r}")
    if len(parts) < 2:
        raise SubmissionError(
            f"Submission path must name a file inside a directory, not {path!r}"
        )
    if any(path == root or path.startswith(root + "/") for root in RESERVED_ROOTS):
        raise SubmissionError(f"Submission path is in a reserved directory: {path!r}")
    return path


def validate_contract(paths) -> tuple[str, ...]:
    """Validate a task's declared submission files; return them in order.

    The list must hold 1 to ``MAX_SUBMISSION_FILES`` distinct absolute,
    canonical, literal file paths below a top-level directory and outside the
    reserved directories. Whether each path is a regular file with no symlink
    components is checked at capture and restore time.
    """
    if not isinstance(paths, (list, tuple)) or not paths:
        raise SubmissionError("submission_files must be a non-empty list of paths")
    if len(paths) > MAX_SUBMISSION_FILES:
        raise SubmissionError(
            f"submission_files declares more than {MAX_SUBMISSION_FILES} files"
        )
    checked = tuple(_validate_path(path) for path in paths)
    if len(set(checked)) != len(checked):
        raise SubmissionError("submission_files lists a path more than once")
    return checked


def _directory(path: Path) -> int:
    path = Path(path)
    if not path.is_absolute() or any(part == ".." for part in path.parts):
        raise SubmissionError("Directory must be absolute and canonical")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _parent(root_fd: int, path: str) -> tuple[int, str]:
    parts = path.lstrip("/").split("/")
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd, parts[-1]
    except BaseException:
        os.close(fd)
        raise


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _state(parent: int, name: str):
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise SubmissionError(
            "Declared submission must be a regular file, not a link or special file"
        )
    return _identity(info)


def _copy(
    parent: int, name: str, output, *, expected: dict | None = None
) -> tuple[int, str]:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise SubmissionError(
                "Submission file is not regular or exceeds size limit"
            )
        digest = hashlib.sha256()
        size = 0
        while data := os.read(fd, _CHUNK):
            size += len(data)
            if size > MAX_FILE_BYTES:
                raise SubmissionError("Submission file exceeds size limit")
            digest.update(data)
            if output is not None:
                output.write(data)
        after = os.fstat(fd)
        if (
            _identity(before) != _identity(after)
            or _state(parent, name) != _identity(after)
            or size != before.st_size
        ):
            raise SubmissionError("Submission changed during capture/admission")
        sha256 = digest.hexdigest()
        if expected is not None and (
            size != expected["size"] or sha256 != expected["sha256"]
        ):
            raise SubmissionError("Submission size or SHA256 mismatch")
        return size, sha256
    finally:
        os.close(fd)


def _create_bundle(bundle: Path) -> int:
    parent = _directory(bundle.parent)
    try:
        os.mkdir(bundle.name, mode=0o700, dir_fd=parent)
        fd = os.open(
            bundle.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
        )
        try:
            os.mkdir("files", mode=0o700, dir_fd=fd)
            return fd
        except BaseException:
            os.close(fd)
            raise
    finally:
        os.close(parent)


def capture_submission(
    bundle: Path, *, paths, filesystem_root: Path = Path("/")
) -> str:
    """Create a fresh evidence bundle; return the SHA256 of its final manifest.

    Incomplete capture has no manifest and cannot be restored. Missing declared
    files stay missing; all other source files remain unread and untouched.
    """
    paths = validate_contract(paths)
    bundle_fd = _create_bundle(Path(bundle))
    root_fd = files_fd = None
    try:
        root_fd = _directory(filesystem_root)
        files_fd = os.open(
            "files", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=bundle_fd
        )
        entries = []
        total = 0
        for index, path in enumerate(paths):
            entry: dict[str, Any] = {
                "id": str(index),
                "path": path,
                "status": "missing",
            }
            try:
                parent, name = _parent(root_fd, path)
            except FileNotFoundError:
                entries.append(entry)
                continue
            try:
                if _state(parent, name) is not None:
                    target = os.open(
                        str(index),
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=files_fd,
                    )
                    with os.fdopen(target, "wb") as output:
                        size, digest = _copy(parent, name, output)
                        output.flush()
                        os.fsync(output.fileno())
                    total += size
                    if total > MAX_TOTAL_BYTES:
                        raise SubmissionError("Submission total exceeds size limit")
                    entry.update(
                        status="present",
                        type="regular",
                        blob=f"files/{index}",
                        size=size,
                        sha256=digest,
                    )
                current_parent, _ = _parent(root_fd, path)
                try:
                    original, current = os.fstat(parent), os.fstat(current_parent)
                    if (original.st_dev, original.st_ino) != (
                        current.st_dev,
                        current.st_ino,
                    ):
                        raise SubmissionError(
                            "Submission parent changed during capture"
                        )
                finally:
                    os.close(current_parent)
                entries.append(entry)
            finally:
                os.close(parent)
        manifest = {"schema_version": 1, "kind": CONTRACT_KIND, "files": entries}
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        fd = os.open(
            ".manifest.tmp",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=bundle_fd,
        )
        with os.fdopen(fd, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(
            ".manifest.tmp", "manifest.json", src_dir_fd=bundle_fd, dst_dir_fd=bundle_fd
        )
        os.fsync(bundle_fd)
        return hashlib.sha256(raw).hexdigest()
    finally:
        if files_fd is not None:
            os.close(files_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(bundle_fd)


def _manifest(bundle_fd: int, expected_digest: str, paths) -> dict:
    paths = validate_contract(paths)
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[a-f0-9]{64}", expected_digest
    ):
        raise SubmissionError("Expected immutable manifest SHA256 is required")
    fd = os.open(
        "manifest.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=bundle_fd
    )
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise SubmissionError("Manifest must be a regular file")
        raw = source.read(MAX_MANIFEST_BYTES + 1)
    if (
        len(raw) > MAX_MANIFEST_BYTES
        or hashlib.sha256(raw).hexdigest() != expected_digest
    ):
        raise SubmissionError("Manifest size or SHA256 mismatch")
    manifest: Any = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "kind", "files"}
        or manifest["schema_version"] != 1
        or manifest["kind"] != CONTRACT_KIND
    ):
        raise SubmissionError("Unsupported submission manifest")
    entries = manifest["files"]
    if not isinstance(entries, list) or len(entries) != len(paths):
        raise SubmissionError("Manifest must contain exactly the declared files")
    total = 0
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise SubmissionError("Invalid submission entry")
        entry = cast(dict[str, Any], item)
        if entry.get("id") != str(index) or entry.get("path") != paths[index]:
            raise SubmissionError("Manifest destinations differ from declared contract")
        fields = {"id", "path", "status"}
        if entry.get("status") == "present":
            fields |= {"type", "blob", "size", "sha256"}
            if (
                entry.get("type") != "regular"
                or entry.get("blob") != f"files/{index}"
                or type(entry.get("size")) is not int
                or not 0 <= entry["size"] <= MAX_FILE_BYTES
                or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
            ):
                raise SubmissionError("Invalid submission file metadata")
            total += entry["size"]
        elif entry.get("status") != "missing":
            raise SubmissionError("Invalid submission file status")
        if set(entry) != fields:
            raise SubmissionError("Unexpected submission metadata")
    if total > MAX_TOTAL_BYTES:
        raise SubmissionError("Submission total exceeds size limit")
    return manifest


def _bounded_names(fd: int, limit: int) -> set[str]:
    names = set()
    with os.scandir(fd) as entries:
        for entry in entries:
            names.add(entry.name)
            if len(names) > limit:
                raise SubmissionError("Unexpected files in submission bundle")
    return names


def _admit(bundle_fd: int, expected_digest: str, paths) -> tuple[dict, int]:
    manifest = _manifest(bundle_fd, expected_digest, paths)
    if _bounded_names(bundle_fd, 2) != {"manifest.json", "files"}:
        raise SubmissionError("Unexpected files in submission bundle")
    files_fd = os.open(
        "files", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=bundle_fd
    )
    try:
        present = {
            entry["id"] for entry in manifest["files"] if entry["status"] == "present"
        }
        if _bounded_names(files_fd, len(manifest["files"])) != present:
            raise SubmissionError("Submission blobs differ from manifest")
        for entry in manifest["files"]:
            if entry["status"] == "present":
                _copy(files_fd, entry["id"], None, expected=entry)
        return manifest, files_fd
    except BaseException:
        os.close(files_fd)
        raise


def validate_submission(bundle: Path, *, expected_manifest_sha256: str, paths) -> dict:
    """Admit only the declared files, with external manifest and per-file hashes."""
    bundle_fd = _directory(bundle)
    try:
        manifest, files_fd = _admit(bundle_fd, expected_manifest_sha256, paths)
        os.close(files_fd)
        return manifest
    finally:
        os.close(bundle_fd)


def restore_submission(
    bundle: Path,
    *,
    expected_manifest_sha256: str,
    paths,
    filesystem_root: Path = Path("/"),
) -> dict:
    """Validate all inputs/targets first, then atomically replace each target.

    Replacement is atomic per file, not a multi-file transaction. Run only in a
    fresh, quiescent sandbox; a failed attempt must never start its verifier.
    """
    bundle_fd = _directory(bundle)
    staged = []
    root_fd = files_fd = None
    try:
        root_fd = _directory(filesystem_root)
        manifest, files_fd = _admit(bundle_fd, expected_manifest_sha256, paths)
        # Check every destination before staging or changing any target.
        for entry in manifest["files"]:
            parent, name = _parent(root_fd, entry["path"])
            try:
                before = _state(parent, name)
            except BaseException:
                os.close(parent)
                raise
            staged.append([entry, parent, name, before, None])
        for item in staged:
            entry, parent, name, before, _ = item
            if entry["status"] == "present":
                temporary = ".benchflow-submission-" + uuid.uuid4().hex
                fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o644,
                    dir_fd=parent,
                )
                item[4] = temporary
                with os.fdopen(fd, "wb") as output:
                    _copy(files_fd, entry["id"], output, expected=entry)
                    output.flush()
                    os.fsync(output.fileno())
        # Recheck all target identities before the first replacement.
        for entry, parent, name, before, _temporary in staged:
            current_parent, _ = _parent(root_fd, entry["path"])
            try:
                original = os.fstat(parent)
                current = os.fstat(current_parent)
                if (original.st_dev, original.st_ino) != (
                    current.st_dev,
                    current.st_ino,
                ) or _state(parent, name) != before:
                    raise SubmissionError(
                        "Restore destination changed during admission"
                    )
            finally:
                os.close(current_parent)
        for item in staged:
            entry, parent, name, before, temporary = item
            if temporary:
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                item[4] = None
            elif before is not None:
                os.unlink(name, dir_fd=parent)
            os.fsync(parent)
        return manifest
    finally:
        for _, parent, _, _, temporary in staged:
            try:
                if temporary:
                    os.unlink(temporary, dir_fd=parent)
            finally:
                os.close(parent)
        if files_fd is not None:
            os.close(files_fd)
        if root_fd is not None:
            os.close(root_fd)
        os.close(bundle_fd)


if __name__ == "__main__":
    action, bundle, paths = sys.argv[1:4]
    if action == "capture":
        print(capture_submission(Path(bundle), paths=json.loads(paths)))
    elif action == "restore":
        restore_submission(
            Path(bundle), paths=json.loads(paths), expected_manifest_sha256=sys.argv[4]
        )
        print("ok")
    else:
        raise SubmissionError("Unknown operation")
