"""Durable workspace evidence shared by solver and reviewer rollouts.

The caller must stop agent writes before capture. A successful capture is an
all-or-nothing directory publication: an archive digest validates transport,
then a file manifest validates the extracted tree and subsequent uploads.
Only the resolved workspace is captured; links cannot import other VM files.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import logging
import os
import posixpath
import shlex
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchflow.agents.credentials import CREDENTIAL_EVIDENCE_PATHS
from benchflow.sandbox.protocol import SANDBOX_RUNTIME_STATE_PATHS, Sandbox
from benchflow.task.config import ArtifactConfig

logger = logging.getLogger(__name__)


class EvidenceError(RuntimeError):
    """Required evidence could not be captured or verified completely."""


class EvidenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    original_path: str
    kind: Literal["file", "directory", "symlink"]
    size: int = Field(ge=0)
    sha256: str | None = None
    link_target: str | None = None
    original_link_target: str | None = None
    # Permission bits the solver left (files and directories). The host copy
    # and the reviewer's copy are normalised to 0644/0755; a restore for
    # verification (regrade, verifier recovery) puts these back.
    mode: int | None = Field(default=None, ge=0, le=0o777)


class EvidenceExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_path: str
    reason: Literal[
        "credential",
        "sandbox_runtime",
        "special_file",
        "symlink_escape",
        "task_exclude",
    ]
    # The link's own text, recorded for a symlink the bundle cannot keep.
    link_target: str | None = None

    @model_validator(mode="after")
    def _link_target_names_escaping_links(self) -> Self:
        if (self.reason == "symlink_escape") != (self.link_target is not None):
            raise ValueError("Only a symlink_escape exclusion records a link target")
        return self


class ArtifactEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    destination: str | None = None
    exclude: tuple[str, ...] = ()
    status: Literal["captured", "missing", "in_workspace"]
    bundle_path: str | None = Field(default=None, pattern=r"^artifacts/[0-9]{4,}$")

    @model_validator(mode="after")
    def _captured_bundle(self) -> Self:
        if (self.status == "captured") != (self.bundle_path is not None):
            raise ValueError("Only captured artifacts have a bundle path")
        return self


class EvidenceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    capture_status: Literal["complete"] = "complete"
    workspace: str
    archive_sha256: str
    entries: tuple[EvidenceEntry, ...]
    exclusions: tuple[EvidenceExclusion, ...] = ()
    artifacts: tuple[ArtifactEvidence, ...] = ()


class _CaptureReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    archive: str = Field(
        pattern=r"^/(?:var/)?tmp/benchflow-evidence-[A-Za-z0-9_-]+\.tar$"
    )
    workspace: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    exclusions: tuple[EvidenceExclusion, ...] = ()


# Runs in both Docker and Daytona without requiring BenchFlow in the image.
# Enumeration is explicit so nothing is silently omitted by a tar command.
# Sockets, devices and FIFOs hold no bytes to preserve, provider runtime state
# is harness-owned, and a symlink leaving the workspace (an agent CLI's helper
# link, say) would import another VM file, so each is recorded as a manifest
# exclusion, a link with its target text. A link is kept only when both its
# resolution and its text stay inside the root, the rule the host extraction,
# validate_workspace and the reviewer's admission check apply to it; a link
# loop has no target to keep. Quota overflows and concurrent changes still
# fail the capture.
_CAPTURE_SCRIPT = r"""
import fnmatch, hashlib, json, os, pathlib, posixpath, stat, sys, tarfile, tempfile
source = pathlib.Path(sys.argv[1]).resolve()
system_roots = {pathlib.Path(path).resolve() for path in ("/", "/proc", "/sys", "/dev", "/etc", "/run", "/home")}
if source in system_roots:
    raise ValueError("refusing to capture a system directory as evidence")
source.stat()
root = source if source.is_dir() else source.parent
max_bytes, max_entries = map(int, sys.argv[2:4])
rules = json.loads(sys.argv[4])
paths, exclusions, total = [], [], 0
def record(path, reason, **details):
    exclusions.append({"original_path": path.as_posix(), "reason": reason, **details})
    if len(paths) + len(exclusions) > max_entries:
        raise ValueError("workspace evidence exceeds configured capture limits")
def kept_link(path, target):
    try:
        os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        pass
    except OSError:
        return False
    if not path.resolve().is_relative_to(root):
        return False
    if target.startswith("/"):
        try:
            return ".." not in pathlib.PurePosixPath(target).relative_to(root).parts
        except ValueError:
            return False
    parent = posixpath.dirname(path.relative_to(root).as_posix())
    return ".." not in posixpath.normpath(posixpath.join(parent, target)).split("/")
def component(path, rule):
    return path.endswith("/" + rule) or "/" + rule + "/" in path
def excluded(path):
    absolute = path.as_posix()
    relative = path.relative_to(root).as_posix()
    credential = any(component(absolute, rule) for rule in rules["credentials"])
    credential |= any(path == pathlib.Path(rule) or path.is_relative_to(rule) for rule in rules["paths"])
    reason = "credential" if credential else None
    # Relative to the captured root: runtime state below the root is dropped,
    # but a root that itself lies under such a directory is still captured.
    if reason is None and any(component("/" + relative, rule) for rule in rules["sandbox_runtime"]):
        reason = "sandbox_runtime"
    if reason is None and any(fnmatch.fnmatchcase(relative, rule) or fnmatch.fnmatchcase(path.name, rule) for rule in rules["exclude"]):
        reason = "task_exclude"
    if reason:
        record(path, reason)
        return True
    return False
def scan_error(error):
    raise error
walk = os.walk(root, followlinks=False, onerror=scan_error) if source.is_dir() else [(root, [], [source.name])]
if excluded(source):
    walk = []
for directory, dirs, files in walk:
    for name in sorted(dirs + files):
        path = pathlib.Path(directory) / name
        if excluded(path):
            if name in dirs:
                dirs.remove(name)
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            if not kept_link(path, target):
                record(path, "symlink_escape", link_target=target)
                continue
        elif not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            record(path, "special_file")
            continue
        total += info.st_size if stat.S_ISREG(info.st_mode) else 0
        paths.append((path, info))
        if total > max_bytes or len(paths) + len(exclusions) > max_entries:
            raise ValueError("workspace evidence exceeds configured capture limits")
temp_root = next((p for p in ("/tmp", "/var/tmp")
                  if not pathlib.Path(p).resolve().is_relative_to(root)), None)
if temp_root is None:
    raise ValueError("no temporary directory outside the evidence workspace")
fd, archive_path = tempfile.mkstemp(prefix="benchflow-evidence-", suffix=".tar", dir=temp_root)
os.close(fd)
try:
    with tarfile.open(archive_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for path, before in paths:
            # Store hard-linked regular files independently without following
            # symbolic links. The receiving tree has no archive hardlinks.
            member = archive.gettarinfo(path, arcname=path.relative_to(root).as_posix())
            if stat.S_ISREG(before.st_mode):
                member.type, member.linkname, member.size = tarfile.REGTYPE, "", before.st_size
                with path.open("rb") as stream:
                    archive.addfile(member, stream)
            else:
                archive.addfile(member)
            after = path.lstat()
            if (before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns) != (
                    after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns):
                raise ValueError("workspace changed during evidence capture: " + str(path))
    digest = hashlib.sha256()
    with open(archive_path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    print(json.dumps({"archive": archive_path, "workspace": str(root), "sha256": digest.hexdigest(), "exclusions": exclusions}))
except BaseException:
    os.unlink(archive_path)
    raise
"""


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise EvidenceError(f"Unsafe evidence path: {value!r}")
    return path


def _link_target(member: tarfile.TarInfo, workspace: str) -> str:
    target = PurePosixPath(member.linkname)
    if target.is_absolute():
        try:
            target = target.relative_to(workspace)
        except ValueError as exc:
            raise EvidenceError(
                f"Evidence symlink escapes workspace: {member.name}"
            ) from exc
        if str(target) != ".":
            _relative_path(str(target))
        return posixpath.relpath(str(target), str(PurePosixPath(member.name).parent))
    normalized = posixpath.normpath(
        posixpath.join(posixpath.dirname(member.name), str(target))
    )
    if normalized != ".":
        _relative_path(normalized)
    return str(target)


def _extract_archive(
    archive_path: Path,
    destination: Path,
    workspace: str,
    *,
    max_bytes: int,
    max_entries: int,
) -> tuple[EvidenceEntry, ...]:
    """Extract ordinary entries first and links last, never using extractall."""
    entries: list[EvidenceEntry] = []
    with tarfile.open(archive_path, "r:") as archive:
        members: list[tarfile.TarInfo] = []
        total = 0
        for member in archive:
            members.append(member)
            total += member.size
            if member.size < 0 or len(members) > max_entries or total > max_bytes:
                raise EvidenceError(
                    "Workspace evidence exceeds configured capture limits"
                )
        names: set[PurePosixPath] = set()
        links: set[PurePosixPath] = set()
        for member in members:
            name = _relative_path(member.name)
            if name in names:
                raise EvidenceError(f"Duplicate evidence path: {member.name}")
            names.add(name)
            if member.issym():
                _link_target(member, workspace)
                links.add(name)
            elif not (member.isfile() or member.isdir()):
                raise EvidenceError(f"Unsupported evidence entry: {member.name}")
        for name in names:
            if any(parent in links for parent in name.parents):
                raise EvidenceError(f"Evidence entry is below a symlink: {name}")
        for member in sorted(
            members,
            key=lambda item: (item.issym(), len(PurePosixPath(item.name).parts)),
        ):
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            original_path = str(PurePosixPath(workspace) / member.name)
            if member.isdir():
                path.mkdir(exist_ok=True)
                entry = EvidenceEntry(
                    path=member.name,
                    original_path=original_path,
                    kind="directory",
                    size=0,
                    mode=member.mode & 0o777,
                )
            elif member.issym():
                target = _link_target(member, workspace)
                path.symlink_to(target)
                entry = EvidenceEntry(
                    path=member.name,
                    original_path=original_path,
                    kind="symlink",
                    size=0,
                    link_target=target,
                    original_link_target=member.linkname,
                )
            else:
                source = archive.extractfile(member)
                if source is None:
                    raise EvidenceError(f"Missing archive file: {member.name}")
                with source, path.open("xb") as output:
                    shutil.copyfileobj(source, output)
                path.chmod(0o755 if member.mode & 0o111 else 0o644)
                entry = EvidenceEntry(
                    path=member.name,
                    original_path=original_path,
                    kind="file",
                    size=member.size,
                    sha256=_digest(path),
                    mode=member.mode & 0o777,
                )
            entries.append(entry)
    return tuple(sorted(entries, key=lambda entry: entry.path))


def validate_workspace(workspace: Path, manifest: EvidenceManifest) -> None:
    """Verify exact inventory, file bytes and links against a trusted manifest."""
    excluded = {exclusion.original_path for exclusion in manifest.exclusions}
    if any(entry.original_path in excluded for entry in manifest.entries):
        raise EvidenceError("Evidence manifest both captures and excludes a path")
    root = workspace.resolve(strict=True)
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*")}
    expected = {entry.path for entry in manifest.entries}
    if len(expected) != len(manifest.entries) or actual.keys() != expected:
        raise EvidenceError("Workspace inventory does not match the evidence manifest")
    for entry in manifest.entries:
        _relative_path(entry.path)
        path = actual[entry.path]
        mode = path.lstat().st_mode
        if entry.kind == "symlink":
            try:
                valid = (
                    stat.S_ISLNK(mode)
                    and os.readlink(path) == entry.link_target
                    and path.resolve().is_relative_to(root)
                )
            except (OSError, RuntimeError):
                valid = False
        elif entry.kind == "directory":
            valid = stat.S_ISDIR(mode)
        else:
            valid = (
                stat.S_ISREG(mode)
                and path.stat().st_size == entry.size
                and _digest(path) == entry.sha256
            )
        if not valid:
            raise EvidenceError(f"Workspace evidence mismatch: {entry.path}")


def python_missing(result: object) -> bool:
    """Whether a ``python3 ...`` exec failed because there is no python3."""
    if getattr(result, "return_code", 0) == 127:
        return True
    text = f"{getattr(result, 'stderr', '') or ''}{getattr(result, 'stdout', '') or ''}"
    return "python3" in text and (
        "not found" in text or "No such file or directory" in text
    )


# Python-free capture for images without python3 (Harbor's ubuntu:24.04
# examples): the sandbox only runs tar; enumeration, exclusions, limits and
# link checks happen on the host, on the downloaded archive.
_TAR_CAPTURE_SCRIPT = r"""
set -eu
src=$1
if [ ! -e "$src" ] && [ ! -L "$src" ]; then echo "no such path: $src" >&2; exit 2; fi
if [ -d "$src" ]; then
  root=$(cd "$src" && pwd -P); name=.
else
  real=$(readlink -f "$src" 2>/dev/null || echo "$src")
  root=$(cd "$(dirname "$real")" && pwd -P); name=$(basename "$real")
fi
case "$root" in
  /|/proc|/sys|/dev|/etc|/run|/home)
    echo "refusing to capture a system directory as evidence" >&2; exit 2;;
esac
tmp=/tmp
case "$root" in /tmp|/tmp/*) tmp=/var/tmp;; esac
out=$(mktemp "$tmp/benchflow-evidence-XXXXXXXX" 2>/dev/null) || {
  out="$tmp/benchflow-evidence-$$"; : > "$out"; }
if ! tar -cf "$out" -C "$root" "$name"; then rm -f "$out"; exit 1; fi
echo "$out"
echo "$root"
if command -v sha256sum >/dev/null 2>&1; then sha256sum "$out" | cut -d' ' -f1; else echo; fi
"""

_SHELL_ARTIFACT_PROBE = r"""
root=$1; shift
for s in "$@"; do
  case "$s" in /*) p=$s;; *) p=$root/$s;; esac
  r=$(readlink -f "$p" 2>/dev/null) || r=$p
  [ -n "$r" ] || r=$p
  if [ -e "$p" ]; then e=1; else e=0; fi
  printf '%s\t%s\n' "$e" "$r"
done
"""


def _component(path: str, rule: str) -> bool:
    return path.endswith("/" + rule) or "/" + rule + "/" in path


ExclusionReason = Literal[
    "credential", "sandbox_runtime", "special_file", "symlink_escape", "task_exclude"
]

# Linux's limit on symlinks followed while resolving one path (ELOOP).
_MAX_LINK_HOPS = 40


def _kept_tar_link(name: str, target: str, links: dict[str, str], root: str) -> bool:
    """The capture script's ``kept_link`` rule, applied to a tar listing.

    A link is kept only when its target text and its resolution both stay
    inside the root. The resolution follows the archive's own links, as
    ``os.path.realpath`` would in the sandbox; a loop is not kept.
    """
    if target.startswith("/"):
        try:
            if ".." in PurePosixPath(target).relative_to(root).parts:
                return False
        except ValueError:
            return False
    else:
        joined = posixpath.join(posixpath.dirname(name), target)
        if ".." in posixpath.normpath(joined).split("/"):
            return False
    pending = list(PurePosixPath(name).parts)
    resolved: list[str] = []
    hops = 0
    while pending:
        part = pending.pop(0)
        if part == "..":
            if not resolved:
                return False
            resolved.pop()
            continue
        link = links.get("/".join([*resolved, part]))
        if link is None:
            resolved.append(part)
            continue
        hops += 1
        if hops > _MAX_LINK_HOPS:
            return False
        if link.startswith("/"):
            try:
                inside = PurePosixPath(link).relative_to(root)
            except ValueError:
                return False
            resolved = []
            pending = [*inside.parts, *pending]
        else:
            pending = [*PurePosixPath(link).parts, *pending]
    return True


def _exclusion_reason(
    absolute: PurePosixPath, relative: PurePosixPath, rules: dict
) -> ExclusionReason | None:
    """The capture script's ``excluded()`` rules, applied on the host."""
    text = absolute.as_posix()
    if any(_component(text, rule) for rule in rules["credentials"]) or any(
        absolute == PurePosixPath(rule) or absolute.is_relative_to(rule)
        for rule in rules["paths"]
    ):
        return "credential"
    if any(
        _component("/" + relative.as_posix(), rule) for rule in rules["sandbox_runtime"]
    ):
        return "sandbox_runtime"
    if any(
        fnmatch.fnmatchcase(relative.as_posix(), rule)
        or fnmatch.fnmatchcase(absolute.name, rule)
        for rule in rules["exclude"]
    ):
        return "task_exclude"
    return None


def _normalize_tar_capture(
    raw: Path,
    dest: Path,
    root: str,
    rules: dict,
    *,
    max_bytes: int,
    max_entries: int,
) -> tuple[EvidenceExclusion, ...]:
    """Rewrite a ``tar -C root .`` archive into capture-script form.

    Drops the root entry and ``./`` prefixes, applies the exclusion rules,
    records special files and links the capture script would not keep
    (``symlink_escape``, with their target text), stores hard links as
    independent regular files and enforces the limits.
    """
    exclusions: list[EvidenceExclusion] = []
    excluded_dirs: list[str] = []
    kept = 0
    total = 0

    def over() -> bool:
        return total > max_bytes or kept + len(exclusions) > max_entries

    with (
        tarfile.open(raw, "r:") as source,
        tarfile.open(dest, "x", format=tarfile.PAX_FORMAT) as archive,
    ):
        members = source.getmembers()
        links = {
            posixpath.normpath(member.name): member.linkname
            for member in members
            if member.issym()
        }
        for member in members:
            name = posixpath.normpath(member.name)
            if name in (".", ""):
                continue
            relative = _relative_path(name)
            if any(name == d or name.startswith(d + "/") for d in excluded_dirs):
                continue
            absolute = PurePosixPath(root) / relative
            reason: ExclusionReason | None = _exclusion_reason(
                absolute, relative, rules
            )
            link_target = None
            if (
                reason is None
                and member.issym()
                and not _kept_tar_link(name, member.linkname, links, root)
            ):
                reason, link_target = "symlink_escape", member.linkname
            if reason is None and not (
                member.isfile() or member.isdir() or member.issym() or member.islnk()
            ):
                reason = "special_file"
            if reason is not None:
                exclusions.append(
                    EvidenceExclusion(
                        original_path=absolute.as_posix(),
                        reason=reason,
                        link_target=link_target,
                    )
                )
                if member.isdir():
                    excluded_dirs.append(name)
                if over():
                    raise EvidenceError(
                        "Workspace evidence exceeds configured capture limits"
                    )
                continue
            copy = tarfile.TarInfo(name)
            copy.mode, copy.mtime = member.mode, member.mtime
            if member.isfile() or member.islnk():
                data = source.extractfile(member)
                if data is None:
                    raise EvidenceError(f"Missing archive file: {name}")
                payload = data.read()
                copy.type, copy.size = tarfile.REGTYPE, len(payload)
                total += copy.size
                kept += 1
                if over():
                    raise EvidenceError(
                        "Workspace evidence exceeds configured capture limits"
                    )
                archive.addfile(copy, io.BytesIO(payload))
                continue
            if member.issym():
                copy.type, copy.linkname = tarfile.SYMTYPE, member.linkname
            else:
                copy.type = tarfile.DIRTYPE
            kept += 1
            if over():
                raise EvidenceError(
                    "Workspace evidence exceeds configured capture limits"
                )
            archive.addfile(copy)
    return tuple(exclusions)


async def _capture_workspace_with_tar(
    env: Sandbox,
    workspace: str,
    destination: Path,
    *,
    python_error: str,
    rules: dict,
    max_bytes: int,
    max_entries: int,
    timeout_sec: int,
) -> EvidenceManifest:
    """:func:`capture_workspace` for sandboxes without ``python3``."""
    result = await env.exec(
        shlex.join(["sh", "-c", _TAR_CAPTURE_SCRIPT, "capture", workspace]),
        user="root",
        timeout_sec=timeout_sec,
    )
    lines = (result.stdout or "").splitlines()
    if result.return_code != 0 or len(lines) < 2:
        raise EvidenceError(
            f"Workspace capture failed: {python_error[-1000:]}; tar fallback: "
            f"{(result.stderr or result.stdout or '')[-1000:]}"
        )
    remote, root = lines[0].strip(), lines[1].strip()
    remote_sha = lines[2].strip() if len(lines) > 2 else ""
    try:
        with tempfile.TemporaryDirectory(
            prefix=".evidence-", dir=destination.parent
        ) as temp:
            staging = Path(temp)
            raw = staging / "raw.tar"
            await env.download_file(remote, raw)
            digest = _digest(raw)
            if remote_sha and remote_sha != digest:
                raise EvidenceError("Workspace archive digest changed during download")
            archive_path = staging / "workspace.tar"
            try:
                exclusions = _normalize_tar_capture(
                    raw,
                    archive_path,
                    root,
                    rules,
                    max_bytes=max_bytes,
                    max_entries=max_entries,
                )
            except tarfile.TarError as exc:
                raise EvidenceError(f"Unreadable workspace archive: {exc}") from exc
            bundle = staging / "bundle"
            tree = bundle / "workspace"
            tree.mkdir(parents=True)
            entries = _extract_archive(
                archive_path,
                tree,
                root,
                max_bytes=max_bytes,
                max_entries=max_entries,
            )
            manifest = EvidenceManifest(
                workspace=root,
                archive_sha256=digest,
                entries=entries,
                exclusions=exclusions,
            )
            validate_workspace(tree, manifest)
            (bundle / "manifest.json").write_text(
                manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            bundle.rename(destination)
            return manifest
    finally:
        try:
            await env.exec(
                shlex.join(["rm", "-f", "--", remote]), user="root", timeout_sec=30
            )
        except Exception:
            logger.warning(
                "Could not remove temporary evidence archive %s", remote, exc_info=True
            )


def _shell_artifact_probe(line: str) -> _ArtifactProbe:
    exists, _, source = line.partition("\t")
    return _ArtifactProbe(source=source, exists=exists == "1")


async def capture_workspace(
    env: Sandbox,
    workspace: str,
    destination: Path,
    *,
    max_bytes: int = 20 * 1024**3,
    max_entries: int = 200_000,
    timeout_sec: int = 600,
    exclude: Sequence[str] = (),
    excluded_paths: Sequence[str] = (),
) -> EvidenceManifest:
    """Capture a stopped agent's complete workspace as an immutable bundle.

    ``destination`` must not exist. It receives ``workspace/`` and
    ``manifest.json`` together after successful validation. Capture limits are
    hard errors: this API never labels a truncated tree complete. Credential
    files, provider runtime state (``SANDBOX_RUNTIME_STATE_PATHS``) and
    sockets, FIFOs or device nodes stay out of the bundle, each recorded in
    ``manifest.exclusions``.
    """
    if destination.exists() or destination.is_symlink():
        raise EvidenceError(f"Evidence destination already exists: {destination}")
    if min(max_bytes, max_entries, timeout_sec) <= 0:
        raise ValueError("Evidence capture limits must be positive")
    if any(not PurePosixPath(path).is_absolute() for path in excluded_paths):
        raise ValueError("Credential exclusion paths must be absolute sandbox paths")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rules = json.dumps(
        {
            "credentials": CREDENTIAL_EVIDENCE_PATHS,
            "sandbox_runtime": SANDBOX_RUNTIME_STATE_PATHS,
            "exclude": list(exclude),
            "paths": list(excluded_paths),
        }
    )
    command = shlex.join(
        [
            "python3",
            "-c",
            _CAPTURE_SCRIPT,
            workspace,
            str(max_bytes),
            str(max_entries),
            rules,
        ]
    )
    result = await env.exec(command, user="root", timeout_sec=timeout_sec)
    if result.return_code != 0 and python_missing(result):
        return await _capture_workspace_with_tar(
            env,
            workspace,
            destination,
            python_error=(result.stderr or result.stdout or "").strip(),
            rules=json.loads(rules),
            max_bytes=max_bytes,
            max_entries=max_entries,
            timeout_sec=timeout_sec,
        )
    if result.return_code != 0:
        raise EvidenceError(
            f"Workspace capture failed: {(result.stderr or result.stdout or '')[-2000:]}"
        )
    try:
        transfer = _CaptureReceipt.model_validate_json(result.stdout or "")
    except ValueError as exc:
        raise EvidenceError(
            "Sandbox did not return a valid evidence capture receipt"
        ) from exc
    try:
        with tempfile.TemporaryDirectory(
            prefix=".evidence-", dir=destination.parent
        ) as temp:
            staging = Path(temp)
            archive_path = staging / "workspace.tar"
            await env.download_file(transfer.archive, archive_path)
            if _digest(archive_path) != transfer.sha256:
                raise EvidenceError("Workspace archive digest changed during download")
            bundle = staging / "bundle"
            tree = bundle / "workspace"
            tree.mkdir(parents=True)
            entries = _extract_archive(
                archive_path,
                tree,
                transfer.workspace,
                max_bytes=max_bytes,
                max_entries=max_entries,
            )
            manifest = EvidenceManifest(
                workspace=transfer.workspace,
                archive_sha256=transfer.sha256,
                entries=entries,
                exclusions=transfer.exclusions,
            )
            validate_workspace(tree, manifest)
            (bundle / "manifest.json").write_text(
                manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            bundle.rename(destination)
            return manifest
    finally:
        # Transport cleanup cannot invalidate an already committed snapshot or
        # replace the original capture failure with a less useful exception.
        try:
            cleanup = await env.exec(
                shlex.join(["rm", "-f", "--", transfer.archive]),
                user="root",
                timeout_sec=30,
            )
            if cleanup.return_code:
                logger.warning(
                    "Could not remove temporary evidence archive %s", transfer.archive
                )
        except Exception:
            logger.warning(
                "Could not remove temporary evidence archive %s",
                transfer.archive,
                exc_info=True,
            )


# The receiving sandbox may not have BenchFlow/Pydantic. This stdlib-only
# admission check verifies the same public manifest after upload, before any
# reviewer sees evidence, with validate_workspace's rules: excluded paths
# (special files, runtime state, escaping links) are never manifest entries,
# and every kept link resolves inside the snapshot. It intentionally does not
# inspect solver paths.
_ADMISSION_SCRIPT = r"""
import hashlib, json, os, pathlib, stat, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text())
entries = manifest["entries"]
exclusions = manifest.get("exclusions", [])
if any((e["reason"] == "symlink_escape") != (e.get("link_target") is not None) for e in exclusions):
    raise ValueError("only a symlink_escape exclusion records a link target")
if {e["original_path"] for e in entries} & {e["original_path"] for e in exclusions}:
    raise ValueError("manifest both captures and excludes a path")
actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*")}
expected = {e["path"] for e in entries}
if len(expected) != len(entries) or actual.keys() != expected:
    raise ValueError("workspace inventory does not match manifest")
for entry in entries:
    path = actual[entry["path"]]
    mode = path.lstat().st_mode
    if entry["kind"] == "symlink":
        valid = stat.S_ISLNK(mode) and os.readlink(path) == entry["link_target"] and path.resolve().is_relative_to(root)
    elif entry["kind"] == "directory":
        valid = stat.S_ISDIR(mode)
    else:
        valid = stat.S_ISREG(mode) and path.stat().st_size == entry["size"]
        if valid:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            valid = digest.hexdigest() == entry["sha256"]
    if not valid:
        raise ValueError("workspace evidence mismatch: " + entry["path"])
"""


async def validate_uploaded_workspace(
    env: Sandbox, workspace: str, manifest_path: str, *, timeout_sec: int = 600
) -> None:
    """Reject missing or changed files in the reviewer sandbox after upload."""
    command = shlex.join(["python3", "-c", _ADMISSION_SCRIPT, workspace, manifest_path])
    result = await env.exec(command, user="root", timeout_sec=timeout_sec)
    if result.return_code:
        raise EvidenceError(
            f"Uploaded workspace validation failed: {(result.stderr or result.stdout or '')[-2000:]}"
        )


def prepare_workspace_upload(bundle: Path, dest_archive: Path) -> None:
    """Package validated evidence without upload_dir's symlink filtering."""
    manifest = EvidenceManifest.model_validate_json(
        (bundle / "manifest.json").read_text()
    )
    workspace = bundle / "workspace"
    validate_workspace(workspace, manifest)
    if dest_archive.resolve().is_relative_to(workspace.resolve()):
        raise EvidenceError("Upload archive must be outside the evidence workspace")
    dest_archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest_archive, "x", format=tarfile.PAX_FORMAT) as archive:
        for entry in manifest.entries:
            member = archive.gettarinfo(workspace / entry.path, arcname=entry.path)
            if entry.kind == "file":
                member.type, member.linkname, member.size = (
                    tarfile.REGTYPE,
                    "",
                    entry.size,
                )
                with (workspace / entry.path).open("rb") as stream:
                    archive.addfile(member, stream)
            else:
                archive.addfile(member)


_UNPACK_SCRIPT = r"""
import json, pathlib, posixpath, shutil, sys, tarfile
destination = pathlib.Path(sys.argv[2])
if destination.exists() or destination.is_symlink():
    raise ValueError("uploaded workspace destination already exists")
manifest = json.loads(pathlib.Path(sys.argv[3]).read_text())
expected = {entry["path"]: entry for entry in manifest["entries"]}
with tarfile.open(sys.argv[1], "r:") as archive:
    members, names, links = [], set(), set()
    for member in archive:
        path = pathlib.PurePosixPath(member.name)
        entry = expected.get(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or entry is None or member.name in names:
            raise ValueError("unsafe or unmanifested archive entry: " + member.name)
        if member.issym():
            target = pathlib.PurePosixPath(member.linkname)
            normalized = pathlib.PurePosixPath(posixpath.normpath(posixpath.join(str(path.parent), member.linkname)))
            if target.is_absolute() or ".." in normalized.parts or entry["kind"] != "symlink" or member.linkname != entry["link_target"]:
                raise ValueError("unsafe archive symlink: " + member.name)
            links.add(path)
        elif member.isdir():
            if entry["kind"] != "directory":
                raise ValueError("archive directory does not match manifest")
        elif member.isfile():
            if entry["kind"] != "file" or member.size != entry["size"]:
                raise ValueError("archive file size does not match manifest")
        else:
            raise ValueError("unsupported archive entry: " + member.name)
        names.add(member.name)
        members.append(member)
    if names != expected.keys():
        raise ValueError("archive inventory does not match manifest")
    if any(parent in links for name in names for parent in pathlib.PurePosixPath(name).parents):
        raise ValueError("archive entry has a symlink parent")
    destination.mkdir(parents=True)
    for member in sorted(members, key=lambda item: (item.issym(), len(pathlib.PurePosixPath(item.name).parts))):
        path = destination / member.name
        path.parent.mkdir(parents=True, exist_ok=True)
        if member.isdir():
            path.mkdir(exist_ok=True)
        elif member.issym():
            path.symlink_to(member.linkname)
        else:
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("archive file has no data: " + member.name)
            with source, path.open("xb") as output:
                shutil.copyfileobj(source, output)
            path.chmod(0o755 if member.mode & 0o111 else 0o644)
if len(sys.argv) > 4 and sys.argv[4] == "restore-modes":
    # Deepest first, so a restrictive directory mode never blocks its entries.
    for name in sorted(expected, key=lambda item: -len(pathlib.PurePosixPath(item).parts)):
        entry = expected[name]
        if entry.get("mode") is not None and entry["kind"] in ("file", "directory"):
            (destination / name).chmod(int(entry["mode"]) & 0o777)
"""


async def install_uploaded_workspace(
    env: Sandbox,
    archive_path: str,
    workspace: str,
    manifest_path: str,
    *,
    timeout_sec: int = 600,
    restore_modes: bool = False,
) -> None:
    """Restore a packaged workspace, preserving links, then verify every byte.

    Call before starting the reviewer, followed by its normal evidence locking.
    The destination must be fresh; a failed installation is never admissible.
    ``restore_modes`` puts back the permission bits the solver left (for a
    verifier run on the restored workspace); otherwise files are 0644/0755.
    """
    argv = ["python3", "-c", _UNPACK_SCRIPT, archive_path, workspace, manifest_path]
    if restore_modes:
        argv.append("restore-modes")
    command = shlex.join(argv)
    result = await env.exec(command, user="root", timeout_sec=timeout_sec)
    if result.return_code:
        raise EvidenceError(
            f"Workspace installation failed: {(result.stderr or result.stdout or '')[-2000:]}"
        )
    await validate_uploaded_workspace(
        env, workspace, manifest_path, timeout_sec=timeout_sec
    )


class _ArtifactProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    exists: bool


_ARTIFACT_PROBE_SCRIPT = r"""
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for source in json.loads(sys.argv[2]):
    path = (root / source).resolve()
    print(json.dumps({"source": str(path), "exists": path.exists()}))
"""


async def capture_task_evidence(
    env: Sandbox,
    workspace: str,
    destination: Path,
    *,
    artifacts: Sequence[str | ArtifactConfig] = (),
    excluded_paths: Sequence[str] = (),
    max_bytes: int = 20 * 1024**3,
    max_entries: int = 200_000,
    timeout_sec: int = 600,
) -> EvidenceManifest:
    """Commit workspace and declared external artifacts as one evidence bundle.

    Sources already in the workspace are recorded without recopying them.
    A missing declared output is evidence for the judge, not a transfer error.
    Artifact destinations remain metadata, never arbitrary host filesystem paths.
    """
    if destination.exists() or destination.is_symlink():
        raise EvidenceError(f"Evidence destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    configs = [
        ArtifactConfig(source=item) if isinstance(item, str) else item
        for item in artifacts
    ]
    with tempfile.TemporaryDirectory(
        prefix=".task-evidence-", dir=destination.parent
    ) as temp:
        bundle = Path(temp) / "bundle"
        manifest = await capture_workspace(
            env,
            workspace,
            bundle,
            excluded_paths=excluded_paths,
            max_bytes=max_bytes,
            max_entries=max_entries,
            timeout_sec=timeout_sec,
        )
        records: list[ArtifactEvidence] = []
        remaining_bytes = max_bytes - sum(entry.size for entry in manifest.entries)
        remaining_entries = max_entries - len(manifest.entries)
        if configs:
            command = shlex.join(
                [
                    "python3",
                    "-c",
                    _ARTIFACT_PROBE_SCRIPT,
                    manifest.workspace,
                    json.dumps([config.source for config in configs]),
                ]
            )
            result = await env.exec(command, user="root", timeout_sec=30)
            if result.return_code and python_missing(result):
                result = await env.exec(
                    shlex.join(
                        [
                            "sh",
                            "-c",
                            _SHELL_ARTIFACT_PROBE,
                            "probe",
                            manifest.workspace,
                            *(config.source for config in configs),
                        ]
                    ),
                    user="root",
                    timeout_sec=30,
                )
                shell_probe = True
            else:
                shell_probe = False
            if result.return_code:
                raise EvidenceError(
                    f"Declared artifact discovery failed: {(result.stderr or result.stdout or '')[-2000:]}"
                )
            probes = [
                _shell_artifact_probe(line)
                if shell_probe
                else _ArtifactProbe.model_validate_json(line)
                for line in (result.stdout or "").splitlines()
            ]
            if len(probes) != len(configs):
                raise EvidenceError(
                    "Sandbox did not return every declared artifact status"
                )
            for index, (config, probe) in enumerate(zip(configs, probes, strict=True)):
                bundle_path = None
                if not probe.exists:
                    status = "missing"
                elif PurePosixPath(probe.source).is_relative_to(manifest.workspace):
                    status = "in_workspace"
                else:
                    if min(remaining_bytes, remaining_entries) <= 0:
                        raise EvidenceError(
                            "Task evidence exceeds configured capture limits"
                        )
                    status = "captured"
                    bundle_path = f"artifacts/{index:04d}"
                    artifact = await capture_workspace(
                        env,
                        probe.source,
                        bundle / bundle_path,
                        excluded_paths=excluded_paths,
                        exclude=config.exclude,
                        max_bytes=remaining_bytes,
                        max_entries=remaining_entries,
                        timeout_sec=timeout_sec,
                    )
                    remaining_bytes -= sum(entry.size for entry in artifact.entries)
                    remaining_entries -= len(artifact.entries)
                records.append(
                    ArtifactEvidence(
                        source=probe.source,
                        destination=config.destination,
                        exclude=tuple(config.exclude),
                        status=status,
                        bundle_path=bundle_path,
                    )
                )
        manifest = manifest.model_copy(update={"artifacts": tuple(records)})
        (bundle / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        bundle.rename(destination)
        return manifest


async def install_review_evidence(
    env: Sandbox,
    manifest: EvidenceManifest,
    *,
    evidence_root: str = "/evidence",
    timeout_sec: int = 600,
) -> None:
    """Install workspace plus every captured artifact before reviewer admission."""
    await install_uploaded_workspace(
        env,
        f"{evidence_root}/workspace.tar",
        f"{evidence_root}/workspace",
        f"{evidence_root}/workspace-manifest.json",
        timeout_sec=timeout_sec,
    )
    for artifact in manifest.artifacts:
        if artifact.bundle_path is None:
            continue
        remote = f"{evidence_root}/{artifact.bundle_path}"
        await install_uploaded_workspace(
            env,
            f"{remote}.tar",
            remote,
            f"{remote}-manifest.json",
            timeout_sec=timeout_sec,
        )
