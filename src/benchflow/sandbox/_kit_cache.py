"""The built protected kit, kept per image on this host (PERF-SBX).

``kit_prepare_command("build")`` copies root-only image files into
``/opt/benchflow/rt`` and captures the image interpreters' search paths:
0.9-1.3 s of a trial on one CPU, the same bytes for every fresh container of
one image. A cache keyed by image and recipe alone is refused because a
Compose mount or the task's entrypoint can change what the container holds
before the build, so the image ID does not prove the container's contents.

This cache is used only when the container's contents provably are the
image's for everything the build reads, checked on every use:

- the container is a fresh one of the task's image on the admitted direct
  path, its only mounts BenchFlow's own ``/logs`` folders (:func:`eligible`);
- the daemon's own diff of the container against its image
  (``GET /containers/{id}/changes``) names nothing but what BenchFlow wrote
  there before the build (the task's instruction and solution and the
  ``/logs`` mount points), none of which the build reads;
- the key holds the image ID, every container setting the build can see
  (environment, entrypoint and command, working folder, user, platform), the
  daemon's storage and security options, and the build recipe's own text;
- the cached tree is the exact bytes a build in such a container wrote
  (exported right after the build, before the workspace grant), checked
  against its recorded SHA-256 before it is used.

Anything else builds as before. The seal, the PATH lock and every later
proof still run on the restored kit, as on a built one.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CachedKit",
    "cache_root",
    "eligible",
    "kit_key",
    "load",
    "staged_tar",
    "store",
    "unchanged",
]

# BenchFlow's own mounts: the rollout's log folders.
LOG_MOUNTS = frozenset({"/logs/agent", "/logs/verifier", "/logs/artifacts"})

# What BenchFlow itself writes into a fresh container before the kit build,
# none of it read by the build: the task's instruction and the oracle's
# solution (uploaded at start) and the log mount points.
_WRITTEN_BEFORE_BUILD = ("/instruction.md", "/solution", "/logs")

# Kept entries; the oldest go first past this.
MAX_ENTRIES = 64

_KEY = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class CachedKit:
    """A cache entry: *key*, and its tar when this host has it."""

    key: str
    tar: Path | None


def cache_root() -> Path:
    cache_home = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "benchflow" / "kit"


# Off in 0.8.3. Both PERF-SBX reviews found the cache unsafe: its key and
# diff are read before the task's setup commands and entrypoint run, and the
# kit carries a per-trial capture of the image Python's search paths, which
# that Python's own start-up hooks may derive from the trial's files. The
# mechanism stays, with its tests, for 0.8.4 (GOALS PERF-SBX-KIT-CACHE).
ENABLED = False


def eligible(inspected: dict[str, Any], task_image_id: str | None) -> str | None:
    """Why the container *inspected* may not use the cache, or None.

    A fresh container of the task's image, running, whose mounts are only
    BenchFlow's log folders, with no shared volume or device of its own.
    """
    if not isinstance(inspected, dict):
        return "no container record"
    image = inspected.get("Image")
    if not isinstance(image, str) or not image.startswith("sha256:"):
        return "no image ID"
    if task_image_id is not None and image != task_image_id:
        return "not the task's image"
    state = inspected.get("State") or {}
    if not isinstance(state, dict) or state.get("Running") is not True:
        return "not running"
    if state.get("Restarting") or inspected.get("RestartCount"):
        return "restarted"
    mounts = inspected.get("Mounts")
    if not isinstance(mounts, list):
        return "no mount list"
    for mount in mounts:
        if (
            not isinstance(mount, dict)
            or mount.get("Type") != "bind"
            or mount.get("Destination") not in LOG_MOUNTS
        ):
            return "a mount other than BenchFlow's log folders"
    host = inspected.get("HostConfig") or {}
    if not isinstance(host, dict):
        return "no host configuration"
    for name in ("VolumesFrom", "Devices", "Tmpfs"):
        if host.get(name):
            return f"{name} set"
    for mount in host.get("Mounts") or ():
        if (
            not isinstance(mount, dict)
            or mount.get("Type") != "bind"
            or mount.get("Target") not in LOG_MOUNTS
        ):
            return "a mount other than BenchFlow's log folders"
    return None


def unchanged(changes: Any) -> str | None:
    """Why the daemon's diff of the container may have changed what the
    build reads, or None: only BenchFlow's own writes before the build."""
    if not isinstance(changes, list):
        return "the daemon gave no change list"
    for change in changes:
        path = change.get("Path") if isinstance(change, dict) else None
        if not isinstance(path, str) or not path.startswith("/"):
            return "an unreadable change"
        if not any(
            path == root or path.startswith(root + "/")
            for root in _WRITTEN_BEFORE_BUILD
        ):
            return f"{path} changed"
    return None


def kit_key(inspected: dict[str, Any], daemon: dict[str, Any], recipe: str) -> str:
    """Everything the build's result depends on, hashed."""
    config = inspected.get("Config") or {}
    host = inspected.get("HostConfig") or {}
    document = {
        "v": 1,
        "image": inspected.get("Image"),
        "platform": inspected.get("Platform"),
        "env": config.get("Env"),
        "entrypoint": config.get("Entrypoint"),
        "cmd": config.get("Cmd"),
        "workdir": config.get("WorkingDir"),
        "user": config.get("User"),
        "privileged": host.get("Privileged"),
        "readonly": host.get("ReadonlyRootfs"),
        "userns": host.get("UsernsMode"),
        "security": host.get("SecurityOpt"),
        "daemon": {
            "id": daemon.get("ID"),
            "driver": daemon.get("Driver"),
            "security": daemon.get("SecurityOptions"),
            "arch": daemon.get("Architecture"),
        },
        "recipe": hashlib.sha256(recipe.encode()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(key: str) -> CachedKit:
    """The entry for *key*; its tar only when its bytes match the record."""
    if not _KEY.fullmatch(key):
        return CachedKit(key, None)
    root = cache_root()
    tar, record = root / f"{key}.tar", root / f"{key}.sha256"
    try:
        expected = record.read_text().strip()
        digest = hashlib.sha256()
        with tar.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return CachedKit(key, None)
    if not _KEY.fullmatch(expected) or digest.hexdigest() != expected:
        return CachedKit(key, None)
    with contextlib.suppress(OSError):
        os.utime(tar)
    return CachedKit(key, tar)


def store(key: str, data: bytes) -> None:
    """Keep *data* (the built kit's tar) for *key*, atomically, in a folder
    only this user may read; the oldest entries past :data:`MAX_ENTRIES` go."""
    if not _KEY.fullmatch(key) or not data:
        return
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    digest = hashlib.sha256(data).hexdigest()
    for suffix, content in ((".tar", data), (".sha256", (digest + "\n").encode())):
        fd, temp = tempfile.mkstemp(dir=root, prefix=f".{key}{suffix}.")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            os.replace(temp, root / f"{key}{suffix}")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp)
            raise
    entries = sorted(root.glob("*.tar"), key=lambda p: p.stat().st_mtime)
    for old in entries[: max(0, len(entries) - MAX_ENTRIES)]:
        for path in (old, old.with_suffix(".sha256")):
            with contextlib.suppress(OSError):
                path.unlink()


def staged_tar(tar: Path, stage: str) -> bytes:
    """*tar* (the kit folder ``rt/...``, as kept) under the folder *stage*
    (a single name, made root's and mode 700 first), for one upload to the
    container's ``/``. Every member keeps its bytes, mode, owner and link."""
    if not re.fullmatch(r"\.benchflow-kit-[0-9a-f]{32}", stage):
        raise ValueError(f"not a kit stage name: {stage!r}")
    out = io.BytesIO()
    with (
        tarfile.open(tar, "r:") as source,
        tarfile.open(fileobj=out, mode="w:", format=tarfile.PAX_FORMAT) as target,
    ):
        top = tarfile.TarInfo(stage)
        top.type, top.mode, top.uid, top.gid = tarfile.DIRTYPE, 0o700, 0, 0
        target.addfile(top)
        for member in source:
            if not (member.name == "rt" or member.name.startswith("rt/")):
                raise ValueError(f"a kept kit holds {member.name!r}")
            data = source.extractfile(member) if member.isreg() else None
            member.name = f"{stage}/{member.name}"
            if member.islnk():
                member.linkname = f"{stage}/{member.linkname}"
            target.addfile(member, data)
    return out.getvalue()
