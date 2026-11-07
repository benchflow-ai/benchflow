"""Hub datasets on disk: reproducible task archives to publish, and digest-checked pulls into the cache.

A published dataset version is one archive per task plus a manifest naming each task's BenchFlow content digest
(:func:`benchflow._utils.task_authoring.task_digest`, the digest git-pinned registries use) and its archive's
sha256. A pull checks the archive's sha256 and size before unpacking, unpacks with tarfile's ``data`` filter plus
our own rules (regular files and folders only, no absolute paths, no ``..``), and checks the task digest of what it
unpacked before the task is used. A cached task whose digest no longer matches is pulled again.

Archives are reproducible: entries sorted by path, mtime 0, uid and gid 0, modes 0644 or 0755 (0755 when the file
had any execute bit), gzip with no timestamp and no file name. The same folder always packs to the same bytes, so a version can be
republished idempotently and two publishers can compare archives by hash.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from benchflow._utils.task_authoring import task_digest
from benchflow.hub.client import HUB_REGISTRY_PREFIX, HubClient, HubError

SOURCE_SIDECAR = (
    ".benchflow-source.json"  # the name hf_datasets and the source checks read
)
MAX_TASK_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_TASK_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
MAX_TASK_FILES = 100_000


def dataset_digest(task_digests: dict[str, str]) -> str:
    """sha256 over the sorted lines ``"<task name> <task digest>\\n"``: the identity of a dataset version."""
    h = hashlib.sha256()
    for name in sorted(task_digests):
        h.update(f"{name} {task_digests[name]}\n".encode())
    return f"sha256:{h.hexdigest()}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_task(task_dir: Path, dest: Path) -> tuple[str, int]:
    """Write ``task_dir`` as a reproducible .tar.gz at ``dest``; return (sha256 hex, size).

    Refuses links and special files: a published task is plain files and folders.
    """
    task_dir = Path(task_dir)
    entries: list[tuple[str, Path, bool]] = []
    for dirpath, dirnames, filenames in os.walk(task_dir):
        dirnames.sort()
        base = Path(dirpath)
        for name in [*dirnames, *filenames]:
            path = base / name
            rel = path.relative_to(task_dir).as_posix()
            if name == SOURCE_SIDECAR:
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise HubError(
                    f"{task_dir.name}: {rel} is a link; a published task holds plain files only"
                )
            if stat.S_ISDIR(info.st_mode):
                entries.append((rel, path, True))
            elif stat.S_ISREG(info.st_mode):
                entries.append((rel, path, False))
            else:
                raise HubError(f"{task_dir.name}: {rel} is not a regular file")
    entries.sort(key=lambda e: e[0])
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for rel, path, is_dir in entries:
            info = tarfile.TarInfo(rel)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if is_dir:
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
                continue
            executable = bool(path.stat().st_mode & 0o111)
            info.mode = 0o755 if executable else 0o644
            info.size = path.stat().st_size
            with open(path, "rb") as f:
                tar.addfile(info, f)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # filename="": gzip would otherwise write dest's name into its header, so one folder packed to two
    # paths would give two different archives.
    with (
        open(dest, "wb") as out,
        gzip.GzipFile(
            filename="", fileobj=out, mode="wb", mtime=0, compresslevel=9
        ) as gz,
    ):
        gz.write(raw.getvalue())
    return _sha256_file(dest), dest.stat().st_size


def unpack_task(archive: Path, dest: Path) -> None:
    """Unpack a task archive into the empty folder ``dest``, refusing anything but plain files and folders."""
    from benchflow._utils.archive_paths import ArchivePaths

    total = 0
    inventory = ArchivePaths()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) > MAX_TASK_FILES:
            raise HubError(f"{archive.name} holds more than {MAX_TASK_FILES} entries")
        for m in members:
            try:
                # Names that alias one another (case, Unicode form) or sit under
                # a file are refused, as in every other archive BenchFlow opens.
                inventory.add(m.name, directory=m.isdir())
            except ValueError as exc:
                raise HubError(
                    f"{archive.name}: refusing entry {m.name!r} ({exc})"
                ) from None
            rel = PurePosixPath(m.name)
            if rel.is_absolute() or ".." in rel.parts or not (m.isfile() or m.isdir()):
                raise HubError(
                    f"{archive.name}: refusing entry {m.name!r} (only plain files and folders inside the task)"
                )
            total += m.size
            if total > MAX_TASK_UNPACKED_BYTES:
                raise HubError(
                    f"{archive.name} unpacks to more than {MAX_TASK_UNPACKED_BYTES} bytes"
                )
        tar.extractall(dest, members=members, filter="data")


@dataclass(frozen=True)
class HubDataset:
    """A dataset version pulled from a Hub: its folder of tasks and what identifies it."""

    name: str
    version: str
    bench_version: str | None
    tasks_dir: Path
    task_digests: dict[str, str]
    digest: str
    document: dict[str, Any]
    provenance: dict[str, Any]


def cache_root() -> Path:
    return Path(".cache") / "datasets-hub"


def _cache_dir(client: HubClient, name: str, version: str) -> Path:
    host = urlparse(client.url).netloc.replace(":", "_")
    org, short = name.split("/", 1)
    safe = [part for part in (host, org, short, version) if part not in ("", ".", "..")]
    if len(safe) != 4 or any("/" in p or "\\" in p for p in safe):
        raise HubError(
            f"refusing to cache {name}@{version} from {client.url}: a name part is not a folder name"
        )
    return cache_root().joinpath(*safe)


def pull_dataset(client: HubClient, name: str, version: str) -> HubDataset:
    """Pull ``name@version`` from the Hub into the cache, every byte checked; reuse what is cached and unchanged."""
    doc = client.dataset(name, version)
    tasks = doc.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise HubError(f"{name}@{version} on {client.url} lists no tasks")
    digests: dict[str, str] = {}
    for task in tasks:
        tname = task.get("name") if isinstance(task, dict) else None
        if (
            not isinstance(tname, str)
            or not tname
            or "/" in tname
            or tname in (".", "..")
            or tname in digests
        ):
            raise HubError(
                f"{name}@{version} has a malformed or repeated task entry: {task!r}"
            )
        digest = task.get("digest")
        archive = task.get("archive")
        if not isinstance(digest, str) or not isinstance(archive, dict):
            raise HubError(f"{name}@{version}: task {tname} has no digest or archive")
        digests[tname] = digest
    expected = dataset_digest(digests)
    if doc.get("digest") != expected:
        raise HubError(
            f"{name}@{version}: the Hub's dataset digest {doc.get('digest')} does not match its tasks ({expected})"
        )
    root = _cache_dir(client, name, version)
    root.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        tname, digest, archive = task["name"], task["digest"], task["archive"]
        target = root / tname
        if (
            target.is_dir()
            and not target.is_symlink()
            and task_digest(target) == digest
        ):
            continue
        sha, size, url = archive.get("sha256"), archive.get("size"), archive.get("url")
        if not (
            isinstance(sha, str)
            and isinstance(size, int)
            and isinstance(url, str)
            and url.startswith("/api/")
        ):
            raise HubError(
                f"{name}@{version}: task {tname} has a malformed archive entry"
            )
        if size > MAX_TASK_ARCHIVE_BYTES:
            raise HubError(
                f"{name}@{version}: task {tname}'s archive is larger than {MAX_TASK_ARCHIVE_BYTES} bytes"
            )
        with tempfile.TemporaryDirectory(prefix=f".{tname}.", dir=root) as scratch:
            blob = Path(scratch) / "task.tar.gz"
            client.download(url, blob, max_bytes=size)
            if blob.stat().st_size != size or _sha256_file(blob) != sha:
                raise HubError(
                    f"{name}@{version}: task {tname}'s archive does not match the manifest's sha256 and size"
                )
            staged = Path(scratch) / "task"
            staged.mkdir()
            unpack_task(blob, staged)
            got = task_digest(staged)
            if got != digest:
                raise HubError(
                    f"{name}@{version}: task {tname} unpacked to {got}, the manifest pins {digest}"
                )
            if target.is_symlink():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)
            staged.rename(target)
    for extra in root.iterdir():  # a task the version no longer lists is not part of it
        if (
            extra.is_dir()
            and not extra.name.startswith(".")
            and extra.name not in digests
        ):
            shutil.rmtree(extra)
    provenance = {
        "type": "benchflow_hub",
        "repo": client.url,
        "path": "",
        "requested_ref": version,
        "resolved_sha": expected,
        "dirty": False,
        "dataset": name,
        "published_at": doc.get("published_at"),
        "published_by": doc.get("published_by"),
        "upstream": doc.get("source"),
    }
    (root / SOURCE_SIDECAR).write_text(json.dumps(provenance, indent=2))
    return HubDataset(
        name=name,
        version=version,
        bench_version=doc.get("bench_version"),
        tasks_dir=root,
        task_digests=digests,
        digest=expected,
        document=doc,
        provenance={**provenance, "local_path": str(root.resolve())},
    )


def build_manifest(
    tasks_dir: Path,
    name: str,
    version: str,
    out_dir: Path,
    *,
    description: str | None = None,
    license: str | None = None,
    bench_version: str | None = None,
    kind: str | None = None,
    source: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    """Pack every task folder of ``tasks_dir`` into ``out_dir``; return the manifest and (task, archive) pairs."""
    task_dirs = sorted(
        p
        for p in Path(tasks_dir).iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.is_symlink()
    )
    if not task_dirs:
        raise HubError(f"no task folders in {tasks_dir}")
    entries, archives = [], []
    for task in task_dirs:
        if not ((task / "task.md").is_file() or (task / "task.toml").is_file()):
            raise HubError(
                f"{task.name} is not a task folder (no task.md or task.toml)"
            )
        archive = out_dir / f"{task.name}.tar.gz"
        sha, size = pack_task(task, archive)
        entries.append(
            {
                "name": task.name,
                "digest": task_digest(task),
                "archive": {"sha256": sha, "size": size},
            }
        )
        archives.append((task.name, archive))
    manifest: dict[str, Any] = {
        "schema": "benchflow.hub.dataset/1",
        "name": name,
        "version": version,
        "digest": dataset_digest({e["name"]: e["digest"] for e in entries}),
        "tasks": entries,
    }
    for key, value in (
        ("description", description),
        ("license", license),
        ("bench_version", bench_version),
        ("kind", kind),
        ("source", source),
    ):
        if value:
            manifest[key] = value
    return manifest, archives


def registry_label(url: str) -> str:
    return f"the Hub at {url}"


def hub_registry_value(url: str) -> str:
    return HUB_REGISTRY_PREFIX + url
