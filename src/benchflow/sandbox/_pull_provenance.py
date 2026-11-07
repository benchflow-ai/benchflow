"""Which task images BenchFlow itself pulled onto a Docker daemon.

Docker keeps a pulled image until someone removes it. Removing every image a
run used would also remove images the user had before the run, so BenchFlow
records an image only when it saw the reference absent and then pulled it:
the daemon (by ``docker info``'s ID, hashed), the reference, the image id it
got, when the daemon says the reference was tagged (``Metadata.LastTagTime``)
and the process that pulled it. ``bench cache clean --only pulled-images``
reads these records and checks whether a reference still
names that image id with the same tag time, no other tag or container uses
the image and the pulling process is gone. A reference that existed before,
was retagged since, was removed and pulled again since (the same image id,
a newer tag time: the user's own pull after a prune), or has no record or no
tag time is never removed. The tag time is recorded only on the containerd
image store (Docker 29's default), which keeps one per name. The classic store
keeps one per image, set by ``docker tag`` (a layer build's pin of the task
image too) and not by a pull, so there no time is recorded and nothing is
removed. Docker has no atomic conditional reference deletion, so even a
reference passing every check is currently retained. A subsequent pull or
tag could replace the checked reference before removal.

Records are JSON lines in ``$BENCHFLOW_PULL_RECORD_DIR`` or
``${XDG_STATE_HOME:-~/.local/state}/benchflow/pulled-images/<daemon>.jsonl``
(a folder of this user's, mode 0700). They hold no credentials: a Docker
reference cannot carry one.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchflow.sandbox._layer_lock import daemon_hash


def _load_fcntl() -> Any:
    """``fcntl``, or None where the platform has none (Windows)."""
    try:
        return importlib.import_module("fcntl")
    except ImportError:  # pragma: no cover - Windows
        return None


fcntl: Any = _load_fcntl()

RECORD_DIR_ENV = "BENCHFLOW_PULL_RECORD_DIR"
# A Docker reference: registry/name[:tag][@digest], nothing else.
_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]*")


@dataclass(frozen=True)
class PullRecord:
    daemon: str  # daemon_hash(docker info ID)
    ref: str
    image_id: str
    pulled_at: str  # ISO 8601, UTC
    owner: str  # env_layers.alias_owner_tag() of the pulling process
    # docker image inspect's Metadata.LastTagTime just after the pull; "" on
    # the classic image store, when the daemon reported none, or in an older
    # record.
    tag_time: str = ""

    @property
    def pulled(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.pulled_at).astimezone(UTC)
        except ValueError:
            return None


def record_dir() -> Path:
    configured = os.environ.get(RECORD_DIR_ENV, "").strip()
    if configured:
        return Path(configured)
    state = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state) if state else Path.home() / ".local" / "state"
    return base / "benchflow" / "pulled-images"


def _path(daemon_id: str) -> Path:
    return record_dir() / f"{daemon_hash(daemon_id)}.jsonl"


@contextlib.contextmanager
def _locked(daemon_id: str) -> Iterator[Path]:
    """The record file's path, with its sidecar lock held (one writer)."""
    path = _path(daemon_id)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = path.with_suffix(".lock")
    fd = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(fd)


def record_pull(
    daemon_id: str, ref: str, image_id: str, owner: str, tag_time: str = ""
) -> None:
    """Note that this process pulled ``ref`` (absent before) as ``image_id``.

    It replaces any older record of ``ref``: the reference was absent, so
    what that record described is gone, and its owner and tag time no longer
    say who holds the reference now. Best effort: a record that cannot be
    written means the image is kept.
    """
    if not _REF_RE.fullmatch(ref) or not image_id.startswith("sha256:"):
        return
    entry = PullRecord(
        daemon=daemon_hash(daemon_id),
        ref=ref,
        image_id=image_id,
        pulled_at=datetime.now(UTC).isoformat(),
        owner=owner,
        tag_time=tag_time,
    )
    with contextlib.suppress(OSError), _locked(daemon_id) as path:
        keep = [r for r in _read(path, entry.daemon) if r.ref != ref]
        _write(path, [*keep, entry])


def _read(path: Path, daemon: str) -> list[PullRecord]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            data = json.loads(line)
            record = PullRecord(
                **{k: str(data[k]) for k in PullRecord.__annotations__ if k in data}
            )
        except (ValueError, KeyError, TypeError):
            continue
        if record.daemon == daemon and _REF_RE.fullmatch(record.ref):
            out.append(record)
    return out


def read_pulls(daemon_id: str) -> list[PullRecord]:
    """This daemon's pull records (read only)."""
    return _read(_path(daemon_id), daemon_hash(daemon_id))


def forget_pulls(daemon_id: str, done: list[PullRecord]) -> None:
    """Drop ``done`` (removed or gone) from this daemon's records."""
    if not done:
        return
    # A concurrent writer can record a new pull of the same ref and id.
    # Retire only the exact receipt checked by the caller, never that one.
    drop = set(done)
    daemon = daemon_hash(daemon_id)
    with contextlib.suppress(OSError), _locked(daemon_id) as path:
        keep = [r for r in _read(path, daemon) if r not in drop]
        _write(path, keep)


def _write(path: Path, records: list[PullRecord]) -> None:
    """Replace the record file with ``records`` (the caller holds its lock)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pulled-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    os.replace(tmp, path)
