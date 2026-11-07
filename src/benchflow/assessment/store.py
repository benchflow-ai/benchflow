"""The sidecar on disk: one append-only, hash-chained JSONL log per assessment.

A sidecar directory holds:

``assessment-events.v1.jsonl``
    One event per line, each line exactly ``JCS(event) + "\\n"``.
``assessment-events.v1.lock``
    The POSIX advisory lock every writer holds while it appends.
``assessment-label.v1.json``
    The current ``benchflow.assessment-label.v1`` record, rewritten from the
    log after each append once a label exists. Derived: the log is the record.

Writers (:meth:`Sidecar.append`) take an exclusive ``fcntl.lockf`` lock, then
read and verify the whole log, then check, in this order:

1. idempotency: an event with the same ``idempotency_key`` and the same
   semantic payload is returned as is (a retry, whatever its expectations);
   the same key with another payload is a :class:`Conflict`;
2. the writer's ``expected_event_seq`` and ``expected_previous_event_sha256``
   still name the last event, else :class:`StaleExpectation`;
3. the state machine accepts the event.

Only then is the line written, with one ``write`` loop on an ``O_APPEND``
descriptor, and ``fsync``ed (``F_FULLFSYNC`` where the platform has it); a new
file also syncs its directory. A failed write truncates its own unacknowledged
bytes, still under the lock. Every refusal appends nothing.

Readers verify the same way and refuse the whole log on any defect: a last
line with no newline (a torn append), a line that is not canonical JCS, a
duplicate member name, an unknown schema version, a broken ``event_seq`` or
hash chain, a reused idempotency key or a transition the state machine
forbids. The log is never repaired or rewritten.

The lock is local: keep sidecars on a local filesystem.
"""

from __future__ import annotations

import errno
import fcntl
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchflow.assessment import records as r
from benchflow.assessment.errors import (
    AssessmentError,
    Conflict,
    CorruptSidecar,
    InvalidRecord,
    StaleExpectation,
)
from benchflow.assessment.jcs import CanonicalizationError, canonicalize, loads_strict
from benchflow.assessment.state import Assessment, Event

__all__ = ["EVENTS_FILE", "LABEL_FILE", "LOCK_FILE", "Appended", "Sidecar"]

EVENTS_FILE = "assessment-events.v1.jsonl"
LOCK_FILE = "assessment-events.v1.lock"
LABEL_FILE = "assessment-label.v1.json"

# One lock per lock file within this process: POSIX record locks belong to the
# process, so threads would not exclude each other through them alone.
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class Appended:
    """The event a write produced, or the original event a retry replayed.

    ``label_error`` is set when the event was appended (it is durable) but the
    derived label file could not be rewritten; ``bench assess label
    <sidecar> --write`` rebuilds it from the log.
    """

    event: Event
    replayed: bool
    assessment: Assessment
    label_error: str | None = None


def _thread_lock(path: Path) -> threading.Lock:
    key = os.path.realpath(path)
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


def _sync(fd: int) -> None:
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is not None:
        try:
            fcntl.fcntl(fd, full)
            return
        except OSError:
            pass  # not supported by this filesystem: fall back to fsync
    os.fsync(fd)


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _sync(fd)
    finally:
        os.close(fd)


def _read_all(fd: int) -> bytes:
    size = os.fstat(fd).st_size
    chunks = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(size - offset, 1 << 20), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def parse_log(data: bytes) -> Assessment:
    """Verify a whole log and replay it; raise :class:`CorruptSidecar` on any defect."""
    assessment = Assessment()
    if not data:
        return assessment
    if not data.endswith(b"\n"):
        offset = data.rfind(b"\n") + 1
        raise CorruptSidecar(
            f"the last line (from byte {offset}) has no newline: a torn append; the "
            "log is left exactly as found",
            hint="move the sidecar aside and keep it as evidence; nothing reads past a torn line",
        )
    previous: str | None = None
    for number, line in enumerate(data[:-1].split(b"\n"), start=1):
        where = f"line {number}"
        if not line:
            raise CorruptSidecar(f"{where} is empty")
        try:
            record = loads_strict(line)
        except CanonicalizationError as exc:
            raise CorruptSidecar(f"{where}: {exc}") from None
        try:
            r.check_envelope(record, where)
        except InvalidRecord as exc:
            raise CorruptSidecar(str(exc)) from None
        try:
            canonical = canonicalize(record)
        except CanonicalizationError as exc:
            raise CorruptSidecar(f"{where}: {exc}") from None
        if canonical != line:
            raise CorruptSidecar(f"{where} is not in RFC 8785 canonical form")
        if record["event_sha256"] != r.event_digest(record):
            raise CorruptSidecar(f"{where}: event_sha256 does not match the event")
        if record["event_seq"] != number:
            raise CorruptSidecar(
                f"{where}: event_seq is {record['event_seq']}, expected {number}"
            )
        if record["previous_event_sha256"] != previous:
            raise CorruptSidecar(
                f"{where}: previous_event_sha256 breaks the hash chain"
            )
        if record["expected_previous_event_sha256"] != previous:
            raise CorruptSidecar(
                f"{where}: expected_previous_event_sha256 differs from the event it followed"
            )
        try:
            assessment.apply(Event(record, line))
        except AssessmentError as exc:
            raise CorruptSidecar(
                f"{where} ({record['kind']}) is not allowed: {exc}"
            ) from None
        previous = record["event_sha256"]
    return assessment


class Sidecar:
    """One assessment's sidecar directory."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.events_path = self.directory / EVENTS_FILE
        self.lock_path = self.directory / LOCK_FILE
        self.label_path = self.directory / LABEL_FILE

    def exists(self) -> bool:
        return self.events_path.exists()

    # -- locking -----------------------------------------------------------

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        thread_lock = _thread_lock(self.lock_path)
        with thread_lock:
            if exclusive:
                self._make_directory()
                flags = os.O_RDWR | os.O_CREAT
            else:
                if not self.lock_path.exists():
                    # A copied sidecar may lack its lock file; no writer uses it.
                    yield
                    return
                flags = os.O_RDONLY
            fd = os.open(self.lock_path, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                try:
                    yield
                finally:
                    fcntl.lockf(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _make_directory(self) -> None:
        if self.directory.is_dir():
            return
        missing = []
        path = self.directory
        while not path.exists():
            missing.append(path)
            path = path.parent
        self.directory.mkdir(parents=True, exist_ok=True)
        for created in reversed(missing):
            _sync_dir(created.parent)

    # -- reading -----------------------------------------------------------

    def _open_log(self, flags: int) -> int:
        return os.open(self.events_path, flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)

    def read_bytes(self) -> bytes:
        try:
            fd = self._open_log(os.O_RDONLY)
        except FileNotFoundError:
            return b""
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise CorruptSidecar(f"{self.events_path} is a symbolic link") from None
            raise
        try:
            return _read_all(fd)
        finally:
            os.close(fd)

    def load(self) -> Assessment:
        """Verify the whole log under a shared lock and replay it."""
        with self._locked(exclusive=False):
            return parse_log(self.read_bytes())

    # -- writing -----------------------------------------------------------

    def append(
        self,
        *,
        kind: str,
        actor: Mapping[str, str],
        payload: Mapping[str, Any],
        idempotency_key: str,
        expected_event_seq: int,
        expected_previous_event_sha256: str | None,
        now: Callable[[], str] = r.utc_now,
    ) -> Appended:
        """Append one event, or return the event an identical retry wrote first."""
        actor = dict(actor)
        payload = dict(payload)
        try:
            semantic = r.semantic_payload(kind, actor, payload)
        except CanonicalizationError as exc:
            raise InvalidRecord(f"payload is not I-JSON: {exc}") from None
        with self._locked(exclusive=True):
            existed = self.events_path.exists()
            try:
                fd = self._open_log(os.O_RDWR | os.O_APPEND | os.O_CREAT)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise CorruptSidecar(
                        f"{self.events_path} is a symbolic link"
                    ) from None
                raise
            try:
                data = _read_all(fd)
                assessment = parse_log(data)
                original = assessment.keys.get(idempotency_key)
                if original is not None:
                    if original.semantic() == semantic:
                        return Appended(original, True, assessment)
                    raise Conflict(
                        f"idempotency key {idempotency_key!r} was used by event "
                        f"{original.id} ({original.kind}) with a different payload; "
                        "nothing was appended",
                    )
                head = (assessment.head_seq, assessment.head_sha256)
                if (expected_event_seq, expected_previous_event_sha256) != head:
                    raise StaleExpectation(
                        f"the log ends at event_seq {head[0]} ({head[1]}); the writer expected "
                        f"{expected_event_seq} ({expected_previous_event_sha256}); nothing was "
                        "appended",
                        hint="reload the sidecar, re-check its state, then retry",
                    )
                record: dict[str, Any] = {
                    "schema_version": r.EVENT_SCHEMA,
                    "assessment_id": assessment.assessment_id or r.uuid7(),
                    "event_id": r.uuid7(),
                    "event_seq": head[0] + 1,
                    "created_at": now(),
                    "kind": kind,
                    "actor": actor,
                    "idempotency_key": idempotency_key,
                    "expected_previous_event_sha256": expected_previous_event_sha256,
                    "previous_event_sha256": head[1],
                    "payload": payload,
                }
                record["event_sha256"] = r.event_digest(record)
                r.check_envelope(record)
                line = canonicalize(record)
                event = Event(record, line)
                assessment.apply(event)
                self._write_line(fd, line + b"\n", len(data))
                if not existed:
                    _sync_dir(self.directory)
            finally:
                os.close(fd)
            try:
                self._write_label(assessment)
            except OSError as exc:
                return Appended(event, False, assessment, label_error=str(exc))
            return Appended(event, False, assessment)

    @staticmethod
    def _write_line(fd: int, line: bytes, size_before: int) -> None:
        view = memoryview(line)
        written = 0
        try:
            while written < len(view):
                written += os.write(fd, view[written:])
            _sync(fd)
        except BaseException:
            # Our own bytes, never acknowledged and still under the lock.
            try:
                os.ftruncate(fd, size_before)
                _sync(fd)
            except OSError:
                pass  # the torn tail stays; every later read refuses it
            raise

    # -- the derived label file -------------------------------------------

    def _write_label(self, assessment: Assessment) -> Path | None:
        label = assessment.label()
        if label is None:
            return None
        data = canonicalize(label) + b"\n"
        temporary = (
            self.directory / f".{LABEL_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            view = memoryview(data)
            written = 0
            while written < len(view):
                written += os.write(fd, view[written:])
            _sync(fd)
        except BaseException:
            os.close(fd)
            temporary.unlink(missing_ok=True)
            raise
        os.close(fd)
        os.replace(temporary, self.label_path)
        _sync_dir(self.directory)
        return self.label_path

    def write_label(self) -> Path | None:
        """Rebuild the label file from the log alone (under the writer lock)."""
        with self._locked(exclusive=True):
            return self._write_label(parse_log(self.read_bytes()))

    def read_label_file(self) -> bytes | None:
        try:
            return self.label_path.read_bytes()
        except FileNotFoundError:
            return None
