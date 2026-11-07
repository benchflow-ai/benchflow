"""Crash-safe live persistence for provider LLM trajectories."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benchflow.trajectories import _snapshot
from benchflow.trajectories.token_capture import TOKEN_CAPTURE_METADATA_KEY
from benchflow.trajectories.types import (
    LLMExchange,
    LLMRequest,
    LLMResponse,
    Trajectory,
)

logger = logging.getLogger(__name__)

# What a published exchange keeps of its bodies in memory: the status code,
# the usage and the model are all a stopped proxy's trajectory is read for
# (usage, provider-failure scans, reported models); the full record is the
# line in ``llm_trajectory.jsonl``.
_REQUEST_KEEP = ("model",)
_RESPONSE_KEEP = ("model", "usage", "usageMetadata")
_COPY_CHUNK = 1 << 20


class PublishedLLMExchange(LLMExchange):
    """A captured exchange whose full record is a line of the published
    ``llm_trajectory.jsonl``; in memory it keeps only what
    :func:`published_exchange` keeps."""


def published_exchange(exchange: LLMExchange) -> PublishedLLMExchange:
    """*exchange* without the bodies a stopped proxy is never read for.

    A Codex request carries the whole conversation twice, so holding every
    exchange whole held a long run's log in the runner several times over.
    """
    request, response = exchange.request, exchange.response
    return PublishedLLMExchange(
        request=LLMRequest(
            timestamp=request.timestamp,
            method=request.method,
            path=request.path,
            body={k: request.body[k] for k in _REQUEST_KEEP if k in request.body},
        ),
        response=LLMResponse(
            timestamp=response.timestamp,
            status_code=response.status_code,
            body={k: response.body[k] for k in _RESPONSE_KEEP if k in response.body},
        ),
        duration_ms=exchange.duration_ms,
        metadata={
            k: v
            for k, v in exchange.metadata.items()
            if k != TOKEN_CAPTURE_METADATA_KEY
        },
    )


def exchange_record(exchange: LLMExchange) -> dict[str, Any]:
    """The JSON record of one exchange, as its published line holds it."""
    return exchange.model_dump(mode="json")


class LiveLLMTrajectoryWriter:
    """Publish redacted provider LLM exchanges, one line per exchange.

    LiteLLM's callback log is append-only, and so is this file while the run
    lasts: :meth:`append` serializes and redacts each new exchange once and
    adds its line. The writer keeps a digest and the end offset of each line,
    never the records, so its memory stays flat however long the run. A
    file is only ever replaced whole through a temporary file and an atomic
    rename (the first write, and a rewrite), so a reader never sees a stale
    attempt's lines mixed with this one's.

    :meth:`reconcile` takes the authoritative exchanges at the end of the run
    one at a time: a record equal to the published line at its position is
    not redacted again, missing lines are appended, and a record that
    differs rewrites the file from that line on. ``preceding_trajectories``
    are earlier roles' captures, published to the same file before this
    writer's lines.
    """

    def __init__(
        self, path: Path, *, preceding_trajectories: tuple[Trajectory, ...] = ()
    ) -> None:
        self._preceding_trajectories = preceding_trajectories
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._tmp.unlink(missing_ok=True)
        # Bytes of the preceding roles' lines this writer's lines follow;
        # None until the first write settles them.
        self._base: int | None = None
        self._digests: list[bytes] = []
        self._ends: list[int] = []
        # The lines of a rewrite whose atomic replace failed: its temporary
        # file is whole, so a retry with the same records publishes it.
        self._pending: tuple[list[bytes], list[int]] | None = None

    @property
    def published(self) -> int:
        """How many of this writer's own lines the file holds."""
        return len(self._digests)

    def append(self, exchanges: Iterable[LLMExchange]) -> bool:
        """Publish *exchanges* after the lines already published."""
        lines = [_line(exchange_record(exchange)) for exchange in exchanges]
        if not lines:
            return False
        self._drop_pending()
        if self._base is None:
            self._rewrite(0, lines)
            return True
        self._append(lines)
        return True

    def write(self, trajectory: Trajectory | None) -> bool:
        """Publish *trajectory* when it is non-empty and changed.

        A trajectory whose exchanges were published as they were read (a
        stopped proxy's) keeps only part of each record, and its writer has
        made this file: nothing is written from it.
        """
        if trajectory is None or not trajectory.exchanges:
            return False
        if any(isinstance(e, PublishedLLMExchange) for e in trajectory.exchanges):
            return False
        if self._pending is not None and self._publish_pending(trajectory):
            return True
        return self.reconcile_records(
            exchange_record(exchange) for exchange in trajectory.exchanges
        )

    def reconcile(self, trajectory: Trajectory | None) -> bool:
        """Publish the authoritative final snapshot.

        Callers may invoke it even if the last poll already captured the
        final exchange: nothing is written then.
        """
        return self.write(trajectory)

    def reconcile_records(self, records: Iterable[dict[str, Any]]) -> bool:
        """Make the writer's lines the redacted *records*, in order.

        *records* is read once, one record at a time; only the records from
        the first one that differs from its published line are redacted. A
        file that is not as this writer left it (removed, or written by
        something else) is rewritten whole.
        """
        self._drop_pending()
        if self._base is not None and self._size() != self._end():
            self._base, self._digests, self._ends = None, [], []
        index = 0
        records = iter(records)
        for record in records:
            raw = json.dumps(record, default=str)
            if (
                self._base is not None
                and index < len(self._digests)
                and _digest(raw) == self._digests[index]
            ):
                index += 1
                continue
            lines = _redacted_lines(raw, records)
            if self._base is not None and index == len(self._digests):
                self._append(lines)
            else:
                self._rewrite(index, lines)
            return True
        if self._base is None or index == len(self._digests):
            return False
        self._rewrite(index, iter(()))
        return True

    def retry_pending(self) -> bool:
        """Publish the last rewrite whose atomic replace failed, from its
        temporary file; False when there is none."""
        if self._pending is None:
            return False
        digests, ends = self._pending
        os.replace(self._tmp, self.path)
        self._digests, self._ends, self._pending = digests, ends, None
        return True

    def _publish_pending(self, trajectory: Trajectory) -> bool:
        """Retry a failed replace whose file holds exactly *trajectory*."""
        assert self._pending is not None
        digests, ends = self._pending
        own = [
            _digest(json.dumps(exchange_record(exchange), default=str))
            for exchange in trajectory.exchanges
        ]
        if own != digests:
            return False
        os.replace(self._tmp, self.path)
        self._digests, self._ends, self._pending = digests, ends, None
        return True

    def _end(self) -> int:
        """The file's size as this writer left it."""
        return self._ends[-1] if self._ends else self._base or 0

    def _size(self) -> int | None:
        try:
            return self.path.stat().st_size
        except OSError:
            return None

    def _drop_pending(self) -> None:
        if self._pending is not None:
            self._pending = None
            self._tmp.unlink(missing_ok=True)

    def _settle_base(self) -> int:
        """The bytes of the preceding roles' lines this writer follows.

        Their writers published them to this file, so it keeps them as they
        are. A file that does not hold them all (an earlier role whose
        publish failed) gets them from memory only if they are whole there;
        otherwise this writer's lines follow what is there, with a warning.
        """
        expected = sum(len(t.exchanges) for t in self._preceding_trajectories)
        if expected == 0:
            return 0
        found, end = _line_prefix(self.path, expected)
        if found == expected:
            return end
        preceding = [
            exchange
            for trajectory in self._preceding_trajectories
            for exchange in trajectory.exchanges
        ]
        if not any(isinstance(e, PublishedLLMExchange) for e in preceding):
            with self._tmp.open("wb") as out:
                for n, exchange in enumerate(preceding):
                    _, line = _line(exchange_record(exchange))
                    out.write(b"\n" + line if n else line)
            os.replace(self._tmp, self.path)
            return self.path.stat().st_size
        logger.warning(
            "%s holds %d of the %d lines earlier roles published; this role's "
            "lines follow them",
            self.path,
            found,
            expected,
        )
        return end

    def _rewrite(self, index: int, lines: Iterable[tuple[bytes, bytes]]) -> None:
        """Keep the first *index* of this writer's lines, then write *lines*."""
        if self._base is None:
            self._base = self._settle_base()
        keep = self._ends[index - 1] if index else self._base
        digests, ends = self._digests[:index], self._ends[:index]
        with self._tmp.open("wb") as out:
            if keep:
                with self.path.open("rb") as current:
                    _copy(current, out, keep)
            offset = keep
            for digest, line in lines:
                if offset:
                    out.write(b"\n")
                    offset += 1
                out.write(line)
                offset += len(line)
                digests.append(digest)
                ends.append(offset)
        try:
            os.replace(self._tmp, self.path)
        except OSError:
            self._pending = (digests, ends)
            raise
        self._digests, self._ends = digests, ends

    def _append(self, lines: Iterable[tuple[bytes, bytes]]) -> None:
        """Add *lines* at the end, each in one write. A write that fails is
        cut off again, so the file ends with the last whole line and the
        offsets kept here stay true; a later reconcile adds what is missing.
        Nothing is added to a file that is not as this writer left it."""
        end = self._end()
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            if os.fstat(fd).st_size != end:
                raise OSError(f"{self.path} changed outside its writer")
            for digest, line in lines:
                data = b"\n" + line if end else line
                try:
                    _write_all(fd, data)
                except OSError:
                    with contextlib.suppress(OSError):
                        os.ftruncate(fd, end)
                    raise
                end += len(data)
                self._digests.append(digest)
                self._ends.append(end)
        finally:
            os.close(fd)


def _digest(raw: str) -> bytes:
    return hashlib.blake2b(raw.encode(), digest_size=16).digest()


def _line(record: dict[str, Any]) -> tuple[bytes, bytes]:
    """A record's digest and its redacted line."""
    raw = json.dumps(record, default=str)
    return _digest(raw), _redact(raw)


def _redact(raw: str) -> bytes:
    return _snapshot.redact_record(raw).encode()


def _redacted_lines(
    first: str, rest: Iterable[dict[str, Any]]
) -> Iterable[tuple[bytes, bytes]]:
    yield _digest(first), _redact(first)
    for record in rest:
        yield _line(record)


def _line_prefix(path: Path, lines: int) -> tuple[int, int]:
    """How many of the first *lines* lines *path* holds, and their bytes
    (without the newline after the last of them)."""
    found = end = line_start = position = 0
    try:
        with path.open("rb") as stream:
            while found < lines:
                block = stream.read(_COPY_CHUNK)
                if not block:
                    if position > line_start:
                        found, end = found + 1, position
                    break
                start = 0
                while found < lines:
                    cut = block.find(b"\n", start)
                    if cut < 0:
                        break
                    found, end = found + 1, position + cut
                    line_start, start = end + 1, cut + 1
                position += len(block)
    except OSError:
        return 0, 0
    return found, end


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _copy(source: Any, target: Any, size: int) -> None:
    while size > 0:
        block = source.read(min(size, _COPY_CHUNK))
        if not block:
            raise OSError("the published trajectory shrank while it was copied")
        target.write(block)
        size -= len(block)
