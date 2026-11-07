"""Byte-count reporting for streamed uploads."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import BinaryIO

_CHUNK_BYTES = 64 * 1024


class ProgressReader:
    """File-like proxy that reports bytes consumed to a callback.

    Supports both transport styles: httpx drains iterables, the Azure SDK
    drains ``read``. ``seek``/``tell`` are delegated so either consumer can
    still derive an exact Content-Length without emitting progress.
    """

    def __init__(self, stream: BinaryIO, on_bytes: Callable[[int], None]) -> None:
        self._stream = stream
        self._on_bytes = on_bytes

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        if chunk:
            self._on_bytes(len(chunk))
        return chunk

    def __iter__(self) -> Iterator[bytes]:
        while chunk := self._stream.read(_CHUNK_BYTES):
            self._on_bytes(len(chunk))
            yield chunk

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def tell(self) -> int:
        return self._stream.tell()
