"""Hash Daytona 0.207.0 contexts in bounded, cancellable filesystem batches."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import logging
import os
import threading
import time
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import Executor, Future
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from importlib.metadata import version
from typing import Any, cast

from benchflow.sandbox.daytona_build_keys import sorted_walk

_PATCHED = "_benchflow_context_hash"
_PARAMETERS = ("self", "path_str", "archive_base_path")
_READ_SIZE = 8192
_BATCH_STEPS = 128
_BATCH_SECONDS = 0.01
logger = logging.getLogger("benchflow")


def _uses_stdlib_executor(loop: asyncio.AbstractEventLoop) -> bool:
    if not isinstance(loop, asyncio.BaseEventLoop):
        return False
    for name in (
        "run_in_executor",
        "set_default_executor",
        "shutdown_default_executor",
        "_check_default_executor",
        "_check_closed",
        "_check_callback",
        "_do_shutdown",
    ):
        original = getattr(asyncio.BaseEventLoop, name, None)
        if original is None or getattr(type(loop), name, None) is not original:
            return False
    return True


class _ContextReaders:
    """Transfer staging cleanup to the last physical reader when necessary."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._release: Callable[[], Any] | None = None
        self._closed = False

    def acquire(self) -> None:
        with self._lock:
            if self._closed:
                raise asyncio.CancelledError
            self._count += 1

    def finished(self) -> None:
        with self._lock:
            self._count -= 1
            release = self._release if self._count == 0 else None
            if release is not None:
                self._release = None
        if release is not None:
            release()

    def defer_release(self, release: Callable[[], Any]) -> bool:
        with self._lock:
            self._closed = True
            if self._count:
                self._release = release
                return True
            return False


_READERS: ContextVar[_ContextReaders | None] = ContextVar(
    "daytona_readers", default=None
)


@contextmanager
def context_readers() -> Iterator[_ContextReaders]:
    readers = _ContextReaders()
    token = _READERS.set(readers)
    try:
        yield readers
    finally:
        _READERS.reset(token)


def _hash_steps(
    storage: Any, path_str: str, archive: str | None
) -> Generator[None, None, str]:
    """Keep the SDK's materialized walk, names, bytes and open-file lifetime,
    with each folder listed in sorted order, so the key is
    :func:`benchflow.sandbox.daytona_build_keys.context_hash`'s, the one
    ``_sdk_ops`` gives Daytona on every filesystem (#35)."""
    digest = hashlib.md5()
    absolute = os.path.abspath(path_str)
    if archive is None:
        archive = storage.compute_archive_base_path(path_str)
    digest.update(archive.encode("utf-8"))

    def read_file(path: str) -> Iterator[None]:
        with open(path, "rb") as source:
            yield
            while chunk := source.read(_READ_SIZE):
                digest.update(chunk)
                yield
        yield

    if os.path.isfile(absolute):
        yield from read_file(absolute)
    else:
        # The async SDK captures the entire walk before reading file contents.
        walk = []
        for entry in sorted_walk(absolute):
            walk.append(entry)
            yield
        for root, directories, files in walk:
            if not directories and not files:
                digest.update(os.path.relpath(root, path_str).encode("utf-8"))
            yield
            for filename in files:
                path = os.path.join(root, filename)
                digest.update(os.path.relpath(path, absolute).encode("utf-8"))
                yield from read_file(path)
    return digest.hexdigest()


class _ObservedExecutor(Executor):
    """Observe default-pool work before asyncio wraps its terminal state."""

    def __init__(
        self, loop: asyncio.AbstractEventLoop, completed: Callable[[Future[Any]], None]
    ) -> None:
        self._loop = loop
        self._completed = completed

    def submit(
        self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Future[Any]:
        # BaseEventLoop exposes the default pool only through these internals.
        loop = cast(Any, self._loop)
        loop._check_default_executor()
        if loop._default_executor is None:
            self._loop.run_in_executor(None, lambda: None).add_done_callback(
                _consume_exception
            )
        completion = loop._default_executor.submit(fn, *args, **kwargs)
        completion.add_done_callback(self._completed)
        return completion


class _HashReader:
    def __init__(self, storage: Any, path: str, archive: str | None) -> None:
        self._steps = _hash_steps(storage, path, archive)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active = False
        self._running = False
        self._closing = False
        self._finished = False
        self._readers = _READERS.get()
        if self._readers is not None:
            self._readers.acquire()

    def _close(self) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
        try:
            self._steps.close()
        finally:
            try:
                if self._readers is not None:
                    self._readers.finished()
            finally:
                with self._lock:
                    self._finished = True
                    self._active = False
                    self._running = False

    def _batch(self) -> str | None:
        with self._lock:
            stopped = self._stop.is_set()
            if not stopped:
                self._running = True
        if stopped:
            self._close()
            return None
        completed = False
        try:
            deadline = time.monotonic() + _BATCH_SECONDS
            for _ in range(_BATCH_STEPS):
                if self._stop.is_set():
                    break
                try:
                    next(self._steps)
                except StopIteration as result:
                    completed = True
                    return result.value
                if time.monotonic() >= deadline:
                    break
            return None
        except BaseException:
            completed = True
            raise
        finally:
            with self._lock:
                close = completed or self._stop.is_set()
                if not close:
                    self._active = False
                    self._running = False
            if close:
                self._close()

    def _submission_failed(self) -> None:
        # submit can raise after enqueueing. Revoke an unclaimed batch before
        # closing; a job that already claimed the generator must close it.
        with self._lock:
            self._stop.set()
            close = not self._running
        if close:
            # Cleanup must not replace the original submission/cancellation.
            with suppress(BaseException):
                self._close()

    def _work_completed(self, completion: Future[Any]) -> None:
        if completion.cancelled() or completion.exception() is not None:
            self._submission_failed()

    async def hash(self) -> str:
        loop = asyncio.get_running_loop()
        executor = _ObservedExecutor(loop, self._work_completed)

        def submit(work: Callable[[], Any]) -> asyncio.Future[Any]:
            try:
                completion = loop.run_in_executor(executor, work)
            except BaseException:
                self._submission_failed()
                raise
            completion.add_done_callback(_consume_exception)
            return completion

        try:
            while True:
                with self._lock:
                    self._active = True
                result = await asyncio.shield(submit(self._batch))
                if result is not None:
                    return result
        except BaseException:
            # Do not drain: the SDK turns cancellation crossing its deadline
            # into a timeout. A running batch owns the file and staging instead.
            with self._lock:
                self._stop.set()
                close = not self._active and not self._finished
                if close:
                    self._active = True
            if close:
                with suppress(BaseException):
                    submit(self._close)
            raise


def _consume_exception(completion: asyncio.Future[Any]) -> None:
    if not completion.cancelled():
        completion.exception()


def patch_context_hashing() -> None:
    """Use the pinned SDK's hash contract off-loop; leave other SDKs alone."""
    if version("daytona") != "0.207.0":
        return
    from daytona._async.object_storage import AsyncObjectStorage
    from daytona._sync.object_storage import ObjectStorage

    storage = cast(Any, AsyncObjectStorage)
    original = getattr(storage, "_compute_hash_for_path_md5", None)
    synchronous = getattr(ObjectStorage, "_compute_hash_for_path_md5", None)
    if original is None or synchronous is None or getattr(original, _PATCHED, False):
        return
    try:
        supported = all(
            tuple(inspect.signature(method).parameters) == _PARAMETERS
            for method in (original, synchronous)
        )
    except (TypeError, ValueError):
        return
    if not supported:
        return

    @functools.wraps(original)
    async def hash_context(
        self: Any, path_str: str, archive_base_path: str | None = None
    ) -> str:
        loop = asyncio.get_running_loop()
        if not _uses_stdlib_executor(loop):
            logger.debug(
                "Using Daytona SDK context hashing for event loop %s.%s",
                type(loop).__module__,
                type(loop).__qualname__,
            )
            return await original(self, path_str, archive_base_path)
        return await _HashReader(self, path_str, archive_base_path).hash()

    setattr(hash_context, _PATCHED, True)
    storage._compute_hash_for_path_md5 = hash_context
