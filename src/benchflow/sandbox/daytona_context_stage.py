"""Keep large Daytona context payloads unwritten on an SDK object-cache hit."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import os
import stat
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, BinaryIO, cast

from benchflow.sandbox.daytona_build_keys import sorted_walk
from benchflow.sandbox.daytona_context import _ContextReaders
from benchflow.task import staging

_MIN_DEFERRED_BYTES = 64 * 1024
_READ_SIZE = 1024 * 1024
_PATCHED = "_benchflow_staged_context_upload"
_CURRENT: ContextVar[_ContextStage | None] = ContextVar("daytona_stage", default=None)


@contextmanager
def staged_contexts(stage: _ContextStage | None) -> Iterator[None]:
    token = _CURRENT.set(stage)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@dataclass
class _Payload:
    sha256: bytes


@dataclass
class _ContextStage(staging.StagedEnvironment):
    pending: dict[str, _Payload] = field(default_factory=dict)
    keys: dict[tuple[str, str | None], str] = field(default_factory=dict)
    _readers: _ContextReaders = field(default_factory=_ContextReaders, repr=False)
    _materialize_lock: Any = field(default_factory=threading.Lock, repr=False)
    _failure: BaseException | None = field(default=None, repr=False)
    _context_payloads: set[str] = field(default_factory=set, repr=False)

    @property
    def daytona_contexts(self) -> _ContextStage | None:
        return self if self.keys else None

    def remove(self) -> None:
        if not self._readers.defer_release(super().remove):
            super().remove()

    def release(self) -> None:
        if not self._readers.defer_release(super().release):
            super().release()

    def changed(self) -> staging.StagingRefused:
        return staging.StagingRefused(
            [
                staging._refusal(
                    str(self.original_task_dir),
                    "changed while being copied; try again",
                )
            ],
            transient=True,
        )

    @contextmanager
    def source(self, relative: str) -> Iterator[BinaryIO]:
        # Reopen through pinned directories, never through a mutable ancestor.
        directory = staging._open_directory(os.sep)
        try:
            path = self.original_task_dir / relative
            for component in path.parts[1:-1]:
                child = staging._open_directory(component, directory)
                os.close(directory)
                directory = child
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                dir_fd=directory,
            )
            try:
                with os.fdopen(fd, "rb", closefd=False) as source:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise self.changed()
                    yield source
            finally:
                os.close(fd)
        except OSError as exc:
            raise self.changed() from exc
        finally:
            os.close(directory)

    def read_payload(
        self, relative: str, consume: Any, executor: ThreadPoolExecutor | None = None
    ) -> None:
        digest = hashlib.sha256()
        with self.source(relative) as source:
            while chunk := source.read(_READ_SIZE):
                if executor is None:
                    digest.update(chunk)
                    consume(chunk)
                else:
                    verification = executor.submit(digest.update, chunk)
                    consume(chunk)
                    verification.result()
        if digest.digest() != self.pending[relative].sha256:
            raise self.changed()

    def context_key(self, path: str, archive: str | None) -> str:
        """Hash the private tree as ``daytona_build_keys.context_hash`` does (the
        SDK's walk, each folder in sorted order: the key ``_sdk_ops`` gives
        Daytona), substituting captured bytes."""
        from daytona._async.object_storage import AsyncObjectStorage

        digest = hashlib.md5()
        absolute = os.path.abspath(path)
        relative = os.path.relpath(os.path.realpath(absolute), self.task_dir)
        if relative == os.curdir:
            relative = ""
        if archive is None:
            archive = AsyncObjectStorage.compute_archive_base_path(path)
        digest.update(archive.encode("utf-8"))

        def read_file(filename: str, relative: str) -> None:
            # The private walk never follows directory links. Only a leaf link
            # needs resolution after the context root has been resolved.
            if relative not in self.pending and os.path.islink(filename):
                relative = os.path.relpath(os.path.realpath(filename), self.task_dir)
            if relative in self.pending:
                self._context_payloads.add(relative)
                self.read_payload(relative, digest.update, executor)
            else:
                with open(filename, "rb") as source:
                    while chunk := source.read(_READ_SIZE):
                        digest.update(chunk)

        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="daytona-hash"
        ) as executor:
            if os.path.isfile(absolute):
                read_file(absolute, relative)
            else:
                for root, directories, files in list(sorted_walk(absolute)):
                    root_relative = os.path.relpath(root, absolute)
                    prefix = "" if root_relative == os.curdir else root_relative
                    if not directories and not files:
                        digest.update(os.path.relpath(root, path).encode("utf-8"))
                    for filename in files:
                        full = os.path.join(root, filename)
                        name = os.path.join(prefix, filename)
                        digest.update(name.encode("utf-8"))
                        read_file(full, os.path.join(relative, name))
        return digest.hexdigest()

    def materialize(self, only: set[str] | None = None) -> None:
        """Fill existing entries, so neither SDK order nor captured bytes change."""
        with self._materialize_lock:
            if self._failure is not None:
                raise self._failure
            try:
                for relative in list(self.pending):
                    if only is not None and relative not in only:
                        continue
                    target = self.task_dir / relative
                    fd = os.open(target, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
                    try:
                        with os.fdopen(fd, "wb", closefd=False) as destination:
                            info = os.fstat(fd)
                            destination.truncate(0)
                            self.read_payload(relative, destination.write)
                            destination.flush()
                            os.fchmod(fd, stat.S_IMODE(info.st_mode) | 0o600)
                            os.utime(fd, ns=(info.st_atime_ns, info.st_mtime_ns))
                    finally:
                        os.close(fd)
                    del self.pending[relative]
            except OSError as exc:
                self._failure = self.changed()
                raise self._failure from exc
            except BaseException as exc:
                self._failure = exc
                raise

    async def complete(self) -> None:
        self._readers.acquire()
        executor = None
        try:
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="daytona-copy"
            )
            work = executor.submit(self.materialize)
        except BaseException:
            self._readers.finished()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
        work.add_done_callback(lambda _: self._readers.finished())
        # Physical completion owns staging cleanup, including after cancellation.
        job = asyncio.wrap_future(work)
        job.add_done_callback(_consume_exception)
        await asyncio.shield(job)


def _consume_exception(job: asyncio.Future[Any]) -> None:
    if not job.cancelled():
        job.exception()


def _patch_upload() -> bool:
    try:
        if version("daytona") != "0.207.0":
            return False
    except PackageNotFoundError:
        return False
    try:
        from daytona._async.object_storage import AsyncObjectStorage
        from daytona._utils.otel_decorator import with_instrumentation
    except ImportError:
        return False

    original = AsyncObjectStorage.upload
    if getattr(original, _PATCHED, False):
        return True
    try:
        parameters = tuple(inspect.signature(original).parameters)
    except (TypeError, ValueError):
        return False
    if parameters != (
        "self",
        "path",
        "organization_id",
        "archive_base_path",
    ):
        return False

    @with_instrumentation()
    @functools.wraps(original)
    async def upload_context(
        self: Any,
        path: str,
        organization_id: str,
        archive_base_path: str | None,
        stage: _ContextStage,
        key: str,
    ) -> str:
        if stage._failure is not None:
            raise stage._failure
        if not await asyncio.to_thread(os.path.exists, path):
            raise FileNotFoundError(f"Path does not exist: {path}")
        s3_key = f"{organization_id}/{key}/context.tar"
        if await self._file_exists_in_s3(s3_key):
            return key
        await stage.complete()
        actual = await self._compute_hash_for_path_md5(path, archive_base_path)
        if actual != key:
            stage._failure = stage.changed()
            raise stage._failure
        await self._upload_as_tar(s3_key, path, archive_base_path)
        return key

    @functools.wraps(original)
    async def upload(
        self: Any, path: str, organization_id: str, archive_base_path: str | None = None
    ) -> str:
        stage = _CURRENT.get()
        if stage is not None:
            key = stage.keys.get((str(path), archive_base_path))
            if key is not None:
                return await upload_context(
                    self, path, organization_id, archive_base_path, stage, key
                )
            if stage._failure is not None:
                raise stage._failure
            # New recipe contexts may overlap captured files. Finish the copy
            # before allowing any unrecognized consumer to read this tree.
            resolved = await asyncio.to_thread(os.path.realpath, path)
            if staging._within(resolved, str(stage.root)) or staging._within(
                str(stage.root), resolved
            ):
                await stage.complete()
        return await original(self, path, organization_id, archive_base_path)

    setattr(upload, _PATCHED, True)
    storage = cast(Any, AsyncObjectStorage)
    storage.upload = upload
    return True


def stage_daytona_environment(
    environment_dir: Path, *, allow_outside: bool = False
) -> staging.StagedEnvironment:
    """Capture large payloads at staging; only SDK cache misses write their bytes."""
    from benchflow.sandbox import daytona_content_snapshots
    from benchflow.sandbox._compose import compose_definition_path

    if allow_outside or daytona_content_snapshots.enabled() or not _patch_upload():
        return staging.stage_environment(environment_dir, allow_outside=allow_outside)
    pending: dict[str, _Payload] = {}
    environment_name = Path(os.path.realpath(environment_dir)).name

    def copy_data(
        relative: str, source: int, destination: int, info: os.stat_result
    ) -> None:
        directory, separator, _ = relative.partition(os.sep)
        if (
            directory != environment_name
            or not separator
            or os.path.basename(relative)
            in (
                "Dockerfile",
                "docker-compose.yaml",
                "docker-compose.yml",
                "compose.yaml",
                "compose.yml",
            )
            or info.st_size < _MIN_DEFERRED_BYTES
        ):
            staging._copy_file_data(source, destination)
            return
        with os.fdopen(source, "rb", closefd=False) as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        pending[relative] = _Payload(digest)

    ordinary = staging.stage_environment(environment_dir, _copy_data=copy_data)
    stage = _ContextStage(
        ordinary.root,
        ordinary.original_environment_dir,
        ordinary.environment_dir,
        pending=pending,
    )
    try:
        if not pending:
            return stage
        if compose_definition_path(stage.environment_dir) is not None:
            stage.materialize()
            return stage
        from daytona import Image

        try:
            image = Image.from_dockerfile(stage.environment_dir / "Dockerfile")
            keys = {}
            for context in image._context_list:
                path = str(context.source_path)
                if not staging._within(os.path.realpath(path), str(stage.task_dir)):
                    stage.materialize()
                    return stage
                keys[path, context.archive_path] = stage.context_key(
                    path, context.archive_path
                )
        except staging.StagingRefused:
            raise
        except Exception:
            # Preserve the normal parser/hash diagnostics at their existing call.
            stage.materialize()
            return stage
        stage.keys = keys
        stage.materialize(set(stage.pending) - stage._context_payloads)
        return stage
    except BaseException:
        stage.remove()
        raise
