"""Controller I/O at sandbox-reachable folders and their private snapshots.

A rollout's ``agent/``, ``verifier/`` and ``artifacts/`` folders are
bind-mounted into Docker sandboxes (``/logs/agent``, ``/logs/verifier``,
``/logs/artifacts``), and on the other backends they are filled from the
sandbox's own files. While BenchFlow writes there, an agent process that is
still running (soft verification, a kept service, a watcher it left behind)
can have put a link, a hard link, a FIFO or a swapped folder at the name, so
``Path.write_text`` would truncate whatever the link names on the host
(CONFIRM-taskmd-rewards-R4-1).

Those directories are :class:`ReachablePath` capabilities, never filesystem
paths handed to writers (CONFIRM-taskmd-rewards-R5-1). Their methods write,
append, or publish through the helpers here. The root is opened once and
held, and the capability implements no filesystem-path conversion. Content
is written in a mode-700 folder beside the root,
outside every mount, then the fixed name is replaced by one
directory-relative rename through held, no-follow handles of each folder on
the way. No path is resolved again and no link is followed: a link or a FIFO
at the name is replaced as an entry, a hard link keeps its old inode, and a
folder on the way that was swapped while the write was pending refuses the
write with :class:`ReachableWriteError`.

Reads and copies use this same boundary. Private custody does not make a
restored symbolic link safe to follow: relocation can change its target
(CONFIRM-taskmd-rewards-R6-1). Snapshot traversal preserves links as entries
or refuses them, and reads regular files only from no-follow descriptors.
"""

from __future__ import annotations

import contextlib
import errno
import fnmatch
import io
import logging
import os
import shutil
import stat
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path, PurePath
from typing import IO, BinaryIO
from uuid import uuid4
from weakref import WeakSet, WeakValueDictionary, finalize

# The rollout folders a sandbox can reach, by their name under the rollout.
REACHABLE_ROLLOUT_DIRS = ("agent", "verifier", "artifacts")

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class UnsafeDownloadEntryError(ValueError):
    """Captured metadata proves a file/directory shape does not match the request."""


class ReachableWriteError(OSError):
    """A write into a sandbox-reachable folder was refused."""


class PrivateStorageError(OSError):
    """An operation on known controller-private storage failed."""


@contextlib.contextmanager
def private_storage_io(operation: str) -> Iterator[None]:
    """Mark a private operation, leaving source I/O and custody checks outside."""
    try:
        yield
    except PrivateStorageError:
        raise
    except OSError as exc:
        if isinstance(exc, (FileExistsError, NotADirectoryError)) or (
            isinstance(exc, ReachableWriteError)
            and exc.errno not in (errno.EACCES, errno.EPERM)
        ):
            raise
        if exc.errno == errno.ENAMETOOLONG:
            raise
        raise PrivateStorageError(
            exc.errno, f"controller private storage {operation} failed"
        ) from exc


@contextlib.contextmanager
def _private_lookup_io(operation: str, *, private: bool = True) -> Iterator[None]:
    """Type actual private lookup I/O, never an explicit custody refusal."""
    if not private:
        yield
        return
    try:
        yield
    except (PrivateStorageError, ReachableWriteError, FileNotFoundError):
        raise
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EXDEV):
            raise
        with private_storage_io(operation):
            raise


def read_private_text(
    path: Path, *, errors: str = "strict", universal_newlines: bool = True
) -> str:
    """Read a completed controller snapshot; decoding stays outside private I/O."""
    stack = contextlib.ExitStack()
    with private_storage_io("snapshot open"):
        stream = stack.enter_context(path.open("rb"))
    try:
        with private_storage_io("snapshot read"):
            data = stream.read()
    finally:
        with private_storage_io("snapshot close"):
            stack.close()
    text = data.decode("utf-8", errors=errors)
    return (
        text.replace("\r\n", "\n").replace("\r", "\n") if universal_newlines else text
    )


def private_stat(target: ReachablePath) -> os.stat_result:
    """Inspect a known private destination through its existing custody guards."""
    with (
        target._folders(directory=not target._parts, private=True) as handles,
        _private_lookup_io("capture destination stat"),
    ):
        if not target._parts:
            return (
                handles[-1].lstat()
                if isinstance(handles[-1], Path)
                else os.fstat(handles[-1])
            )
        return _at_stat(target.name, handles[-1])


@contextlib.contextmanager
def private_temporary_directory(*, prefix: str) -> Iterator[str]:
    """Type temporary creation/cleanup, never errors from the caller's body."""
    # Location discovery/refusal is not an attempted private allocation.
    parent = Path(tempfile.gettempdir())
    if _inside_declared_mount(parent):
        raise ReachableWriteError("temporary storage is inside a sandbox mount")
    with private_storage_io("temporary directory create"):
        directory = tempfile.TemporaryDirectory(prefix=prefix, dir=parent)
    try:
        yield directory.name
    finally:
        with private_storage_io("temporary directory cleanup"):
            directory.cleanup()


class PrivateStagingWriteError(PrivateStorageError):
    """Creation in a validated private staging parent failed."""

    attempts: tuple[OSError, ...] = ()


@contextlib.contextmanager
def _staging_io(operation: str) -> Iterator[None]:
    try:
        yield
    except (ReachableWriteError, FileExistsError, NotADirectoryError):
        # A collision or failed custody check remains a refusal, even below
        # a parent selected for private creation.
        raise
    except OSError as exc:
        raise PrivateStagingWriteError(
            exc.errno, f"controller private staging {operation} failed"
        ) from exc


class PrivateSpoolError(PrivateStorageError):
    """An anonymous controller spool could not be created, written or read."""


@contextlib.contextmanager
def _spool_io(operation: str) -> Iterator[None]:
    try:
        yield
    except PrivateStorageError:
        raise
    except OSError as exc:
        raise PrivateSpoolError(
            exc.errno, f"controller private spool {operation} failed"
        ) from exc


class _PrivateSpool[T: (str, bytes)]:
    """Keep provenance on each private stream operation, never its caller's I/O."""

    def __init__(
        self,
        stream: IO[T],
        *,
        boundary: Callable[[str], contextlib.AbstractContextManager[None]] = _spool_io,
    ) -> None:
        self._stream = stream
        self._io = boundary

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def read(self, size: int = -1) -> T:
        with self._io("read"):
            return self._stream.read(size)

    def write(self, data: T) -> int:
        with self._io("write"):
            count = self._stream.write(data)
            if count != len(data):
                raise OSError(errno.EIO, "short private spool write")
            return count

    def flush(self) -> None:
        with self._io("flush"):
            self._stream.flush()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        with self._io("seek"):
            return self._stream.seek(offset, whence)

    def tell(self) -> int:
        with self._io("tell"):
            return self._stream.tell()

    def fileno(self) -> int:
        with self._io("descriptor"):
            return self._stream.fileno()

    def close(self) -> None:
        with self._io("close"):
            self._stream.close()


@contextlib.contextmanager
def _private_spool() -> Iterator[_PrivateSpool[bytes]]:
    with _spool_io("create"):
        stream = _PrivateSpool(tempfile.TemporaryFile())  # noqa: SIM115 - closed below
    try:
        yield stream
    finally:
        stream.close()


@contextlib.contextmanager
def _copy_source(
    fd: int, *, private: bool
) -> Iterator[BinaryIO | _PrivateSpool[bytes]]:
    # Publication duplicates a held descriptor. Preserve the private source's
    # provenance through that duplication without relabelling destination I/O.
    with _spool_io("duplicate") if private else contextlib.nullcontext():
        duplicate = os.dup(fd)
        try:
            stream = os.fdopen(duplicate, "rb")
        except BaseException:
            os.close(duplicate)
            raise
    source = _PrivateSpool(stream) if private else stream
    try:
        yield source
    finally:
        source.close()


def _open_regular(name: str, dir_fd: int) -> int:
    """Open regular file ``name`` in ``dir_fd`` for reading, never following a link.

    Anything else raises :class:`ReachableWriteError` without being opened: a device
    node a leftover agent process swaps into a live bind mount would run its
    driver's open (a watchdog starts on open) before any ``fstat`` could
    refuse it. On Linux the leaf is pinned with ``O_PATH`` (no open of the
    inode itself), checked, and that same inode reopened through
    ``/proc/self/fd``. Elsewhere it is checked with ``lstat`` first and the
    open must yield that inode, which narrows the swap to a race.
    """
    flags = os.O_RDONLY | _NOFOLLOW | os.O_NONBLOCK | _CLOEXEC
    if getattr(os, "O_PATH", 0) and os.path.isdir("/proc/self/fd"):
        pinned = os.open(
            name, getattr(os, "O_PATH", 0) | _NOFOLLOW | _CLOEXEC, dir_fd=dir_fd
        )
        try:
            if not stat.S_ISREG(os.fstat(pinned).st_mode):
                raise ReachableWriteError(errno.EINVAL, "not a regular file", name)
            return os.open(f"/proc/self/fd/{pinned}", flags & ~_NOFOLLOW)
        finally:
            os.close(pinned)
    before = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ReachableWriteError(errno.EINVAL, "not a regular file", name)
    fd = os.open(name, flags, dir_fd=dir_fd)
    after = os.fstat(fd)
    if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
        os.close(fd)
        raise ReachableWriteError(errno.EINVAL, "changed while opened", name)
    return fd


def _held_handles_available() -> bool:
    return (
        bool(_NOFOLLOW)
        and bool(_DIRECTORY)
        and os.rename in os.supports_dir_fd
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    )


def _parts(path: Path, root: Path) -> tuple[str, ...]:
    relative = Path(os.path.abspath(path)).relative_to(os.path.abspath(root))
    parts = relative.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{path} is not a file under {root}")
    return parts


def _same(a: os.stat_result, b: os.stat_result) -> bool:
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _open_dir(name: str | PurePath, dir_fd: int | None = None) -> int:
    flags = os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC
    if dir_fd is None:
        return os.open(name, flags)
    return os.open(name, flags, dir_fd=dir_fd)


# openat2(2) resolution for every step below a trusted root: no
# symbolic link, no magic link, no mount crossing, never above the folder.
_RESOLVE_NO_XDEV = 0x01
_RESOLVE_NO_MAGICLINKS = 0x02
_RESOLVE_NO_SYMLINKS = 0x04
_RESOLVE_BENEATH = 0x08
_RESOLVE = (
    _RESOLVE_NO_XDEV | _RESOLVE_NO_MAGICLINKS | _RESOLVE_NO_SYMLINKS | _RESOLVE_BENEATH
)
_SYS_OPENAT2 = 437  # one number on every Linux architecture since 5.6
_openat2_usable: list[bool | None] = [None]


def _openat2(dir_fd: int, name: str, flags: int, mode: int) -> int | None:
    """``openat2`` with :data:`_RESOLVE`; None where the kernel or seccomp lacks it."""
    if not sys.platform.startswith("linux") or _openat2_usable[0] is False:
        return None
    import ctypes

    class OpenHow(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_uint64),
            ("mode", ctypes.c_uint64),
            ("resolve", ctypes.c_uint64),
        ]

    libc = ctypes.CDLL(None, use_errno=True)
    tmpfile = getattr(os, "O_TMPFILE", 0)
    # A mode only with O_CREAT or O_TMPFILE (whose bits include O_DIRECTORY).
    creates = bool(flags & os.O_CREAT) or bool(tmpfile and flags & tmpfile == tmpfile)
    how = OpenHow(flags, mode if creates else 0, _RESOLVE)
    fd = libc.syscall(
        ctypes.c_long(_SYS_OPENAT2),
        ctypes.c_int(dir_fd),
        ctypes.c_char_p(os.fsencode(name)),
        ctypes.byref(how),
        ctypes.c_size_t(ctypes.sizeof(how)),
    )
    if fd >= 0:
        _openat2_usable[0] = True
        return fd
    err = ctypes.get_errno()
    if err in (errno.ENOSYS, errno.EPERM) and _openat2_usable[0] is not True:
        # Older kernel, or a seccomp profile without openat2: the portable walk.
        _openat2_usable[0] = False
        return None
    raise OSError(err, os.strerror(err), name)


def _open_beneath(
    dir_fd: int, name: str, flags: int, mode: int = 0o600, *, where: str | Path
) -> int:
    """Open ``name`` in ``dir_fd`` without following a link or crossing a mount.

    Linux uses ``openat2`` (:data:`_RESOLVE`); elsewhere, or where it is
    filtered, the component is opened with O_NOFOLLOW and its st_dev (and,
    on Linux, its mount ID) must match the folder above it. A link or a mount
    refuses with :class:`ReachableWriteError` naming ``where``.
    """
    if not name or name in (".", "..") or "/" in name or os.sep in name:
        raise ValueError(f"unsafe path component at {where}")
    try:
        fd = _openat2(dir_fd, name, flags | _CLOEXEC, mode)
        if fd is None:
            fd = os.open(name, flags | _NOFOLLOW | _CLOEXEC, mode, dir_fd=dir_fd)
            try:
                crossed = os.fstat(fd).st_dev != os.fstat(dir_fd).st_dev
                if not crossed and sys.platform.startswith("linux"):
                    above, here = _mount_id(dir_fd), _mount_id(fd)
                    crossed = above is not None and here is not None and above != here
            except BaseException:
                os.close(fd)
                raise
            if crossed:
                os.close(fd)
                raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), name)
        return fd
    except OSError as exc:
        if isinstance(exc, ReachableWriteError):
            raise
        if exc.errno in (errno.ENOTDIR, getattr(errno, "EMLINK", -1)):
            # O_DIRECTORY|O_NOFOLLOW on a link: ENOTDIR (Linux), EMLINK (BSD).
            with contextlib.suppress(OSError):
                if stat.S_ISLNK(
                    os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
                ):
                    exc = OSError(errno.ELOOP, os.strerror(errno.ELOOP), name)
        if exc.errno == errno.ELOOP:
            raise ReachableWriteError(
                exc.errno,
                f"{where} is a symbolic link: BenchFlow follows no link below a "
                "destination root",
            ) from exc
        if exc.errno == errno.EXDEV:
            raise ReachableWriteError(
                exc.errno,
                f"{where} is another mount: BenchFlow crosses no mount below a "
                "destination root",
            ) from exc
        raise


def _remove_beneath(parent_fd: int, name: str, *, where: str | Path) -> None:
    """Remove the entry ``name`` (a tree too) without following a link or crossing a mount."""
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return
    fd = _open_beneath(parent_fd, name, os.O_RDONLY | _DIRECTORY, where=where)
    try:
        if info.st_mode & 0o300 != 0o300:
            # A received read-only folder (mode 0555): its children cannot be
            # unlinked until it is writable again (TASKMD-R14 B-L1).
            with contextlib.suppress(OSError):
                os.fchmod(fd, stat.S_IMODE(info.st_mode) | 0o700)
        for child in os.listdir(fd):
            _remove_beneath(fd, child, where=f"{where}/{child}")
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent_fd)


@contextlib.contextmanager
def _held_folders(
    root: Path,
    folders: tuple[str, ...],
    *,
    parents: bool,
    root_fd: int | None = None,
    private: bool = False,
):
    """Hold the root and each folder under it, opened without following a link."""
    try:
        with _private_lookup_io("capture root open", private=private):
            handles = [_open_dir(root) if root_fd is None else os.dup(root_fd)]
    except PrivateStorageError:
        raise
    except OSError as exc:
        raise ReachableWriteError(
            exc.errno, f"{root} is not a real folder ({exc.strerror})"
        ) from exc
    try:
        where = Path(root)
        for part in folders:
            where = where / part
            flags = os.O_RDONLY | _DIRECTORY
            try:
                try:
                    with _private_lookup_io("capture folder open", private=private):
                        child = _open_beneath(handles[-1], part, flags, where=where)
                except FileNotFoundError:
                    if not parents:
                        raise
                    with (
                        contextlib.suppress(FileExistsError),
                        _private_lookup_io("capture folder create", private=private),
                    ):
                        os.mkdir(part, 0o777, dir_fd=handles[-1])
                    with _private_lookup_io("capture folder open", private=private):
                        child = _open_beneath(handles[-1], part, flags, where=where)
            except (PrivateStorageError, FileNotFoundError, ReachableWriteError):
                raise
            except OSError as exc:
                raise ReachableWriteError(
                    exc.errno,
                    f"{part} under {root} is a link or a file, not a folder ({exc.strerror})",
                ) from exc
            handles.append(child)
        yield handles
    finally:
        for handle in reversed(handles):
            with _private_lookup_io("capture folder close", private=private):
                os.close(handle)


def _check_in_place(
    root: Path, folders: tuple[str, ...], handles: list[int], *, private: bool = False
) -> None:
    """Refuse when a held folder is no longer the entry its path names."""
    try:
        with _private_lookup_io("capture root identity read", private=private):
            here = os.stat(root, follow_symlinks=False)
            held = os.fstat(handles[0])
        if not _same(here, held):
            raise ReachableWriteError(f"{root} was swapped while BenchFlow wrote in it")
        for index, part in enumerate(folders):
            with _private_lookup_io("capture folder identity read", private=private):
                entry = os.stat(part, dir_fd=handles[index], follow_symlinks=False)
                held = os.fstat(handles[index + 1])
            if not _same(entry, held):
                raise ReachableWriteError(
                    f"{part} under {root} was swapped while BenchFlow wrote in it"
                )
    except (PrivateStorageError, ReachableWriteError):
        raise
    except OSError as exc:
        raise ReachableWriteError(
            exc.errno, f"a folder under {root} was moved while BenchFlow wrote in it"
        ) from exc


@contextlib.contextmanager
def private_folder(root: Path | ReachablePath) -> Iterator[Path]:
    """A mode-700 folder beside ``root``: on its filesystem, outside every mount.

    Download or build a sandbox file or tree here, then :func:`place_reachable`
    it: nothing is ever opened by a name the sandbox can reach. Removed, with
    anything still in it, on exit.
    """
    if isinstance(root, ReachablePath):
        root = root._anchor.path
    root = Path(os.path.abspath(root))
    device = os.stat(root).st_dev if os.path.lexists(root) else None
    # Publication renames into ``root``: same filesystem, no temp fallback.
    with _private_staging(_owners(root.parent), device, temp=False) as folder:
        yield folder


def _owners(start: Path) -> list[Path]:
    """``start`` and each folder above it: nearest staging candidates first."""
    return [start, *start.parents]


class _PrivateStage:
    """A staging directory whose parent and own handle survive until cleanup."""

    def __init__(self, path: Path, parent_context, handles, capability: ReachablePath):
        self.path = path
        self.parent_context = parent_context
        self.handles = handles
        self.capability = capability

    def close(self) -> None:
        try:
            parent = self.handles[-1]
            if isinstance(parent, Path):
                self.capability._anchor.finalizer()
                shutil.rmtree(self.path, ignore_errors=True)
            else:
                try:
                    _remove_beneath(parent, self.path.name, where=self.path)
                except OSError:
                    # A retained sandbox writer can mutate a retired tree.
                    # Refuse that cleanup, preserving the published result and
                    # the caller's original exception/cancellation, if any.
                    logging.getLogger(__name__).warning(
                        "Private staging cleanup refused a changed tree: %s", self.path
                    )
        finally:
            self.capability._anchor.finalizer()
            self.parent_context.__exit__(None, None, None)


def _make_private_in(
    owner: Path, device: int | None, reasons: list[str]
) -> _PrivateStage | None:
    """Create staging below an already pinned root, retaining both directory handles."""
    context = None
    entered = False
    created: tuple[Path | int, str, Path] | None = None
    try:
        owner_cap = _staging_owner(owner)
        canonical = owner_cap._anchor.path.joinpath(*owner_cap._parts)
        if _inside_declared_mount(canonical):
            reasons.append(f"{owner} is a folder the harness mounts into a sandbox")
            return None
        context = owner_cap._folders(directory=True)
        handles = context.__enter__()
        entered = True
        parent = handles[-1]
        info = os.stat(parent) if isinstance(parent, Path) else os.fstat(parent)
        if device is not None and info.st_dev != device:
            reasons.append(f"{owner} is on another filesystem")
            context.__exit__(None, None, None)
            return None
        name = f".benchflow-private-{uuid4().hex}"
        path = canonical / name
        # Selection, mount/device checks and parent pinning above are refusals.
        # Only the actual creation in that validated parent is a private write.
        with _staging_io("mkdir"):
            if isinstance(parent, Path):
                path.mkdir(mode=0o700)
            else:
                os.mkdir(name, 0o700, dir_fd=parent)
        created = (parent, name, path)
        if isinstance(parent, Path):
            with _staging_io("open"):
                capability = ReachablePath.directory(path)
                capability._anchor.controller_owned = True
        else:
            with _staging_io("open"):
                fd = _open_beneath(parent, name, os.O_RDONLY | _DIRECTORY, where=path)
            anchor = _Anchor.__new__(_Anchor)
            anchor.path = path
            anchor.controller_owned = True
            anchor.reachable = False
            anchor.moved_to = None
            anchor.staging_parent = None
            anchor.fd = fd
            anchor.windows = None
            anchor.opened = True
            anchor.finalizer = finalize(anchor, os.close, fd)
            capability = ReachablePath(anchor)
        return _PrivateStage(path, context, handles, capability)
    except OSError as exc:
        if created is not None:
            # No member has been written yet. Remove only our still-empty
            # directory entry; rmdir never follows a replacement symlink.
            created_parent, created_name, created_path = created
            try:
                if isinstance(created_parent, Path):
                    created_path.rmdir()
                else:
                    os.rmdir(created_name, dir_fd=created_parent)
            except OSError as cleanup:
                exc.add_note(f"private staging cleanup failed: {cleanup}")
        if entered and context is not None:
            context.__exit__(None, None, None)
        if isinstance(exc, PrivateStorageError):
            raise
        reasons.append(f"{owner}: {exc.strerror or exc}")
        return None


@contextlib.contextmanager
def _private_staging(
    owners: list[Path], device: int | None, *, temp: bool
) -> Iterator[Path]:
    """Expose a staging spelling while retaining its no-follow directory capability."""
    reasons: list[str] = []
    creation_failures: list[PrivateStagingWriteError] = []
    stage = None
    candidates = [*owners, *([Path(tempfile.gettempdir())] if temp else [])]
    for owner in candidates:
        try:
            stage = _make_private_in(owner, device, reasons)
        except PrivateStagingWriteError as exc:
            creation_failures.append(exc)
            continue
        if stage is not None:
            break
    if stage is None:
        if creation_failures:
            # A safe fallback may succeed. If none does, retain every actual
            # creation failure and chain the first original host-write cause.
            error = creation_failures[0]
            error.attempts = tuple(
                failure.__cause__
                for failure in creation_failures
                if isinstance(failure.__cause__, OSError)
            )
            for later in creation_failures[1:]:
                error.add_note(
                    f"another private staging attempt failed: {later.__cause__}"
                )
            raise error
        raise ReachableWriteError(
            "no staging folder private to the harness: " + "; ".join(reasons)
        )
    with _DESTINATION_LOCK:
        _DESTINATION_ROOTS[stage.path] = stage.capability._anchor
    try:
        yield stage.path
    finally:
        try:
            stage.close()
        finally:
            with _DESTINATION_LOCK:
                _DESTINATION_ROOTS.pop(stage.path, None)


def _retire(name: str, folder_fd: int, private_fd: int) -> None:
    """Move whatever is at ``name`` into private custody, as an entry."""
    with contextlib.suppress(FileNotFoundError):
        os.rename(
            name,
            f".retired-{uuid4().hex}",
            src_dir_fd=folder_fd,
            dst_dir_fd=private_fd,
        )


def _install(
    source: Path | ReachablePath,
    path: Path,
    *,
    root: Path,
    parents: bool,
    refuse_special: bool = False,
    root_fd: int | None = None,
) -> None:
    """Publish through held source AND destination parents, without path reopens."""
    *above, name = _parts(path, root)
    folders = tuple(above)
    staged = destination(source)
    with staged._folders() as source_handles:
        private = source_handles[-1]
        if isinstance(private, Path) or not _held_handles_available():
            _install_by_path(
                staged._anchor.path.joinpath(*staged._parts),
                path,
                root=root,
                folders=folders,
                parents=parents,
                refuse_special=refuse_special,
            )
            return
        tree = stat.S_ISDIR(_at_stat(staged.name, private).st_mode)
        with _held_folders(root, folders, parents=parents, root_fd=root_fd) as handles:
            _check_in_place(root, folders, handles)
            try:
                try:
                    existing = os.stat(name, dir_fd=handles[-1], follow_symlinks=False)
                except FileNotFoundError:
                    existing = None
                if (
                    refuse_special
                    and existing is not None
                    and not stat.S_ISREG(existing.st_mode)
                ):
                    raise ReachableWriteError(
                        f"{name} under {root} is a link or a special file"
                    )
                if existing is not None and (tree or stat.S_ISDIR(existing.st_mode)):
                    _retire(name, handles[-1], private)
                os.rename(staged.name, name, src_dir_fd=private, dst_dir_fd=handles[-1])
            except OSError as exc:
                raise ReachableWriteError(
                    exc.errno,
                    f"BenchFlow did not place {name} under {root} ({exc.strerror})",
                ) from exc
            _check_in_place(root, folders, handles)


def _install_by_path(
    source: Path,
    path: Path,
    *,
    root: Path,
    folders: tuple[str, ...],
    parents: bool,
    refuse_special: bool = False,
) -> None:
    """Controllers without directory handles (Windows): check each folder, then rename.

    ``os.replace`` replaces a link at the name as an entry; a folder swapped
    between the check and the rename is not caught here.
    """
    here = Path(os.path.abspath(root))
    for part in ("", *folders):
        if part:
            here = here / part
            if parents and not os.path.lexists(here):
                with contextlib.suppress(FileExistsError):
                    here.mkdir()
        info = os.stat(here, follow_symlinks=False)
        reparse = getattr(info, "st_file_attributes", 0) & 0x400
        if not stat.S_ISDIR(info.st_mode) or reparse:
            raise ReachableWriteError(f"{here} is not a real folder")
    target = here / path.name
    if (
        refuse_special
        and os.path.lexists(target)
        and not stat.S_ISREG(os.stat(target, follow_symlinks=False).st_mode)
    ):
        raise ReachableWriteError(
            f"{path.name} under {root} is a link or a special file"
        )
    try:
        if os.path.isdir(target) and not os.path.islink(target):
            os.replace(target, source.parent / f".retired-{uuid4().hex}")
        os.replace(source, target)
    except OSError as exc:
        raise ReachableWriteError(
            exc.errno,
            f"BenchFlow did not place {path.name} under {root} ({exc.strerror})",
        ) from exc


def write_reachable(
    path: Path,
    data: str | bytes,
    *,
    root: Path,
    mode: int = 0o644,
    parents: bool = False,
    refuse_special: bool = False,
) -> None:
    """Replace the file at ``path`` (under the reachable folder ``root``) with ``data``.

    ``parents`` makes missing folders between ``root`` and the file, each
    through the held handle of the one above it. ``refuse_special`` refuses
    when a link, a FIFO or a folder is at the name instead of replacing it.
    """
    try:
        target = ReachablePath.directory(root) / Path(*_parts(path, root))
    except OSError as exc:
        raise ReachableWriteError(
            exc.errno, "publication root is not a real folder"
        ) from exc
    target.write(data, mode=mode, refuse_special=refuse_special)


def _read_existing(path: Path, *, root: Path) -> bytes:
    """The regular file at ``path`` read through held folders, or b"" when absent."""
    from benchflow.rewards._safe_evidence import open_file

    _parts(path, root)
    try:
        with open_file(path, evidence_root=root) as stream:
            return stream.read(os.fstat(stream.fileno()).st_size)
    except FileNotFoundError:
        return b""
    except OSError as exc:
        raise ReachableWriteError(f"{path.name} is not a safe regular input") from exc


def append_reachable(path: Path, data: str | bytes, *, root: Path) -> None:
    """Append ``data`` to the file at ``path`` by placing a new copy.

    The old content is read through the held folders without following a link
    (a link or a special file there refuses); old and new are written
    privately and placed like :func:`write_reachable`, so the append never
    opens an inode the sandbox supplied for writing.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    write_reachable(
        path,
        _read_existing(path, root=root) + payload,
        root=root,
        refuse_special=True,
    )


def place_reachable(
    path: Path, source: Path, *, root: Path, parents: bool = False
) -> None:
    """Move the private file or tree ``source`` to ``path`` under ``root``.

    ``source`` sits in a :func:`private_folder` of the same root (one
    filesystem, never a sandbox-reachable folder); its links are moved as
    they are, never followed.
    """
    info = os.stat(source, follow_symlinks=False)
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ReachableWriteError(f"{source} is not a regular file or a folder")
    _install(source, path, root=root, parents=parents)


def place_reachable_entries(
    folder: Path, source_dir: Path, *, root: Path, parents: bool = False
) -> None:
    """Place each entry of the private ``source_dir`` at the same name in ``folder``.

    For a download of a whole sandbox folder (``/logs/verifier``) over the
    rollout folder that mirrors it: each top-level entry replaces its own name.
    """
    for child in sorted(source_dir.iterdir()):
        # A link the sandbox made is moved as a link, never followed.
        _install(child, folder / child.name, root=root, parents=parents)


def rename_reachable(path: Path, new_name: str, *, root: Path) -> None:
    """Rename the entry at ``path`` to ``new_name`` in the same folder under ``root``.

    One rename through the held folder: neither name is resolved as a path,
    and an entry already at ``new_name`` is replaced as an entry.
    """
    if Path(new_name).name != new_name or new_name in (".", ".."):
        raise ValueError("rename_reachable takes a single new name")
    *above, name = _parts(path, root)
    folders = tuple(above)
    if not _held_handles_available():
        _install_by_path(
            path, path.with_name(new_name), root=root, folders=folders, parents=False
        )
        return
    with _held_folders(root, folders, parents=False) as handles:
        _check_in_place(root, folders, handles)
        os.rename(name, new_name, src_dir_fd=handles[-1], dst_dir_fd=handles[-1])
        _check_in_place(root, folders, handles)


class _Anchor:
    """One mounted directory, pinned for the lifetime of all its capabilities."""

    def __init__(
        self, path: Path, *, controller_owned: bool = False, reachable: bool = False
    ):
        # Only the parent is controller-owned and may be an operator's alias.
        # A drive, UNC share or / has no name: zero components (R9-3).
        self.path = path.parent.resolve() / path.name if path.name else path
        # A controller-owned anchor (a destination's canonical parent) is
        # never reachable from a sandbox; staging may live inside it.
        self.controller_owned = controller_owned
        # A rollout folder the harness mounts into (or fills from) a sandbox:
        # recorded when opened so rename staging never lands in it.
        self.reachable = reachable
        # Where BenchFlow itself moved this folder (note_reachable_move).
        self.moved_to: Path | None = None
        # A pinned root's parent, held for rename staging (TASKMD-R14 B-M3).
        self.staging_parent: _Anchor | None = None
        self.fd = None
        self.windows = None
        self.opened = False
        self._open()

    def _open(self) -> None:
        if self.opened:
            return
        if not os.path.lexists(self.path):
            return
        self.fd = _open_dir(self.path) if _held_handles_available() else None
        if self.fd is not None:
            self.finalizer = finalize(self, os.close, self.fd)
            if self.reachable:
                _DECLARED_MOUNTS.add(_identity(os.fstat(self.fd)))
                _REACHABLE_ANCHORS.add(self)
        elif os.name == "nt":
            from benchflow.rewards._safe_evidence import open_directory

            context = open_directory(self.path, evidence_root=self.path)
            self.windows = context.__enter__()
            self.finalizer = finalize(self, context.__exit__, None, None, None)
            if self.reachable:
                _DECLARED_MOUNTS.add(_identity(os.stat(self.path)))
        else:
            raise ReachableWriteError(
                "safe reachable directories require filesystem handles"
            )
        self.opened = True

    def ensure(self) -> None:
        if not self.opened:
            self.path.mkdir(parents=True, exist_ok=True)
            self._open()


def _at_stat(name, directory):
    if isinstance(directory, Path):
        return (directory / name).lstat()
    return os.stat(name, dir_fd=directory, follow_symlinks=False)


def _at_unlink(name, directory):
    if isinstance(directory, Path):
        return (directory / name).unlink()
    return os.unlink(name, dir_fd=directory)


def _at_rename(name, destination, source_dir, dest_dir=None):
    if isinstance(source_dir, Path):
        return os.replace(
            source_dir / name, dest_dir / destination if dest_dir else destination
        )
    return os.rename(name, destination, src_dir_fd=source_dir, dst_dir_fd=dest_dir)


def _at_readlink(name, directory):
    if isinstance(directory, Path):
        return os.readlink(directory / name)
    return os.readlink(name, dir_fd=directory)


class ReachablePath:
    """A held directory and names under it, deliberately NOT an os.PathLike.

    No filesystem path is exported. A callee can only read through no-follow
    handles, publish a completed private entry, or write a fresh private inode.
    Joining a name returns another capability, never a Path or string. The
    mounted root is opened once before the sandbox starts and stays held even
    if the sandbox replaces an entry later (CONFIRM-taskmd-rewards-R5-1).
    """

    def __init__(self, anchor: _Anchor, parts: tuple[str, ...] = ()):
        self._anchor = anchor
        self._parts = parts

    @classmethod
    def directory(cls, path: Path) -> ReachablePath:
        """Boundary for a caller that owns a directory before sandbox launch."""
        return cls(_Anchor(path))

    @classmethod
    def rollout(cls, rollout_dir: Path, name: str) -> ReachablePath:
        if name not in REACHABLE_ROLLOUT_DIRS:
            raise ValueError("not a sandbox-reachable rollout directory")
        return cls(_Anchor(rollout_dir / name, reachable=True))

    def __truediv__(self, name: str | PurePath) -> ReachablePath:
        parts = Path(name).parts
        if (
            not parts
            or Path(name).is_absolute()
            or any(p in ("..", ".") for p in parts)
        ):
            raise ValueError(
                "a reachable name must be relative and stay in its directory"
            )
        if self._anchor.windows is not None:
            from benchflow.rewards._safe_evidence import _windows_component

            for part in parts:
                _windows_component(part)
        return ReachablePath(self._anchor, self._parts + parts)

    def __str__(self) -> str:
        return f"<reachable:{'/'.join((self._anchor.path.name, *self._parts))}>"

    def __repr__(self) -> str:
        return str(self)

    def details_message(self) -> str:
        """A complete diagnostic, not a reusable filesystem pathname."""
        return f"(details: {self._anchor.path.joinpath(*self._parts)})"

    def __lt__(self, other: ReachablePath) -> bool:
        return self._parts < other._parts

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ReachablePath) and (
            self._anchor.path,
            self._parts,
        ) == (other._anchor.path, other._parts)

    def __hash__(self) -> int:
        return hash((self._anchor.path, self._parts))

    @property
    def name(self) -> str:
        return self._parts[-1] if self._parts else self._anchor.path.name

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix

    @property
    def stem(self) -> str:
        return Path(self.name).stem

    @property
    def parent(self) -> ReachablePath | Path:
        # The root's parent is private, outside every sandbox mount.
        return (
            ReachablePath(self._anchor, self._parts[:-1])
            if self._parts
            else self._anchor.path.parent
        )

    def relative_to(self, other: ReachablePath | Path) -> Path:
        if isinstance(other, Path):
            # The anchor's path is absolute, its parent resolved (_Anchor):
            # compare a relative or unresolved folder in that spelling too,
            # or a relative --jobs-dir fails every caller (DF-p4-upgrader-2).
            # Spelled as the anchor is: absolute, only the parent resolved, so
            # a link at other's own name is not followed (review DX-e L4).
            here = self._anchor.path.joinpath(*self._parts)
            if not other.is_absolute() or not here.is_relative_to(other):
                spelled = Path(os.path.abspath(other))
                if spelled.name:
                    spelled = spelled.parent.resolve() / spelled.name
                other = spelled
            return here.relative_to(other)
        if self._anchor.path != other._anchor.path:
            raise ValueError("different reachable directories")
        return Path(*self._parts).relative_to(Path(*other._parts))

    def is_relative_to(self, other: ReachablePath | Path) -> bool:
        try:
            self.relative_to(other)
        except ValueError:
            return False
        return True

    def with_name(self, name: str) -> ReachablePath:
        if not self._parts or Path(name).name != name:
            raise ValueError("expected a single entry name")
        return ReachablePath(self._anchor, self._parts[:-1]) / name

    @contextlib.contextmanager
    def _folders(
        self, *, directory: bool = False, parents: bool = False, private: bool = False
    ):
        if not self._anchor.opened:
            with _private_lookup_io("capture anchor open", private=private):
                self._anchor._open()
        if not self._anchor.opened:
            raise FileNotFoundError(self._anchor.path)
        parts = self._parts if directory else self._parts[:-1]
        if self._anchor.fd is None:
            # Windows denies write/delete sharing on each held ancestor. The
            # root stays open for the capability lifetime; child opens are
            # relative to that root and reject every reparse-point spelling.
            root = self._anchor.windows
            if root is None:
                raise ReachableWriteError("directory handles are unavailable")
            api = root.api
            opened = []
            paths = [self._anchor.path]
            parent = root.handle
            try:
                for part in parts:
                    here = paths[-1] / part
                    if parents:
                        with _private_lookup_io(
                            "capture folder create", private=private
                        ):
                            here.mkdir(exist_ok=True)
                    with _private_lookup_io("capture folder open", private=private):
                        child = api.open(Path(part), directory=True, parent=parent)
                    opened.append(child)
                    paths.append(here)
                    parent = child
                yield paths
            finally:
                for child in reversed(opened):
                    with _private_lookup_io("capture folder close", private=private):
                        api.close(child)
            return
        with _held_folders(
            self._anchor.path,
            parts,
            parents=parents,
            root_fd=self._anchor.fd,
            private=private,
        ) as handles:
            _check_in_place(self._anchor.path, parts, handles, private=private)
            yield handles
            _check_in_place(self._anchor.path, parts, handles, private=private)

    def mkdir(
        self, parents: bool = False, exist_ok: bool = False, *, private: bool = False
    ) -> None:
        # Opening walks only real directories and creates missing ones through
        # the held parent. The root already exists and cannot be recreated.
        with _private_lookup_io("capture anchor create", private=private):
            self._anchor.ensure()
        with self._folders(directory=True, parents=True, private=private):
            pass

    def stat(self, *, follow_symlinks: bool = False) -> os.stat_result:
        with self._folders(directory=not self._parts) as handles:
            if not self._parts:
                return (
                    handles[-1].lstat()
                    if isinstance(handles[-1], Path)
                    else os.fstat(handles[-1])
                )
            return _at_stat(self.name, handles[-1])

    def lstat(self) -> os.stat_result:
        return self.stat()

    def exists(self) -> bool:
        try:
            self.stat()
        except (FileNotFoundError, NotADirectoryError):
            return False
        return True

    def is_file(self) -> bool:
        try:
            return stat.S_ISREG(self.stat().st_mode)
        except (FileNotFoundError, NotADirectoryError):
            return False

    def is_dir(self) -> bool:
        try:
            return stat.S_ISDIR(self.stat().st_mode)
        except (FileNotFoundError, NotADirectoryError):
            return False

    def is_symlink(self) -> bool:
        try:
            return stat.S_ISLNK(self.stat().st_mode)
        except (FileNotFoundError, NotADirectoryError):
            return False

    def open(self, mode: str = "r", *, encoding: str = "utf-8", errors: str = "strict"):
        """Read only. In particular, open(handle, 'w') cannot obtain a path."""
        if mode not in ("r", "rb") or not self._parts:
            raise ValueError("reachable open is only for reading regular files")
        fd = None
        try:
            with self._folders() as handles:
                if isinstance(handles[-1], Path):
                    from benchflow.rewards._safe_evidence import open_directory

                    # The root and parents remain pinned while open_file acquires
                    # the leaf with no reparse traversal and no write sharing.
                    with open_directory(
                        handles[-1], evidence_root=self._anchor.path
                    ) as directory:
                        source = directory.open_file(self.name)
                    return (
                        source
                        if mode == "rb"
                        else io.TextIOWrapper(source, encoding=encoding, errors=errors)
                    )
                fd = _open_regular(self.name, handles[-1])
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise ReachableWriteError("reachable input is not a regular file")
            if mode == "rb":
                return os.fdopen(fd, "rb")
            return os.fdopen(fd, "r", encoding=encoding, errors=errors)
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise

    def read_bytes(self) -> bytes:
        with self.open("rb") as stream:
            return stream.read(os.fstat(stream.fileno()).st_size)

    def read_text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        return self.read_bytes().decode(encoding, errors)

    def write(
        self, data: str | bytes, *, refuse_special: bool = False, mode: int = 0o644
    ) -> None:
        self._anchor.ensure()
        payload = data.encode("utf-8") if isinstance(data, str) else data
        with _private_spool() as source:
            source.write(payload)
            source.seek(0)
            _publish_stream(source, self, mode=mode, refuse_special=refuse_special)

    def write_text(
        self,
        data: str,
        encoding: str = "utf-8",
        errors: str = "strict",
        newline: str | None = None,
    ) -> int:
        # Match Path.write_text's newline contract in private memory. The
        # encoded bytes still enter the mounted tree only through publication.
        buffer = io.BytesIO()
        with io.TextIOWrapper(
            buffer, encoding=encoding, errors=errors, newline=newline
        ) as stream:
            written = stream.write(data)
            stream.flush()
            self.write(buffer.getvalue())
            return written

    def write_bytes(self, data: bytes) -> int:
        self.write(data)
        return len(data)

    def append(self, data: str | bytes) -> None:
        try:
            previous = self.read_bytes()
        except FileNotFoundError:
            previous = b""
        except PermissionError as exc:
            # This read touches the existing entry, before private staging or
            # publication. Only a sandbox-reachable entry is submitted evidence;
            # the same denial on a private session file is a controller failure.
            if not self._anchor.reachable:
                raise
            raise ReachableWriteError(
                errno.EACCES, "unreadable sandbox output entry", self.name
            ) from exc
        self.write(
            previous + (data.encode() if isinstance(data, str) else data),
            refuse_special=True,
        )

    def private_folder(self):
        return private_folder(self)

    def publish(
        self, source: Path | ReachablePath, *, refuse_special: bool = False
    ) -> None:
        if not self._parts:
            raise ValueError("cannot replace a mounted root")
        self._anchor.ensure()
        # No pathname fallback: the handle's authority must survive every call.
        if self._anchor.fd is None:
            staged = destination(source)
            with self._folders(parents=True), staged._folders():
                _install_by_path(
                    staged._anchor.path.joinpath(*staged._parts),
                    self._anchor.path.joinpath(*self._parts),
                    root=self._anchor.path,
                    folders=self._parts[:-1],
                    parents=False,
                    refuse_special=refuse_special,
                )
            return
        _install(
            source,
            self._anchor.path.joinpath(*self._parts),
            root=self._anchor.path,
            parents=True,
            refuse_special=refuse_special,
            root_fd=self._anchor.fd,
        )

    def publish_entries(self, source: Path) -> None:
        for child in sorted(source.iterdir()):
            (self / child.name).publish(child)

    def unlink(self, missing_ok: bool = False, *, private: bool = False) -> None:
        if not self._parts:
            raise ValueError("cannot unlink a mounted root")
        try:
            with (
                self._folders(private=private) as handles,
                _private_lookup_io("capture entry removal", private=private),
            ):
                _at_unlink(self.name, handles[-1])
        except FileNotFoundError:
            if not missing_ok:
                raise

    def rename(self, target: ReachablePath | Path) -> None:
        if isinstance(target, ReachablePath):
            with self._folders() as source, target._folders(parents=True) as dest:
                _at_rename(self.name, target.name, source[-1], dest[-1])
        else:
            # A private custody destination; never used to publish shared data.
            with self._folders() as source:
                _at_rename(self.name, target, source[-1])

    def rename_entry(self, name: str) -> None:
        self.rename(self.with_name(name))

    def iterdir(self) -> Iterator[ReachablePath]:
        with self._folders(directory=True) as handles:
            names = os.listdir(handles[-1])
        for name in names:
            yield self / name

    def glob(self, pattern: str) -> Iterator[ReachablePath]:
        for child in self.iterdir():
            if fnmatch.fnmatchcase(child.name, pattern):
                yield child

    def rglob(self, pattern: str) -> Iterator[ReachablePath]:
        for child in self.iterdir():
            if fnmatch.fnmatchcase(child.name, pattern):
                yield child
            if child.is_dir():
                yield from child.rglob(pattern)

    def copy_to(self, dest: Path) -> None:
        """Copy evidence into private custody without exporting shared paths."""
        snapshot_tree(self, dest)

    def readlink_text(self) -> str:
        with self._folders() as handles:
            return _at_readlink(self.name, handles[-1])

    def log(self, *, append: bool = False, initial: str = "") -> ReachableLog:
        return ReachableLog(self, append=append, initial=initial)

    def is_under(self, parent: Path) -> bool:
        return self._anchor.path.is_relative_to(parent.resolve())


def _publish_stream(
    source: BinaryIO | _PrivateSpool[bytes],
    target: ReachablePath,
    *,
    mode: int = 0o644,
    refuse_special: bool = False,
) -> None:
    """Publish from a held stream; never reopen a staging pathname for writing."""
    target._anchor.ensure()
    with target._folders(parents=True) as handles:
        parent = handles[-1]
        if refuse_special:
            try:
                info = _at_stat(target.name, parent)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(info.st_mode):
                    raise ReachableWriteError(
                        "publication target is not a regular file"
                    )
        if not isinstance(parent, Path):
            try:
                existing = _at_stat(target.name, parent)
            except FileNotFoundError:
                existing = None
            if existing is not None and stat.S_ISDIR(existing.st_mode):
                _remove_beneath(parent, target.name, where=target.details_message())
            _copy_opened(
                source.fileno(),
                parent,
                target.name,
                mode=mode,
                private_source=isinstance(source, _PrivateSpool),
            )
            return
        # Windows holds the root and every parent without delete sharing.
        # Exclusive creation never truncates an agent-supplied entry.
        temporary = parent / f".benchflow-copy-{uuid4().hex}"
        try:
            with temporary.open("xb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
            os.replace(temporary, parent / target.name)
        finally:
            temporary.unlink(missing_ok=True)


class ReachableLog(io.TextIOBase):
    """Private streaming log; each flush publishes a fresh, completed copy.

    Never read the published entry to append: the sandbox can replace it at
    any time. Shutdown stderr goes to the same private stream as stdout.
    """

    def __init__(
        self, target: ReachablePath, *, append: bool = False, initial: str = ""
    ):
        self.target = target
        target._anchor.ensure()
        with _spool_io("create"):
            self._stream = _PrivateSpool(
                tempfile.TemporaryFile(  # noqa: SIM115 - lifetime owned by close()
                    mode="w+", encoding="utf-8", errors="backslashreplace"
                )
            )
        self._failed = False
        self._closing = False
        if append:
            try:
                with target.open(errors="replace") as previous:
                    info = os.fstat(previous.fileno())
                    if info.st_nlink == 1:
                        remaining = info.st_size
                        while remaining and (
                            chunk := previous.read(min(65536, remaining))
                        ):
                            self._stream.write(chunk)
                            remaining -= len(chunk)
            except PrivateStorageError:
                raise
            except OSError:
                pass
        self._stream.write(initial)

    def write(self, text: str) -> int:
        if self._failed:
            raise ReachableWriteError("this log's publication was refused")
        return self._stream.write(text)

    def flush(self) -> None:
        if self.closed:
            return
        if self._failed:
            if self._closing:
                return
            raise ReachableWriteError("this log's publication was refused")
        self._stream.flush()
        position = self._stream.tell()
        try:
            self._stream.seek(0)
            with _copy_source(self._stream.fileno(), private=True) as source:
                _publish_stream(source, self.target)
        except BaseException:
            self._failed = True
            raise
        finally:
            self._stream.seek(position)

    def close(self) -> None:
        if self.closed:
            return
        try:
            # TextIOBase.close calls flush; close before removing private custody.
            self._closing = True
            super().close()
        finally:
            self._stream.close()


def mount_environment(
    values: dict[str, object], *, include_os_env: bool
) -> dict[str, str]:
    """Serialize handles only at the Docker environment boundary, never for I/O."""
    result = dict(os.environ) if include_os_env else {}
    for name, value in values.items():
        if isinstance(value, ReachablePath):
            value._anchor.ensure()
            result[name.upper()] = _mount_source(value)
        elif value is not None:
            result[name.upper()] = str(value)
    return result


def _mount_source(value: ReachablePath) -> str:
    """The folder's path for Compose, refused when it is not the folder held.

    After BenchFlow's own move (:func:`note_reachable_move`) that is where the
    folder is now, checked the same way: the mount follows the folder, so
    teardown's Compose calls name it truly (DF-1).
    """
    anchor = value._anchor
    moved = anchor.moved_to
    if moved is None or anchor.fd is None:
        with value._folders(directory=True):
            return str(anchor.path)
    with _held_folders(moved, (), parents=False, root_fd=anchor.fd) as handles:
        _check_in_place(moved, (), handles)
    return str(moved)


def note_reachable_move(destination: Path) -> None:
    """BenchFlow itself moved a sandbox-reachable rollout folder to ``destination``.

    A sandbox that mounts it (Docker's ``/logs/verifier``) keeps the folder:
    the separate verifier moves the agent's ``verifier/`` aside while the
    agent's container may still run (the serial order, or a stop not yet
    confirmed). Its Compose variables then name the folder where it is, so
    teardown never reads BenchFlow's own move as a swap and skips
    ``compose down`` (DF-1). Matched by identity, never by name. Reads and
    writes through the capability still refuse: they never follow the folder
    out of its place.
    """
    try:
        moved = os.stat(destination, follow_symlinks=False)
    except OSError:
        return
    for anchor in list(_REACHABLE_ANCHORS):
        try:
            held = os.fstat(anchor.fd) if anchor.fd is not None else None
        except OSError:
            continue
        if held is not None and _same(held, moved):
            anchor.moved_to = Path(os.path.abspath(destination))


def replace_verifier_root(root: Path, attempt: Path) -> None:
    """Recovery retires a complete mounted root before publishing its revision."""
    canonical = root / "verifier"
    staged = root / f".verifier-{attempt.name}.staging"
    previous = attempt / "previous-verifier"
    shutil.rmtree(staged, ignore_errors=True)
    try:
        snapshot_tree(
            source_capability(attempt) / "verifier",
            source_capability(root) / staged.name,
        )
        if canonical.is_dir():
            canonical.rename(previous)
        try:
            staged.rename(canonical)
        except BaseException:
            if previous.is_dir() and not canonical.exists():
                previous.rename(canonical)
            raise
    finally:
        shutil.rmtree(staged, ignore_errors=True)


def rollout_input(rollout_dir: Path, relative: PurePath) -> Path | ReachablePath:
    """Resolve a declared evidence name without leaking a mounted pathname."""
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("expected a relative rollout evidence name")
    first, *rest = relative.parts
    if first in REACHABLE_ROLLOUT_DIRS:
        root = ReachablePath.rollout(rollout_dir, first)
        return root / Path(*rest) if rest else root
    return rollout_dir / relative


class SnapshotLimitError(OSError):
    """A regular input exceeded the caller's snapshot budget."""


class PrivateSnapshotWriteError(PrivateStorageError):
    """The controller could not create or write its private snapshot destination."""


@contextlib.contextmanager
def _snapshot_write() -> Iterator[None]:
    """Mark destination I/O only; source reads retain their evidence provenance."""
    try:
        yield
    except PrivateStorageError:
        raise
    except OSError as exc:
        if isinstance(exc, (FileExistsError, NotADirectoryError)) or (
            isinstance(exc, ReachableWriteError)
            and exc.errno not in (errno.EACCES, errno.EPERM)
        ):
            raise
        if exc.errno == errno.ENAMETOOLONG:
            raise ReachableWriteError("unrepresentable snapshot entry name") from exc
        failure = PrivateSnapshotWriteError(
            exc.errno, f"private snapshot destination write failed: {exc}"
        )
        for note in getattr(exc, "__notes__", ()):
            failure.add_note(note)
        raise failure from exc


@contextlib.contextmanager
def _snapshot_output(target: Path | ReachablePath) -> Iterator[BinaryIO]:
    stack = contextlib.ExitStack()
    with _snapshot_write():
        output = stack.enter_context(create_private_file(target))
    try:
        yield output
    finally:
        with _snapshot_write():
            stack.close()


@contextlib.contextmanager
def create_private_file(target: Path | ReachablePath) -> Iterator[BinaryIO]:
    """Create an exclusive staging inode below a held no-follow parent.

    The returned stream owns that inode even if an agent renames its entry.
    Metadata must be applied to this descriptor, never by reopening its name.
    """
    capability = destination(target)
    with capability._folders(parents=True, private=True) as handles:
        parent = handles[-1]
        if isinstance(parent, Path):
            with (parent / capability.name).open("xb") as stream:
                yield stream
        else:
            fd = _open_beneath(
                parent,
                capability.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                where=capability.details_message(),
            )
            with os.fdopen(fd, "wb") as stream:
                yield stream


@contextlib.contextmanager
def private_output(
    context: contextlib.AbstractContextManager[BinaryIO],
) -> Iterator[_PrivateSpool[bytes]]:
    """A known private destination; each stream operation retains its cause."""
    stack = contextlib.ExitStack()
    with _private_lookup_io("file create"):
        output = stack.enter_context(context)
    try:
        yield _PrivateSpool(output, boundary=private_storage_io)
    finally:
        with _private_lookup_io("file close"):
            stack.close()


def _snapshot_target(target: Path | ReachablePath) -> ReachablePath:
    return destination(target)


def _snapshot_stream(
    source: BinaryIO, destination: Path | ReachablePath, *, max_bytes: int | None = None
) -> None:
    """Copy one already-open regular inode into a new private file, bounded by fstat.

    Never reopen a source by name. Exclusive destination creation also refuses
    a link left by an earlier entry when callers accidentally reuse a tree.
    """
    info = os.fstat(source.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise ReachableWriteError("snapshot input must be a regular file")
    if max_bytes is not None and info.st_size > max_bytes:
        raise SnapshotLimitError("input exceeds the snapshot byte limit")
    with _snapshot_output(destination) as output:
        remaining = info.st_size
        while remaining:
            block = source.read(min(65536, remaining))
            if not block:
                raise ReachableWriteError("input shrank during snapshot")
            with _snapshot_write():
                output.write(block)
            remaining -= len(block)
        if os.fstat(source.fileno()).st_size != info.st_size:
            raise ReachableWriteError("input changed size during snapshot")
        with _snapshot_write():
            output.flush()
            if hasattr(os, "fchmod"):
                os.fchmod(output.fileno(), stat.S_IMODE(info.st_mode))
            if os.utime in os.supports_fd:
                os.utime(output.fileno(), ns=(info.st_atime_ns, info.st_mtime_ns))


@contextlib.contextmanager
def read_stream(
    source: Path | ReachablePath, *, evidence_root: Path | None = None
) -> Iterator[BinaryIO]:
    """Read the opened no-follow inode directly; never materialize a snapshot."""
    from benchflow.rewards._safe_evidence import open_file

    reader = (
        source.open("rb")
        if isinstance(source, ReachablePath)
        else open_file(source, evidence_root=evidence_root)
    )
    with reader as stream:
        yield stream


@contextlib.contextmanager
def snapshot_file(
    source: Path | ReachablePath,
    *,
    evidence_root: Path | None = None,
    max_bytes: int | None = None,
) -> Iterator[Path]:
    """Give a path-based consumer a private no-follow snapshot of a regular file.

    Raw paths are anchored component by component by the portable handle layer;
    mounted paths retain their original capability. Links and special files
    are refused, including substitution between enumeration and opening.
    """
    with (
        read_stream(source, evidence_root=evidence_root) as stream,
        private_temporary_directory(prefix="benchflow-snapshot-") as temp,
    ):
        target = Path(temp) / source.name
        _snapshot_stream(stream, target, max_bytes=max_bytes)
        yield target


@contextlib.contextmanager
def snapshot_text(
    source: Path | ReachablePath, *, evidence_root: Path | None = None, limit: int
) -> Iterator[str]:
    """Bounded text input through the same no-follow source handles as snapshots."""
    with read_stream(source, evidence_root=evidence_root) as stream:
        yield io.TextIOWrapper(stream, errors="replace").read(limit)


def snapshot_deliverables(
    directory: Path,
    *,
    evidence_root: Path | None,
    accept: Callable[[str], bool],
    max_bytes: int,
) -> Iterator[tuple[str, Path]]:
    """Yield private snapshots of admitted direct children; refuse unsafe entries."""
    from benchflow.rewards._safe_evidence import open_directory

    with open_directory(directory, evidence_root=evidence_root) as folder:
        for name in sorted(folder.names()):
            if not accept(name):
                continue
            try:
                stream = folder.open_file(name)
            except (OSError, ValueError):
                logging.getLogger(__name__).warning(
                    "judge deliverables: skipping unsafe file %s", name
                )
                continue
            with (
                stream,
                private_temporary_directory(prefix="benchflow-deliverable-") as temp,
            ):
                target = Path(temp) / name
                try:
                    _snapshot_stream(stream, target, max_bytes=max_bytes)
                except PrivateStorageError:
                    raise
                except OSError:
                    continue
                yield name, target


def _snapshot_source(source: Path | ReachablePath) -> ReachablePath:
    if isinstance(source, ReachablePath):
        return source
    # Do not resolve the source's parent: it too may contain a sandbox link.
    absolute = Path(os.path.abspath(source))
    anchored = ReachablePath.directory(Path(absolute.anchor))
    return anchored / Path(*absolute.parts[1:]) if absolute.parts[1:] else anchored


def snapshot_tree(
    source: Path | ReachablePath,
    destination: Path | ReachablePath,
    *,
    refuse_links: bool = False,
) -> None:
    """Copy to a new private tree, preserving links only as uninterpreted entries.

    This is the only recursive copier for sandbox-derived trees, including
    restored output already in private custody (R6-1). Every regular leaf is
    read through a no-follow descriptor. Destination entries are created once;
    a preserved link can never become the parent of a later copied file.
    """
    destination = _snapshot_target(destination)
    source = _snapshot_source(source)
    info = source.lstat()
    if stat.S_ISDIR(info.st_mode):
        with _snapshot_write(), destination._folders() as parents:
            parent = parents[-1]
            if isinstance(parent, Path):
                (parent / destination.name).mkdir()
            else:
                os.mkdir(destination.name, 0o700, dir_fd=parent)
        for child in source.iterdir():
            snapshot_tree(child, destination / child.name, refuse_links=refuse_links)
        with _snapshot_write(), destination._folders(directory=True) as handles:
            if not isinstance(handles[-1], Path):
                os.fchmod(handles[-1], stat.S_IMODE(info.st_mode))
                os.utime(handles[-1], ns=(info.st_atime_ns, info.st_mtime_ns))
    elif stat.S_ISLNK(info.st_mode):
        if refuse_links:
            raise ReachableWriteError("download source contains a symbolic link")
        link = source.readlink_text()
        with _snapshot_write(), destination._folders() as parents:
            parent = parents[-1]
            if isinstance(parent, Path):
                (parent / destination.name).symlink_to(link)
            else:
                os.symlink(link, destination.name, dir_fd=parent)
    else:
        with source.open("rb") as stream:
            _snapshot_stream(stream, destination)


def _open_snapshot_directory(path: Path) -> int:
    """Open every component without following a symbolic link."""
    if os.name != "posix" or not all(
        hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    ):
        raise ValueError(
            "safe artifact traversal requires POSIX no-follow descriptors; "
            "run the upload on Linux, macOS or WSL"
        )
    parts = Path(os.path.abspath(path)).parts
    descriptor = os.open(parts[0], os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[1:]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ValueError(f"unsafe or unreadable source folder: {path}") from exc


def snapshot_regular_files(
    folder: Path,
    *,
    exclude: Callable[[Path, bool], bool] | None = None,
) -> Iterator[tuple[Path, BinaryIO]]:
    """Yield relative paths and open regular files, valid until the next item.

    A renamed ancestor cannot redirect traversal; a replaced leaf cannot become
    a link or block on a FIFO. Special files and multiply linked files are omitted.
    """

    def walk(descriptor: int, relative: Path):
        with os.scandir(descriptor) as entries:
            for entry in entries:
                path = relative / entry.name
                try:
                    info = entry.stat(follow_symlinks=False)
                    directory = stat.S_ISDIR(info.st_mode)
                    if exclude is not None and exclude(path, directory):
                        continue
                    if directory:
                        child = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor,
                        )
                        try:
                            yield from walk(child, path)
                        finally:
                            os.close(child)
                    elif stat.S_ISREG(info.st_mode):
                        child = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                            dir_fd=descriptor,
                        )
                        with os.fdopen(child, "rb") as stream:
                            current = os.fstat(stream.fileno())
                            if stat.S_ISREG(current.st_mode) and current.st_nlink == 1:
                                yield path, stream
                except OSError:
                    # Optional artifacts can disappear or become unreadable.
                    continue

    descriptor = _open_snapshot_directory(folder)
    try:
        yield from walk(descriptor, Path())
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def snapshot_folder(folder: Path) -> Iterator[Path]:
    """A private, regular-file-only snapshot for path-based upload clients."""
    with tempfile.TemporaryDirectory(prefix="benchflow-publish-") as temporary:
        staged = Path(temporary)
        for relative, stream in snapshot_regular_files(folder):
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _snapshot_stream(stream, destination)
        yield staged


def require_descriptor_io() -> None:
    """Refuse unsupported hosts before any download or judge call."""
    if (
        not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not {"open", "mkdir", "unlink", "rename"}
        <= {getattr(function, "__name__", "") for function in os.supports_dir_fd}
        or not getattr(shutil.rmtree, "avoids_symlink_attacks", False)
    ):
        raise RuntimeError(
            "Safe host artifact writes require directory descriptors; "
            "run the controller on Linux, macOS or WSL"
        )


# Open rollout anchors, retained only to track harness-owned moves.
_REACHABLE_ANCHORS: WeakSet[_Anchor] = WeakSet()


def _identity(info: os.stat_result) -> tuple[int, int]:
    return (info.st_dev, info.st_ino)


def _mount_id(fd: int) -> int | None:
    """The mount a held descriptor is on, from /proc/self/fdinfo (Linux)."""
    try:
        with open(f"/proc/self/fdinfo/{fd}") as info:
            for line in info:
                if line.startswith("mnt_id:"):
                    return int(line.split(":", 1)[1])
    except (OSError, ValueError):
        return None
    return None


# Folders the harness itself bind-mounts into its sandboxes (a rollout's
# agent/, verifier/ and artifacts/, Apple's whole rollout folder), by identity.
# Used only to keep rename staging out of them: publication never asks
# whether a sandbox can reach a path.
_DECLARED_MOUNTS: set[tuple[int, int]] = set()


def declare_harness_mount(path: str | Path) -> None:
    """Record a folder the harness mounts into a sandbox, if it exists."""
    with contextlib.suppress(OSError):
        _DECLARED_MOUNTS.add(_identity(os.stat(path, follow_symlinks=False)))


def _inside_declared_mount(path: Path) -> bool:
    for folder in (path, *path.parents):
        with contextlib.suppress(OSError):
            if _identity(os.stat(folder)) in _DECLARED_MOUNTS:
                return True
    return False


def _trusted_anchor(root: Path, *, create: bool = True) -> _Anchor:
    """The trusted root, canonicalised once with OS aliases resolved.

    macOS /tmp and /var/folders, or an operator's symlinked jobs folder, are
    resolved here; nothing below the root is ever resolved again.
    """
    canonical = Path(os.path.realpath(Path(root).absolute()))
    if create:
        canonical.mkdir(parents=True, exist_ok=True)
    elif not canonical.is_dir():
        raise FileNotFoundError(f"transfer source {root} does not exist")
    return _Anchor(canonical, controller_owned=True)


# Only the controller registers roots, before untrusted execution. Keep both
# spellings so an operator alias is never resolved again during publication.
_DESTINATION_ROOTS: WeakValueDictionary[Path, _Anchor] = WeakValueDictionary()
_DESTINATION_LOCK = threading.RLock()


def pin_destination(root: Path) -> ReachablePath:
    """Pin a caller-chosen output directory BEFORE starting an agent.

    Carry the capability through transfers, or use plain paths below its
    registered spelling. Other output filesystems need explicit registration;
    a download never trusts its mutable parent as a new root.
    """
    spelling = Path(os.path.abspath(root))
    with _DESTINATION_LOCK:
        anchor = _DESTINATION_ROOTS.get(spelling)
        if anchor is None:
            anchor = _trusted_anchor(spelling)
            _DESTINATION_ROOTS[spelling] = anchor
            _DESTINATION_ROOTS.setdefault(anchor.path, anchor)
            _hold_staging_parent(spelling, anchor)
        return ReachablePath(anchor)


# Parents of pinned roots, held at pin time for rename staging only, never
# as destinations (TASKMD-R14 B-M3): a root the harness mounts whole (Apple's
# /logs) stages in its writable parent, as on integ, on any filesystem.
_STAGING_PARENTS: WeakValueDictionary[Path, _Anchor] = WeakValueDictionary()


def _hold_staging_parent(spelling: Path, anchor: _Anchor) -> None:
    if not anchor.path.name:
        return
    try:
        parent = _Anchor(anchor.path.parent, controller_owned=True)
    except OSError:
        return
    if not parent.opened:
        return
    # The root anchor keeps its parent's handle alive for as long as it lives.
    anchor.staging_parent = parent
    for name in (spelling.parent, parent.path):
        _STAGING_PARENTS.setdefault(name, parent)


def _staging_owner(owner: Path) -> ReachablePath:
    """The held capability for a staging candidate folder.

    A parent held when its child root was pinned is used directly; any other
    folder is selected below a pinned destination root (no link, no mount).
    """
    with _DESTINATION_LOCK:
        held = _STAGING_PARENTS.get(Path(os.path.abspath(owner)))
    if held is not None:
        return ReachablePath(held)
    # The sentinel selects the existing owner capability; it is never created.
    owner_cap = destination(owner / ".benchflow-stage-anchor").parent
    assert isinstance(owner_cap, ReachablePath)
    return owner_cap


def destination(
    target: Path | ReachablePath, *, root: Path | None = None
) -> ReachablePath:
    """Select a pinned destination without resolving its mutable parent.

    Explicit ``root`` authorizes a caller boundary before agent execution;
    carry the returned capability thereafter. Plain paths select only pinned
    roots, with no-follow/no-xdev traversal below them.
    """
    if isinstance(target, ReachablePath):
        if not target._parts:
            raise ValueError("cannot replace a mounted root")
        return target
    absolute = Path(os.path.abspath(target))
    if ".." in Path(target).parts or not absolute.name:
        raise ValueError("Unsafe copy destination")
    if root is not None:
        base = Path(os.path.abspath(root))
        capability = pin_destination(base)
    else:
        with _DESTINATION_LOCK:
            candidates = [
                (base, anchor)
                for base, anchor in _DESTINATION_ROOTS.items()
                if absolute.is_relative_to(base)
            ]
        if not candidates:
            raise ReachableWriteError("pin the destination root before agent execution")
        base, anchor = max(candidates, key=lambda item: len(item[0].parts))
        capability = ReachablePath(anchor)
    relative = absolute.relative_to(base)
    if not relative.parts:
        raise ValueError("cannot replace a pinned destination root")
    return capability / relative


def source_capability(root: Path | ReachablePath) -> ReachablePath:
    """A transfer source root: canonicalised once, everything below it no-follow."""
    if isinstance(root, ReachablePath):
        return root
    return ReachablePath(_trusted_anchor(Path(root), create=False))


def _destination_device(capability: ReachablePath) -> int:
    """st_dev of the folder the final entry goes in (or the deepest existing one).

    Folders still to be made inherit the filesystem of the folder they are
    made in, so this is the destination's own filesystem (R9-4).
    """
    anchor = capability._anchor
    if anchor.fd is None:
        here = anchor.path
        for part in capability._parts[:-1]:
            if not (here / part).is_dir() or (here / part).is_symlink():
                break
            here = here / part
        return os.stat(here).st_dev
    fd = os.dup(anchor.fd)
    try:
        for part in capability._parts[:-1]:
            try:
                child = _open_dir(part, fd)
            except FileNotFoundError:
                break
            os.close(fd)
            fd = child
        return os.fstat(fd).st_dev
    finally:
        os.close(fd)


@contextlib.contextmanager
def staging_for(
    target: Path | ReachablePath, *, same_filesystem: bool = False
) -> Iterator[Path]:
    """Private staging for a transfer to ``target``.

    A transport writes its staging by pathname, so it never stages in or
    near the destination, which a sandbox may mount: it stages in a fresh
    mode-700 folder in the controller's temporary folder (Docker, the one
    backend that binds arbitrary host folders, streams instead and never
    stages). Publication then copies through held, beneath-resolved
    descriptors. A publication that renames (``same_filesystem``,
    Windows) stages on the destination's own filesystem: the destination
    folder or a controller folder above it, never a harness-mounted one.
    """
    if not same_filesystem:
        with _private_staging(
            [Path(tempfile.gettempdir())], None, temp=False
        ) as folder:
            yield folder
        return
    capability = destination(target)
    anchor = capability._anchor
    anchor.ensure()
    device = _destination_device(capability)
    start = anchor.path if anchor.controller_owned else anchor.path.parent
    with _private_staging(_owners(start), device, temp=False) as folder:
        yield folder


def write_host_text(target: Path | ReachablePath, text: str) -> None:
    """Publish controller output into a possibly live mount without link writes."""
    require_descriptor_io()
    with _private_spool() as source:
        source.write(text.encode("utf-8"))
        source.seek(0)
        capability = destination(target)
        capability._anchor.ensure()
        with capability._folders(parents=True) as handles:
            _copy_opened(
                source.fileno(), handles[-1], capability.name, private_source=True
            )


def copy_mounted(
    root: Path,
    parts: tuple[str, ...],
    target: Path | ReachablePath,
    *,
    directory: bool,
    mode: int | None = None,
) -> None:
    """Anchor at a trusted mount root; admit only opened directories/files.

    Every source component is opened relative to its parent's descriptor with
    O_NOFOLLOW. A concurrent replacement can cause refusal, but cannot redirect
    a read. Symlinks and special entries are refused, including internal links:
    copying a link would leave later host consumers to resolve it unsafely.
    ``root`` (the harness's own source folder) and the destination root are
    canonicalised once; every component below either, and the final entries,
    follow no link and cross no mount.
    """
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Unsafe mounted path")
    if os.name == "nt":
        # The portable capability implementation holds no-reparse handles
        # without write/delete sharing. Keep a private copy until publication,
        # staged on the destination's volume so publishing is a rename there.
        source = source_capability(root)
        if parts:
            source = source / Path(*parts)
        if directory != source.is_dir():
            raise UnsafeDownloadEntryError("download source has the wrong type")
        capability = destination(target)
        with staging_for(capability, same_filesystem=True) as raw:
            staged = raw / "payload"
            snapshot_tree(source, staged, refuse_links=True)
            if mode is not None:
                staged.chmod(mode)
            capability.publish(staged)
        return
    require_descriptor_io()
    # The source root is the harness's own folder (staging, an upload's
    # source, the held rollout folder): canonicalised once, so an OS alias
    # above it (macOS /tmp) is followed; nothing below it is (R9-5, D867).
    fd = _open_dir(Path(os.path.realpath(Path(root).absolute())))
    try:
        where = Path(root)
        for part in parts[:-1] if parts else ():
            where = where / part
            child = _open_beneath(fd, part, os.O_RDONLY | _DIRECTORY, where=where)
            os.close(fd)
            fd = child
        if parts:
            child = _open_beneath(
                fd,
                parts[-1],
                os.O_RDONLY | getattr(os, "O_NONBLOCK", 0),
                where=where / parts[-1],
            )
            os.close(fd)
            fd = child
        if directory != stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError(
                "Mounted source is not the requested regular file/directory"
            )
        if directory:
            _validate_download_tree(fd)
        # The destination can also be mounted in the running sandbox. Hold
        # every folder from its trusted anchor before clearing or creating
        # anything; a later rename cannot redirect operations through a
        # replacement symlink, and a swapped folder refuses.
        capability = destination(target)
        capability._anchor.ensure()
        with capability._folders(parents=True) as handles:
            parent = handles[-1]
            if directory:
                _remove_beneath(
                    parent, capability.name, where=capability.details_message()
                )
            _copy_opened(fd, parent, capability.name, mode=mode)
    finally:
        os.close(fd)


def _validate_download_tree(root: int) -> None:
    """Preflight received filenames before publishing any output tree entries."""
    from benchflow._utils.archive_paths import ArchivePaths

    inventory = ArchivePaths()

    def walk(folder: int, parts: tuple[str, ...]) -> None:
        for name in os.listdir(folder):
            child = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=folder
            )
            try:
                mode = os.fstat(child).st_mode
                directory = stat.S_ISDIR(mode)
                inventory.add("/".join((*parts, name)), directory=directory)
                if directory:
                    walk(child, (*parts, name))
                elif not stat.S_ISREG(mode):
                    raise ReachableWriteError("download contains a special entry")
            finally:
                os.close(child)

    walk(root, ())


def _copy_opened(
    fd: int,
    parent: int,
    name: str,
    *,
    mode: int | None = None,
    private_source: bool = False,
) -> None:
    with _spool_io("stat") if private_source else contextlib.nullcontext():
        info = os.fstat(fd)
    permissions = mode if mode is not None else info.st_mode & 0o777
    if stat.S_ISDIR(info.st_mode):
        os.mkdir(name, mode=0o700, dir_fd=parent)
        destination = _open_beneath(parent, name, os.O_RDONLY | _DIRECTORY, where=name)
        try:
            for entry in os.listdir(fd):
                from benchflow._utils.archive_paths import archive_path

                archive_path(entry)
                child = os.open(
                    entry, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
                try:
                    _copy_opened(child, destination, entry)
                finally:
                    os.close(child)
            os.fchmod(destination, permissions)
            os.utime(destination, ns=(info.st_atime_ns, info.st_mtime_ns))
        finally:
            os.close(destination)
    elif stat.S_ISREG(info.st_mode):
        # Never truncate an existing inode: it may be a hard link to another
        # host file. Publish a new inode relative to the same opened parent.
        temporary = f".benchflow-copy-{uuid4().hex}"
        out = _open_beneath(
            parent,
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            where=name,
        )
        try:
            with (
                os.fdopen(out, "wb") as destination,
                _copy_source(fd, private=private_source) as source,
            ):
                shutil.copyfileobj(source, destination, 1024 * 1024)
                destination.flush()
                os.fchmod(destination.fileno(), permissions)
                os.utime(destination.fileno(), ns=(info.st_atime_ns, info.st_mtime_ns))
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=parent)
    else:
        raise ValueError(
            "Mounted source must be a regular file or directory, not a special entry"
        )


@contextlib.contextmanager
def _archive_source(
    archive, member, original, spool, regulars
) -> Iterator[_PrivateSpool[bytes]]:
    """Retain validated member bytes without retaining one descriptor per file."""
    from benchflow._utils.archive_paths import archive_path

    with _private_spool() as content:
        if member.islnk():
            span = regulars.get(archive_path(member.linkname).parts)
            if span is None:
                raise ReachableWriteError(
                    "archive hard link must name an earlier in-tree regular file"
                )
            start, remaining = span
            spool.seek(start)
            while remaining:
                chunk = spool.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise PrivateSpoolError(
                        errno.EIO, "private archive spool was truncated"
                    )
                content.write(chunk)
                remaining -= len(chunk)
        else:
            start = spool.seek(0, os.SEEK_END)
            source = archive.extractfile(member)
            if source is not None:
                with source:
                    while chunk := source.read(1024 * 1024):
                        content.write(chunk)
                        spool.write(chunk)
            spool.flush()
            span = (start, content.tell())
        regulars[original] = span
        content.flush()
        content.seek(0)
        yield content


def _windows_extract(
    stream: BinaryIO, work: ReachablePath, *, directory: bool, private: bool
) -> None:
    """Extract through Windows no-reparse directory handles, not POSIX dir_fd."""
    import tarfile

    from benchflow._utils.archive_paths import ArchivePaths, archive_path

    inventory = ArchivePaths()
    regulars: dict[tuple[str, ...], tuple[int, int]] = {}
    with (
        _private_spool() as spool,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            original = archive_path(member.name, allow_root=member.isdir()).parts
            parts = original if member.name.startswith("./") else original[1:]
            if parts:
                inventory.add("/".join(parts), directory=member.isdir())
            output = work / Path("payload", *parts)
            if member.isdir():
                with _snapshot_write() if private else contextlib.nullcontext():
                    output.mkdir(parents=True, exist_ok=True)
            elif member.isreg() or member.islnk():
                if not parts and directory:
                    raise UnsafeDownloadEntryError("download source has the wrong type")
                with (
                    _archive_source(
                        archive, member, original, spool, regulars
                    ) as content,
                    _snapshot_write() if private else contextlib.nullcontext(),
                ):
                    _publish_stream(content, output)
            else:
                raise ReachableWriteError("download contains a link or special entry")
    payload = work / "payload"
    with _snapshot_write() if private else contextlib.nullcontext():
        if directory and not payload.exists():
            payload.mkdir()
        if directory != payload.is_dir():
            raise UnsafeDownloadEntryError("download source has the wrong type")


# Inside a StagedTar's private staging: the destination folder a commit replaces.
_PREVIOUS = "previous"

_RENAME_NOREPLACE = 0x1  # Linux renameat2
_RENAME_EXCL = 0x4  # macOS renameatx_np
# renameat2 by number where libc predates its wrapper (glibc < 2.28).
_SYS_RENAMEAT2 = {"x86_64": 316, "aarch64": 276}


def _renameat_noreplace(
    src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int
) -> bool:
    """One atomic rename that fails EEXIST if ``dst`` exists.

    False where the platform, libc, kernel or filesystem has no such rename.
    """
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    names = (ctypes.c_char_p(os.fsencode(src)), ctypes.c_char_p(os.fsencode(dst)))
    folders = (ctypes.c_int(src_dir_fd), ctypes.c_int(dst_dir_fd))
    if sys.platform.startswith("linux"):
        call = getattr(libc, "renameat2", None)
        if call is not None:
            flags = ctypes.c_uint(_RENAME_NOREPLACE)
            result = call(folders[0], names[0], folders[1], names[1], flags)
        else:
            import platform

            number = _SYS_RENAMEAT2.get(platform.machine())
            if number is None:
                return False
            result = libc.syscall(
                ctypes.c_long(number),
                folders[0],
                names[0],
                folders[1],
                names[1],
                ctypes.c_uint(_RENAME_NOREPLACE),
            )
        unavailable = (errno.ENOSYS, errno.EINVAL, errno.EPERM)
    elif sys.platform == "darwin":
        call = getattr(libc, "renameatx_np", None)
        if call is None:
            return False
        flags = ctypes.c_uint(_RENAME_EXCL)
        result = call(folders[0], names[0], folders[1], names[1], flags)
        unavailable = (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP)
    else:
        return False
    if result == 0:
        return True
    err = ctypes.get_errno()
    if err in unavailable:
        # Old kernel, a seccomp filter, or a filesystem without the flag.
        return False
    raise OSError(err, os.strerror(err), dst)


def _rename_noreplace(src: str, dst: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
    """Rename ``src`` to ``dst`` only if nothing is at ``dst`` (TASKMD-R16).

    Never replaces an entry: EEXIST if the name is taken. Without a kernel
    no-replace rename, a non-folder is hard-linked (a link never replaces)
    and then unlinked; a folder raises ENOTSUP, since no portable call moves
    a folder without possibly replacing an empty one there.
    """
    if _renameat_noreplace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd):
        return
    kind = os.stat(src, dir_fd=src_dir_fd, follow_symlinks=False)
    if stat.S_ISDIR(kind.st_mode):
        raise OSError(errno.ENOTSUP, "no atomic no-replace rename for a folder", dst)
    os.link(
        src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd, follow_symlinks=False
    )
    os.unlink(src, dir_fd=src_dir_fd)


class StagedTar:
    """A ``docker cp SRC -`` tar stream, extracted beside ``target``, then committed.

    Nothing is staged by pathname. The copied entry's own name (the first
    component of every member) is dropped; the tree is built in a fresh
    mode-700 folder beside the target, every member opened beneath its held
    parent (no link followed, no mount crossed). Names are preflighted like
    any received tree; links and special entries are refused, as
    :func:`copy_mounted` refuses them.

    Extraction and commit are separate (TASKMD-R14 B-M1): a transport that
    fails after delivering whole members must not replace the destination,
    so the caller commits only once the archive validated AND its producer
    exited cleanly. Closing without commit discards the staging and leaves
    the old destination untouched. Use as a context manager; every step
    blocks, so run each in a worker thread.
    """

    def __init__(self, target: ReachablePath, *, directory: bool):
        self.target = target
        self.directory = directory
        self.committed = False
        self._stack = contextlib.ExitStack()
        self._work_fd: int | None = None
        self._windows_work: ReachablePath | None = None
        self._work_name = f".benchflow-download-{uuid4().hex}"
        self._kept_previous: OSError | None = None
        self._private = False

    def __enter__(self) -> StagedTar:
        try:
            self.target._anchor.ensure()
            handles = self._stack.enter_context(self.target._folders(parents=True))
            self._parent = handles[-1]
            self._private = (
                self.target._anchor.controller_owned
                and not self.target._anchor.reachable
                and not _inside_declared_mount(
                    self.target._anchor.path.joinpath(*self.target._parts[:-1])
                )
            )
            if self.target._anchor.fd is None:
                with (
                    _staging_io("mkdir") if self._private else contextlib.nullcontext()
                ):
                    work_path = Path(
                        tempfile.mkdtemp(
                            prefix=".benchflow-download-", dir=self._parent
                        )
                    )
                self._work_name = work_path.name
                with _staging_io("open") if self._private else contextlib.nullcontext():
                    self._windows_work = ReachablePath.directory(work_path)
            else:
                with (
                    _staging_io("mkdir") if self._private else contextlib.nullcontext()
                ):
                    os.mkdir(self._work_name, 0o700, dir_fd=self._parent)
                with _staging_io("open") if self._private else contextlib.nullcontext():
                    self._work_fd = _open_beneath(
                        self._parent,
                        self._work_name,
                        os.O_RDONLY | _DIRECTORY,
                        where=self._work_name,
                    )
        except BaseException as exc:
            self._discard(primary=exc)
            raise
        return self

    def extract(self, stream: BinaryIO) -> None:
        """Extract and validate every member; nothing outside staging changes."""
        from benchflow._utils.archive_paths import ArchivePaths

        if self._windows_work is not None:
            _windows_extract(
                stream,
                self._windows_work,
                directory=self.directory,
                private=self._private,
            )
            return
        import tarfile

        assert self._work_fd is not None
        _extract_members(
            stream,
            self._work_fd,
            self.directory,
            ArchivePaths(),
            tarfile,
            private=self._private,
        )
        with _snapshot_write() if self._private else contextlib.nullcontext():
            kind = os.stat("payload", dir_fd=self._work_fd, follow_symlinks=False)
        if self.directory != stat.S_ISDIR(kind.st_mode):
            raise UnsafeDownloadEntryError("download source has the wrong type")

    def commit(self) -> None:
        """Replace the destination with the validated payload."""
        with _snapshot_write() if self._private else contextlib.nullcontext():
            self._commit_payload()

    def _commit_payload(self) -> None:
        target = self.target
        if self._windows_work is not None:
            # Work's directory handle stays held until the child is renamed;
            # target.publish also holds every destination ancestor on Windows.
            target.publish(self._windows_work / "payload")
            self.committed = True
            return
        parent, work = self._parent, self._work_fd
        assert work is not None
        existing = None
        with contextlib.suppress(FileNotFoundError):
            existing = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
        if existing is None or not (self.directory or stat.S_ISDIR(existing.st_mode)):
            # Absent, or a file replaced by a file: one atomic rename.
            os.rename("payload", target.name, src_dir_fd=work, dst_dir_fd=parent)
            self.committed = True
            return
        # TASKMD-R15 B-R14-M2: a folder is never removed before its
        # replacement is in. It moves aside into this private staging, the
        # payload is renamed in, and only then is the old one removed; a
        # failed rename puts the old folder back, or, if the name has been
        # taken meanwhile, keeps it in staging and names it in the error.
        os.rename(target.name, _PREVIOUS, src_dir_fd=parent, dst_dir_fd=work)
        try:
            os.rename("payload", target.name, src_dir_fd=work, dst_dir_fd=parent)
        except BaseException as exc:
            try:
                # TASKMD-R16 B-R15-M1: another publication may have completed
                # while the name was vacant. Putting the old output back never
                # replaces a name this commit did not vacate.
                _rename_noreplace(
                    _PREVIOUS, target.name, src_dir_fd=work, dst_dir_fd=parent
                )
            except OSError as undo:
                # Staging now holds the only copy of the old output: keep it.
                self._kept_previous = undo
                exc.add_note(
                    f"previous {target.details_message()} kept in "
                    f"{self._work_name}/{_PREVIOUS}: {undo}"
                )
            raise
        self.committed = True
        with contextlib.suppress(OSError):
            # A failure here is retried, and reported, by the staging cleanup.
            _remove_beneath(work, _PREVIOUS, where=self._work_name)

    def __exit__(self, kind, value, traceback) -> None:
        self._discard(primary=value)

    def _discard(self, *, primary: BaseException | None) -> None:
        """Remove staging; a cleanup failure never replaces the primary error.

        A staging tree that cannot be removed is logged by path, and noted on
        the primary exception if there is one (TASKMD-R14 B-L1).
        """
        try:
            if self._work_fd is not None:
                os.close(self._work_fd)
                self._work_fd = None
            if self._windows_work is not None:
                work, self._windows_work = self._windows_work, None
                if work._anchor.windows is not None:
                    work._anchor.finalizer()
                shutil.rmtree(work._anchor.path, ignore_errors=True)
                if work._anchor.path.exists():
                    raise OSError(errno.ENOTEMPTY, "staging kept", work._anchor.path)
            elif self._kept_previous is not None:
                raise self._kept_previous
            elif hasattr(self, "_parent"):
                _remove_beneath(self._parent, self._work_name, where=self._work_name)
        except OSError as exc:
            kept = f"{self.target.details_message()}: {self._work_name}"
            logging.getLogger(__name__).warning(
                "Download staging kept beside %s (%s)", kept, exc
            )
            if primary is not None:
                primary.add_note(f"download staging kept beside {kept}: {exc}")
        finally:
            self._stack.close()


def publish_tar_stream(
    stream: BinaryIO, target: ReachablePath, *, directory: bool
) -> None:
    """Extract, validate and commit a whole tar stream (see :class:`StagedTar`).

    Only for a stream already known complete: a live transport commits with
    :class:`StagedTar` after its producer's exit status is known.
    """
    with StagedTar(target, directory=directory) as staged:
        staged.extract(stream)
        staged.commit()


@contextlib.contextmanager
def _archive_sink(
    parent: int, name: str, where: str, *, private: bool
) -> Iterator[BinaryIO]:
    # The source archive is consumed elsewhere. Only this validated private
    # destination's creation and close belong to the storage boundary.
    with _snapshot_write() if private else contextlib.nullcontext():
        fd = _open_beneath(
            parent, name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600, where=where
        )
        try:
            stream = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
    try:
        yield stream
    finally:
        with _snapshot_write() if private else contextlib.nullcontext():
            stream.close()


def _extract_members(
    stream, work: int, directory: bool, inventory, tarfile, *, private: bool
) -> None:
    from benchflow._utils.archive_paths import archive_path

    folders: list[tuple[tuple[str, ...], int, float]] = []
    # Retain validated bytes in an anonymous spool, never reopen published names.
    regulars: dict[tuple[str, ...], tuple[int, int]] = {}
    seen_root = False
    with (
        _private_spool() as spool,
        tarfile.open(fileobj=stream, mode="r|") as archive,
    ):
        for member in archive:
            original = archive_path(member.name, allow_root=member.isdir()).parts
            parts = original if member.name.startswith("./") else original[1:]
            relative = ("payload", *parts)
            if parts:
                inventory.add("/".join(parts), directory=member.isdir())
            if member.isdir():
                with _snapshot_write() if private else contextlib.nullcontext():
                    os.close(_make_folders(work, relative))
                folders.append((relative, member.mode & 0o777, member.mtime))
                seen_root = seen_root or not parts
            elif member.isreg() or member.islnk():
                if not parts and directory:
                    raise UnsafeDownloadEntryError("download source has the wrong type")
                with _snapshot_write() if private else contextlib.nullcontext():
                    parent = _make_folders(work, relative[:-1])
                try:
                    with (
                        _archive_sink(
                            parent,
                            relative[-1],
                            "/".join(parts) or "payload",
                            private=private,
                        ) as sink,
                        _archive_source(
                            archive, member, original, spool, regulars
                        ) as source,
                        _snapshot_write() if private else contextlib.nullcontext(),
                    ):
                        shutil.copyfileobj(source, sink, 1024 * 1024)
                        sink.flush()
                        os.fchmod(sink.fileno(), member.mode & 0o777)
                        os.utime(sink.fileno(), (member.mtime, member.mtime))
                finally:
                    os.close(parent)
            else:
                raise ReachableWriteError(
                    f"download contains a link or special entry at {member.name}"
                )
    if directory and not seen_root:
        with _snapshot_write() if private else contextlib.nullcontext():
            os.close(_make_folders(work, ("payload",)))
    for relative, mode, mtime in reversed(folders):
        with _snapshot_write() if private else contextlib.nullcontext():
            fd = _make_folders(work, relative)
        try:
            with _snapshot_write() if private else contextlib.nullcontext():
                os.fchmod(fd, mode)
                os.utime(fd, (mtime, mtime))
        finally:
            os.close(fd)


def _make_folders(work: int, relative: tuple[str, ...]) -> int:
    """Open (making as needed) ``relative`` below ``work``; return its descriptor."""
    fd = os.dup(work)
    try:
        for part in relative:
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, 0o700, dir_fd=fd)
            child = _open_beneath(fd, part, os.O_RDONLY | _DIRECTORY, where=part)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


# Import precedes agent execution: defaults are not inferred from transfers.
_INITIAL_ROOTS = tuple(
    pin_destination(root)
    for root in (Path(Path.cwd().anchor), Path.cwd(), Path(tempfile.gettempdir()))
)
