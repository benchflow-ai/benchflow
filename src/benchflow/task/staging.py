"""A private, no-follow snapshot of a task's files.

A sandbox is built and launched from files BenchFlow copied before checking
them, never from the task's own folder: a link in the task is copied as a
link (never read through), a link that leaves the copied folder and a FIFO,
socket or device are refused, and nothing outside BenchFlow can write the
copy (a ``0700`` folder of the operator's). So the files the launch gate
reads are the files Compose, Docker or a provider's upload read afterwards,
whatever another task's container does to the task's folder meanwhile, and
no link can turn an operator file into a task file (-R3-8).

The whole task is copied at the original's absolute path below its private
root (``<root>/tmp/tasks/t/environment``). A relative path into ``../tests``
or a task payload remains in that instance's private copy. Paths beyond
the task are mapped back only for the checks and explicit host-access
policy.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import stat
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from benchflow.task.env import TaskEnvRequest

STAGE_PREFIX = "benchflow-stage-"

_LINK_REMEDY = (
    "BenchFlow copies a task's folder before it checks and runs it, and "
    "never reads a file through a link that leaves the folder, or a FIFO, "
    "socket or device: replace it with the file itself"
)

_live: set[Path] = set()
_live_lock = threading.Lock()
# Roots a garbage-collected sandbox's finalizer left for removal. The
# finalizer never waits for ``_live_lock``: the collector runs it on
# whatever thread allocates, which can be the one already holding that lock
# inside ``_live.add``, and a non-reentrant lock then waits for itself
# forever (PY8-B23-STAGING-GC). It takes the lock only when it is free and
# otherwise appends here (atomic); every holder of the lock drains this
# first (:func:`_drain_locked`). A root whose removal an error or an
# interrupt cut short comes back here too (:func:`_remove_roots`).
_released: deque[Path] = deque()


def _drain_locked() -> list[Path]:
    """Take the finalizers' queued roots out of the live set (the caller
    holds ``_live_lock``); the caller removes them once it lets go."""
    roots = []
    while True:
        try:
            root = _released.popleft()
        except IndexError:
            return roots
        _live.discard(root)
        roots.append(root)


def _remove_roots(roots: list[Path]) -> None:
    """Delete roots already out of the live set, each on its own.

    A root an error or an interrupt leaves on disk goes back on the queue,
    so the next holder of the lock (or exit) tries again, and one sandbox's
    failed clean-up never fails another trial's staging. An interrupt still
    propagates once the rest are queued. No lock and no logging: a
    finalizer runs this.
    """
    retry: list[Path] = []
    done = 0
    try:
        for root in roots:
            try:
                _rmtree(root)
            except Exception:
                retry.append(root)
            done += 1
    finally:
        # Only a root still on disk: a queued name that is gone could be
        # mkdtemp's next folder, which the drain would then drop.
        for root in dict.fromkeys((*retry, *roots[done:])):
            if os.path.lexists(root):
                _released.append(root)


def remove_released() -> None:
    """Delete the copies of every sandbox collected so far (a job's end).

    A finalizer that found the lock taken left its root queued; without
    this it waits for the next staging call or the process's exit, which an
    idle SDK process may not reach for a long time.
    """
    with _live_lock:
        roots = _drain_locked()
    _remove_roots(roots)


def _remove_all() -> None:
    with _live_lock:
        roots = [*_drain_locked(), *_live]
        _live.clear()
    _remove_roots(roots)


atexit.register(_remove_all)


def _rmtree(root: Path) -> None:
    def writable(function: Callable, path: str, _exc: object) -> None:
        # A folder the run made read-only (or a container's root-owned
        # file, which stays): try once more with the owner's write bit.
        with contextlib.suppress(OSError):
            os.chmod(os.path.dirname(path), 0o700)
            function(path)

    if root.exists():
        shutil.rmtree(root, onexc=writable)


class StagingRefused(Exception):
    """The folder holds something BenchFlow will not copy."""

    def __init__(
        self, refused: list[TaskEnvRequest], *, transient: bool = False
    ) -> None:
        self.refused = refused
        #: The folder changed while it was copied: another try may pass.
        self.transient = transient
        super().__init__("; ".join(r.describe() for r in refused))


def _refusal(label: str, what: str) -> TaskEnvRequest:
    return TaskEnvRequest(label, "", None, f"{label} {what}", remedy=_LINK_REMEDY)


def _within(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def mirror(root: Path, path: Path) -> Path:
    """Where *path* (absolute, resolved) sits below a staging *root*."""
    return root.joinpath(*path.parts[1:])


def copy_tree(
    source: Path,
    destination: Path,
    *,
    label: str = "",
    allow_outside: bool = False,
) -> None:
    """Copy *source* to *destination* without following any link.

    A regular file is opened with ``O_NOFOLLOW`` and checked to still be a
    regular file once open; a folder is created, never entered through a
    link; a link inside *source* (its target, links followed, inside the
    folder) is made again as a relative link. Refused (raises
    :class:`StagingRefused` naming every one): a link whose target leaves
    *source* (unless *allow_outside*: then it is made again pointing at
    that target), and a FIFO, socket or device. Modes and times are kept.
    """
    real = os.path.realpath(source)
    refused: list[TaskEnvRequest] = []
    os.makedirs(destination, mode=0o700, exist_ok=True)

    def name(relative: str) -> str:
        return "/".join(p for p in (label, relative.replace(os.sep, "/")) if p)

    def walk(directory: int, relative: str) -> None:
        # Keep the directory open through enumeration and every child open.
        # O_NOFOLLOW on a leaf alone still follows a swapped ancestor
        # .
        with os.scandir(directory) as entries:
            items = sorted(entries, key=lambda e: e.name)
        for entry in items:
            rel = os.path.join(relative, entry.name) if relative else entry.name
            target = os.path.join(destination, rel)
            info = os.stat(entry.name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = _open_directory(entry.name, directory)
                os.mkdir(target, 0o700)
                try:
                    walk(child, rel)
                finally:
                    os.close(child)
                os.chmod(target, stat.S_IMODE(info.st_mode) | 0o700)
                os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
            elif stat.S_ISREG(info.st_mode):
                _copy_file(entry.name, target, info, directory)
            elif stat.S_ISLNK(info.st_mode):
                link = os.readlink(entry.name, dir_fd=directory)
                parent = os.path.dirname(os.path.join(real, rel))
                lands = os.path.realpath(os.path.join(parent, link))
                if _within(lands, real):
                    os.symlink(os.path.relpath(lands, parent), target)
                elif allow_outside:
                    os.symlink(lands, target)
                else:
                    refused.append(
                        _refusal(
                            name(rel),
                            f"is a link to {link}, outside the task's folder",
                        )
                    )
            else:
                kind = (
                    "a FIFO"
                    if stat.S_ISFIFO(info.st_mode)
                    else "a socket"
                    if stat.S_ISSOCK(info.st_mode)
                    else "a device"
                )
                refused.append(_refusal(name(rel), f"is {kind}, not a file"))

    # Pin every component of the root as well, starting at /. A single
    # absolute open would leave the root's ancestors open to substitution.
    directory = _open_directory(os.sep)
    try:
        for component in Path(real).parts[1:]:
            child = _open_directory(component, directory)
            os.close(directory)
            directory = child
        walk(directory, "")
    except OSError as exc:
        raise StagingRefused(
            [_refusal(label or str(source), "changed while being copied; try again")],
            transient=True,
        ) from exc
    finally:
        os.close(directory)
    if refused:
        raise StagingRefused(refused)


def _open_directory(path: str, parent: int | None = None) -> int:
    return os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=parent,
    )


def _copy_file(source: str, target: str, info: os.stat_result, directory: int) -> None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    fd = os.open(source, flags, dir_fd=directory)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):  # swapped since the listing
            raise StagingRefused(
                [_refusal(source, "changed into something that is not a file")],
                transient=True,
            )
        out = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            with (
                os.fdopen(fd, "rb", closefd=False) as src,
                os.fdopen(out, "wb", closefd=False) as dst,
            ):
                shutil.copyfileobj(src, dst, 1024 * 1024)
            os.fchmod(out, stat.S_IMODE(opened.st_mode) | 0o600)
        finally:
            os.close(out)
    finally:
        os.close(fd)
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns), follow_symlinks=False)


@dataclass
class StagedEnvironment:
    """A task's environment folder as BenchFlow copied it (see module doc)."""

    #: The private root (``0700``) holding the copy, and the launch files.
    root: Path
    #: The task's environment folder, resolved, as it was copied.
    original_environment_dir: Path
    #: The copy: what every sandbox call uses as the environment folder.
    environment_dir: Path
    _removed: bool = field(default=False, repr=False)

    @property
    def original_task_dir(self) -> Path:
        return self.original_environment_dir.parent

    @property
    def task_dir(self) -> Path:
        """The private copy of the whole task, including ../tests mounts."""
        return self.environment_dir.parent

    def original(self, path: str) -> str:
        """Keep the task snapshot's paths; map a path climbing beyond the
        copied task back to its original for the host-access check."""
        path = os.path.normpath(path)
        root = str(self.root)
        if (
            _within(path, str(self.task_dir))
            or not _within(path, root)
            or path[len(root) :].startswith(f"{os.sep}.benchflow-")
        ):
            return path
        rest = path[len(root) :]
        return rest if rest.startswith(os.sep) else os.sep + rest

    def snapshot_path(self, path: str) -> str:
        """Keep task-relative consumers in this instance's private copy.

        Absolute references to the original task need the same treatment
        as ../ references. Only explicitly allowed paths
        outside the task may still name originals.
        """
        path = self.original(path)
        if _within(path, str(self.original_task_dir)):
            return str(mirror(self.root, Path(path)))
        return path

    def private_folder(self, name: str) -> Path:
        """A ``0700`` folder of BenchFlow's own below the root, outside the
        copy (the frozen Compose project, the docker client's config)."""
        path = self.root / f".benchflow-{name}"
        path.mkdir(mode=0o700, exist_ok=True)
        return path

    def stage_file(self, original: Path) -> Path | None:
        """Copy one more file of the original task (a Compose dependency
        outside the environment folder, when the operator allowed host
        access) to its place below the root; ``None`` if it is not a
        regular file there."""
        try:
            info = os.stat(original, follow_symlinks=True)
        except OSError:
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        target = mirror(self.root, Path(os.path.realpath(original)))
        if target.exists():
            return target
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copy2(original, target)
        return target

    def remove(self) -> None:
        if self._removed:
            return
        self._removed = True
        with _live_lock:
            released = _drain_locked()
            _live.discard(self.root)
        _remove_roots([*released, self.root])

    def release(self) -> None:
        """:meth:`remove` for a finalizer: never waits for the lock.

        When the lock is free the copy goes at once. When it is held (by
        this thread, inside staging, or by another) the root is queued and
        stays on disk, so no new staging folder can take its name, until
        the next staging call, removal, job end or process exit.
        """
        if self._removed:
            return
        self._removed = True
        if not _live_lock.acquire(blocking=False):
            _released.append(self.root)
            return
        try:
            released = _drain_locked()
            _live.discard(self.root)
        finally:
            _live_lock.release()
        _remove_roots([*released, self.root])


def stage_environment(
    environment_dir: Path, *, label: str = "environment", allow_outside: bool = False
) -> StagedEnvironment:
    """Copy the task containing *environment_dir* to a new private folder.

    Raises :class:`StagingRefused` for a link that leaves the folder or a
    special file (*allow_outside*: links out are kept as links, for an
    operator who allowed host access). The copy is removed by
    :meth:`StagedEnvironment.remove`, or when the process exits.
    """
    original = Path(os.path.realpath(environment_dir))
    root = Path(os.path.realpath(tempfile.mkdtemp(prefix=STAGE_PREFIX)))
    with _live_lock:
        released = _drain_locked()
        _live.add(root)
    staged = StagedEnvironment(root, original, mirror(root, original))
    try:
        _remove_roots(released)
        staged.environment_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if original.is_dir():
            copy_tree(
                original.parent,
                staged.task_dir,
                label="",
                allow_outside=allow_outside,
            )
        else:
            staged.environment_dir.mkdir(mode=0o700)
    except BaseException:
        staged.remove()
        raise
    return staged
