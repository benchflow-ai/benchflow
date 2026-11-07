#!/usr/bin/env python3
"""Move CI job evidence through regular-file-only archives.

Agent-writable job mounts may contain links into the runner. Upload actions
follow those links, so their input must be a private archive, never a job tree.
Walk by directory descriptors and open every component without following links;
checking a pathname and then reopening it would leave a replacement race.
The receiving job likewise materializes only directories and regular files.
"""

from __future__ import annotations

import argparse
import errno
import os
import stat
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_SKIP = {errno.ENOENT, errno.ELOOP, errno.ENOTDIR, errno.ENXIO}


@contextmanager
def _directory(path: Path, *, create: bool = False) -> Iterator[int]:
    """Anchor every parent, including the supplied root, without links."""
    absolute = path.absolute()
    fd = os.open(absolute.anchor, _DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            if part in {".", ".."}:
                raise ValueError("artifact paths must not contain dot components")
            if create:
                with suppress(FileExistsError):
                    os.mkdir(part, dir_fd=fd)
            child = os.open(part, _DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _pack_directory(archive: tarfile.TarFile, directory: int, prefix: str = "") -> None:
    for name in sorted(os.listdir(directory)):
        # Match upload-artifact's default hidden-file exclusion. Archiving
        # must not newly publish private dotfiles that the action omitted.
        if name.startswith("."):
            continue
        try:
            fd = os.open(name, _FILE, dir_fd=directory)
        except OSError as exc:
            if exc.errno in _SKIP:
                continue
            raise
        try:
            info = os.fstat(fd)
            member = tarfile.TarInfo(prefix + name)
            if stat.S_ISDIR(info.st_mode):
                member.type = tarfile.DIRTYPE
                member.mode = 0o755
                archive.addfile(member)
                _pack_directory(archive, fd, member.name + "/")
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                member.size = info.st_size
                member.mode = 0o644
                with os.fdopen(fd, "rb", closefd=False) as source:
                    archive.addfile(member, source)
            # Links, sockets, devices and FIFOs have no archive representation.
        finally:
            os.close(fd)


def pack(source: Path, output: Path) -> None:
    if not source.exists() and not source.is_symlink():
        return  # An always() step may follow a run that never created jobs.
    with (
        _directory(source) as directory,
        _directory(output.parent, create=True) as parent,
    ):
        # The output and every parent obey the same no-follow rule.
        fd = os.open(
            output.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        with (
            os.fdopen(fd, "wb") as target,
            tarfile.open(fileobj=target, mode="w:gz", dereference=False) as archive,
        ):
            _pack_directory(archive, directory)


def unpack(archives: Path, destination: Path) -> None:
    with _directory(destination, create=True):
        pass
    with _directory(archives) as directory:
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".tar.gz"):
                continue
            fd = os.open(name, _FILE, dir_fd=directory)
            with os.fdopen(fd, "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("job archive must be a regular file")
                with tarfile.open(fileobj=source, mode="r:gz") as archive:
                    for member in archive:
                        # Never restore links, even links that currently resolve
                        # inside the destination: downstream readers follow them.
                        if not (member.isfile() or member.isdir()):
                            raise ValueError(
                                "job archives allow only files and directories"
                            )
                        parts = member.name.rstrip("/").split("/")
                        if any(part in {"", ".", ".."} for part in parts):
                            raise ValueError("invalid job archive member path")
                        path = destination.joinpath(*parts)
                        if member.isdir():
                            with _directory(path, create=True):
                                pass
                            continue
                        with _directory(path.parent, create=True) as parent:
                            output = os.open(
                                parts[-1],
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                0o644,
                                dir_fd=parent,
                            )
                            with os.fdopen(output, "wb") as target:
                                body = archive.extractfile(member)
                                assert body is not None
                                with body:
                                    while chunk := body.read(1024 * 1024):
                                        target.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack", "unpack"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.operation == "pack":
        pack(args.source, args.destination)
    else:
        unpack(args.source, args.destination)


if __name__ == "__main__":
    main()
