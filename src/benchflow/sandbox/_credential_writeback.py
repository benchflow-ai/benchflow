"""Root-side symlink-safe write-back of scrubbed credential files.

Runs inside the sandbox as root, driven by
:mod:`benchflow.sandbox._snapshot_credentials`. Each credential file is written
back into a path the agent controls (``~/.codex/auth.json`` etc.). Because the
snapshot may be taken while the agent's own processes are alive (opt-in
checkpoints), the agent can race the write: swap a credential file — or a
directory above it — for a symbolic link between the scrub's ``rm`` and this
write, so that root writes through the link and then hands the target to the
agent (e.g. ``sitecustomize.py`` in shared site-packages).

To close that race, every destination is created **without following links and
exclusively**: the parent chain is opened component by component with
``O_NOFOLLOW`` from ``/`` down, and the leaf is created with
``O_CREAT | O_EXCL | O_NOFOLLOW``. Owner and mode are then set on the open file
descriptor (``fchown`` / ``fchmod``), never on the path. A link anywhere on the
chain, or a destination that already exists, is refused and named — never
followed. Stdlib only, so the source can be shipped into the sandbox and run
with ``python3 -I -c``.
"""

from __future__ import annotations

import json
import os
import stat
import sys

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW


class _Refused(Exception):
    """A credential path could not be written without following a link."""


def _open_parent(parent_parts: list[str]) -> int:
    """Open the destination's parent directory, following no symlinks.

    Walks from ``/`` opening each component with ``O_NOFOLLOW``. A missing
    component is created (``mkdir`` at the parent fd, so still no link is
    followed); an existing component that is a symlink or not a directory
    raises :class:`_Refused` rather than being traversed.
    """
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in parent_parts:
            try:
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                os.mkdir(part, 0o755, dir_fd=fd)
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError as exc:
                # ELOOP (symlink under O_NOFOLLOW) or ENOTDIR land here.
                raise _Refused(
                    f"parent component {part!r} is a link or not a directory "
                    f"(errno {exc.errno})"
                ) from exc
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _install(entry: dict) -> None:
    dest = entry["path"]
    parts = [p for p in dest.split("/") if p]
    parent_parts, name = parts[:-1], parts[-1]

    src_fd = os.open(entry["staged"], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(src_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
    finally:
        os.close(src_fd)

    parent_fd = _open_parent(parent_parts)
    try:
        fd = _open_leaf(parent_fd, name, dest)
        try:
            os.ftruncate(fd, 0)
            os.write(fd, data)
            os.fchmod(fd, int(str(entry["mode"]), 8))
            os.fchown(fd, int(entry["uid"]), int(entry["gid"]))
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _open_leaf(parent_fd: int, name: str, dest: str) -> int:
    """Open the destination for writing without ever following a symlink.

    New files are created exclusively (``O_EXCL``) — the normal case, since the
    scrub removed them first. An existing name is accepted only when it opens
    under ``O_NOFOLLOW`` (so it is not a symlink) and turns out to be a plain,
    un-hardlinked regular file (the scrub-rollback / re-put-back case); a
    symlink swap, a special file, or a hard link is refused and named. The
    checks run on the open descriptor before it is truncated, so the inode
    cannot change under us.
    """
    try:
        return os.open(name, _NEW_FILE_FLAGS, 0o600, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise _Refused(
            f"{dest!r} could not be created without following a link "
            f"(errno {exc.errno})"
        ) from exc
    try:
        fd = os.open(name, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        # ELOOP: the name became a symlink after the scrub (the swap attack).
        raise _Refused(
            f"{dest!r} is a symlink after the scrub; refusing to write through "
            f"it (errno {exc.errno})"
        ) from exc
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise _Refused(
            f"{dest!r} is not a plain, un-hardlinked regular file "
            f"(mode {info.st_mode:#o}, nlink {info.st_nlink}); refusing"
        )
    return fd


def main(argv: list[str]) -> int:
    entries = json.loads(argv[1])
    refused: list[str] = []
    for entry in entries:
        try:
            _install(entry)
        except _Refused as exc:
            refused.append(str(exc))
    if refused:
        sys.stderr.write("REFUSED: " + "; ".join(refused) + "\n")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
