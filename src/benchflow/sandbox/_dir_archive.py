"""A directory moved as one archive (DAYTONA-GAPS rule 1).

A backend whose provider moves files one by one (Daytona's ``upload_files``,
Modal's ``open``) moves a directory as one plain tar instead: built here on
the host, one source file open at a time, uploaded as one file to a fresh
folder in root's staging folder, and unpacked there by root's system-shell
route in one exec (:func:`extract_command`), after the proof a staged
command gets (:func:`benchflow.sandbox._transport.stage_proof`). Open files
and provider calls are then bounded whatever the directory's file count:
Daytona's SDK opened every source file at once and failed at the host's
``ulimit -n`` (2,000 verifier files, ``EMFILE``).

The archive holds the regular files :func:`~benchflow._paths.iter_safe_tree`
yields and nothing else: no directory entry (a folder that already exists
keeps its owner and mode; one the unpacking makes is made as before, by root,
mode 755),
no link, no device. Each member is owned by uid and gid 0 with the mode the
backend's rule gives (never a set-id bit), so root's ``tar`` writes the same
contents, modes and owners the per-file upload and its ``chown``/``chmod``
wrote.
"""

from __future__ import annotations

import os
import shlex
import tarfile
import tempfile
from collections.abc import Callable, Collection
from pathlib import Path

from benchflow._paths import iter_safe_tree


def daytona_mode(st_mode: int) -> int:
    """Daytona's rule: executable for anyone when any execute bit is set."""
    return 0o755 if st_mode & 0o111 else 0o644


def source_mode(st_mode: int) -> int:
    """The source's permission bits (Modal's rule), never a set-id or
    sticky bit."""
    return st_mode & 0o777


def write_archive(
    source_dir: Path | str,
    *,
    exclude: Collection[str] = (),
    mode: Callable[[int], int] = daytona_mode,
    context: str = "upload_dir",
) -> tuple[Path, int]:
    """A temporary tar of the regular files under *source_dir* and their
    count. *exclude*: top-level entries left out. The caller removes the
    file."""
    source_dir = Path(source_dir)
    fd, name = tempfile.mkstemp(prefix="benchflow-dir-", suffix=".tar")
    os.close(fd)
    archive = Path(name)
    count = 0
    try:
        with tarfile.open(
            archive,
            "w",
            format=tarfile.GNU_FORMAT,
            encoding="utf-8",
            errors="surrogateescape",
        ) as tar:
            for file_path in iter_safe_tree(source_dir, context=context):
                relative = file_path.relative_to(source_dir)
                if relative.parts[0] in exclude:
                    continue
                with open(file_path, "rb") as stream:
                    st = os.fstat(stream.fileno())
                    info = tarfile.TarInfo(relative.as_posix())
                    info.size = st.st_size
                    info.mtime = int(st.st_mtime)
                    info.mode = mode(st.st_mode) & 0o777
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tar.addfile(info, stream)
                count += 1
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    return archive, count


#: Exit status of :func:`extract_command` when the sandbox has no ``tar``.
NO_TAR = 127
#: Exit status of :func:`extract_command` when the archive's route is not
#: root's alone (nothing was unpacked).
NOT_ROOTS = 126


def extract_command(archive: str, target_dir: str, proof: str) -> str:
    """Root's system-shell text: once *proof* (the staged archive's route
    is root's alone) passed, unpack *archive* into *target_dir* (made when
    missing) and remove the archive's folder. Exits :data:`NOT_ROOTS`
    (nothing unpacked, nothing removed) when the proof fails, and
    :data:`NO_TAR` when no ``tar`` is on the route's PATH."""
    folder = shlex.quote(archive.rsplit("/", 1)[0])
    target = shlex.quote(target_dir)
    return (
        f"{proof} || exit {NOT_ROOTS}; umask 022; "
        f"if command -v tar >/dev/null 2>&1; then "
        f"mkdir -p -- {target} && tar -x -f {shlex.quote(archive)} -C {target}; "
        f"bf_s=$?; else bf_s={NO_TAR}; fi; rm -rf -- {folder}; exit $bf_s"
    )
