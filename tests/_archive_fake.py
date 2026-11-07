"""A fake sandbox's side of the one-archive directory upload (DAYTONA-GAPS
rule 1).

A fake whose files live under a host folder sees the archive uploaded as one
file and then root's unpack command (:func:`benchflow.sandbox._dir_archive.
extract_command`). :func:`unpack_archive` does what that command does on the
far side once its route proof passed: unpack into the target folder and remove
the archive's staging folder.
"""

from __future__ import annotations

import re
import shlex
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path

_WORD = r"'[^']*'|\S+"
_UNPACK = re.compile(
    rf"mkdir -p -- (?P<target>{_WORD}) && tar -x -f (?P<archive>{_WORD}) -C "
)


def unpack_archive(command: str, path: Callable[[str], Path]) -> bool:
    """When *command* is root's unpack command, unpack its archive into its
    target under the fake's file system (*path* maps a sandbox path to the
    host), remove the archive's folder and return True; else False."""
    match = _UNPACK.search(command)
    if match is None:
        return False
    target = path(shlex.split(match["target"])[0])
    archive = path(shlex.split(match["archive"])[0])
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="tar")
    shutil.rmtree(archive.parent)
    return True
