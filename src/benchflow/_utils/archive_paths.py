"""Validate archive names before any extraction writes, on every backend."""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import PurePosixPath

from benchflow.rewards._safe_evidence import _windows_component


def archive_path(value: str, *, allow_root: bool = False) -> PurePosixPath:
    """Accept only relative names that stay entries of this controller's
    folder, before normalizing tar's ./.

    Refused on every controller: an absolute path, a backslash, ``..`` and a
    NUL or other control character. Refused only on a Windows controller,
    where Win32 makes them something else: a device name in any case and with
    any extension (``aux.py``, ``con.txt``), a ``name:stream`` or drive
    (``C:x``), ``*?"<>|``, a trailing dot or space, and an 8.3 alias
    (``LONGNA~1``). On Linux and macOS those are ordinary names a workspace
    can hold (``A:1022428818870.json``, a Next.js chunk ``0jr8wo_7ak~7n.js``).
    Refusing them there protected no reader: BenchFlow needs a POSIX
    controller, and a Windows host that reads a copied bundle applies this
    same rule itself (``os.name``, as ``ReachablePath`` does). It only failed
    the capture, and so the trial's review and score, on every retry.
    """
    if not value or value.startswith("/") or "\\" in value:
        raise ValueError(f"Unsafe archive path: {value!r}")
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise ValueError(f"Unsafe archive path: {value!r}")
        if any(ord(c) < 32 for c in part):
            raise ValueError(f"Unsafe archive filename alias: {value!r}")
        if os.name == "nt":
            _windows_component(part)
            if re.search(r"~[0-9]", part):
                raise ValueError(f"Unsafe archive filename alias: {value!r}")
    path = PurePosixPath(value)
    if not path.parts and not allow_root:
        raise ValueError(f"Empty archive path: {value!r}")
    return path


def filename_identity(component: str) -> str:
    """Use the target's filename identity, including implicit parent folders."""
    if os.name == "nt":
        return component.casefold()
    if sys.platform == "darwin":
        # Accept safely on both case-sensitive and default case-insensitive APFS.
        return unicodedata.normalize("NFD", component).casefold()
    return component


class ArchivePaths:
    """A preflight inventory that cannot represent aliases or non-folder parents."""

    def __init__(self) -> None:
        self.spellings: dict[tuple[str, ...], tuple[str, ...]] = {}
        self.entries: dict[tuple[str, ...], bool] = {}
        self.parents: set[tuple[str, ...]] = set()

    def add(self, value: str, *, directory: bool) -> PurePosixPath:
        path = archive_path(value, allow_root=directory)
        identity = tuple(filename_identity(part) for part in path.parts)
        for length in range(1, len(identity) + 1):
            prefix, spelling = identity[:length], path.parts[:length]
            if self.spellings.setdefault(prefix, spelling) != spelling:
                raise ValueError(f"Archive filename collision: {value!r}")
            if length < len(identity):
                if self.entries.get(prefix) is False:
                    raise ValueError(f"Archive parent is not a directory: {value!r}")
                self.parents.add(prefix)
        if identity in self.entries or (not directory and identity in self.parents):
            raise ValueError(
                f"Duplicate or conflicting archive path: {value!r} names an entry more than once"
            )
        self.entries[identity] = directory
        return path
