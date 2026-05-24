"""Path-safety helpers shared across ingestion and upload code paths.

``Path.is_file()`` and ``Path.rglob("*")`` both follow symlinks by default.
When walking a directory we do not own — agent artifacts, task deliverables,
dependency stages, skill trees — that means an attacker-placed symlink can
pull arbitrary host-readable files into a dashboard payload, a judge prompt,
or a remote sandbox upload.

The fix shape is the same everywhere:

* check ``Path.is_symlink()`` (which does *not* follow links) before reading,
* fall back to ``os.lstat`` and ``stat.S_ISREG`` to confirm we're looking at
  a regular file rather than a fifo/socket/device, and
* emit one warning per skipped entry so missing artifacts are explainable.

A parallel PR (path-traversal hardening) adds ``safe_path_segment`` and
``assert_within`` to this same module. The two function sets are independent;
when both PRs land the file is a flat union.
"""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path

__all__ = [
    "is_safe_regular_file",
    "is_safe_regular_dir",
    "iter_safe_children",
    "iter_safe_tree",
    "ignore_symlinks",
]

logger = logging.getLogger(__name__)


def is_safe_regular_file(path: Path) -> bool:
    """True if *path* exists, is a regular file, and is not a symlink.

    Uses ``os.lstat`` so symlinks, fifos, sockets, and device files all
    return False. A non-existent path also returns False.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode)


def is_safe_regular_dir(path: Path) -> bool:
    """True if *path* is a directory and not a symlink to one."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode)


def iter_safe_children(
    directory: Path,
    *,
    context: str = "directory walk",
) -> Iterator[Path]:
    """Yield direct children of *directory*, skipping symlinks with a warning.

    Symlinks are skipped regardless of where they point — confirming
    containment on every entry is cheaper than re-checking ``realpath``
    upstream, and a same-tree symlink usually still indicates abuse.
    """
    try:
        entries = sorted(directory.iterdir())
    except (OSError, NotADirectoryError):
        return
    for child in entries:
        if child.is_symlink():
            logger.warning(
                "%s: skipping symlink %s (refusing to follow)", context, child
            )
            continue
        yield child


def iter_safe_tree(
    root: Path,
    *,
    context: str = "tree walk",
) -> Iterator[Path]:
    """Recursively yield regular files under *root*, never following symlinks.

    Uses ``os.walk(followlinks=False)`` so directory symlinks are also not
    descended into. Both symlinked files and symlinked directories produce a
    single warning each.
    """
    if not is_safe_regular_dir(root):
        if Path(root).is_symlink():
            logger.warning(
                "%s: refusing to descend into symlinked root %s", context, root
            )
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        kept_dirs: list[str] = []
        for name in dirnames:
            child = base / name
            if child.is_symlink():
                logger.warning(
                    "%s: skipping symlinked directory %s (refusing to follow)",
                    context,
                    child,
                )
                continue
            kept_dirs.append(name)
        # In-place edit controls os.walk traversal.
        dirnames[:] = sorted(kept_dirs)
        for name in sorted(filenames):
            f = base / name
            if not is_safe_regular_file(f):
                logger.warning(
                    "%s: skipping non-regular path %s (symlink or special file)",
                    context,
                    f,
                )
                continue
            yield f


def ignore_symlinks(directory: str, contents: list[str]) -> list[str]:
    """``shutil.copytree`` ``ignore=`` callback that drops every symlink.

    Use together with ``symlinks=False`` (the default) so that links are
    neither copied as links nor followed and resolved. Combine with any
    existing ignore predicate via a small wrapper if needed.
    """
    skipped: list[str] = []
    for name in contents:
        if Path(directory, name).is_symlink():
            skipped.append(name)
    if skipped:
        logger.warning(
            "copytree: skipping symlinked entries under %s: %s",
            directory,
            ", ".join(sorted(skipped)),
        )
    return skipped
