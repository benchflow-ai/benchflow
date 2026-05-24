"""Path safety helpers — reject unsafe inputs and refuse to follow symlinks.

Two independent helper sets live here:

1. **Segment validation** (``safe_path_segment``, ``assert_within``):
   Reject user-controlled strings (case ids, skill names) that would traverse
   outside the intended tree.

2. **Symlink defense** (``is_safe_regular_file``, ``iter_safe_tree``, etc.):
   Walk directories we do not own without following symlinks, so an
   attacker-placed link cannot pull host files into dashboard payloads,
   judge prompts, or sandbox uploads.
"""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path

__all__ = [
    "safe_path_segment",
    "assert_within",
    "is_safe_regular_file",
    "is_safe_regular_dir",
    "iter_safe_children",
    "iter_safe_tree",
    "ignore_symlinks",
]

logger = logging.getLogger(__name__)


# ── Segment validation ───────────────────────────────────────────────


def safe_path_segment(name: str, *, kind: str = "name") -> str:
    """Return ``name`` unchanged if safe as a single path segment.

    Raises :class:`ValueError` for inputs that cannot be used as a directory
    or file name without risking path traversal or shell ambiguity.

    Rejected forms:

    * empty string
    * ``.`` or ``..`` (current/parent directory references)
    * any string containing ``/`` or ``\\`` (multi-segment paths)
    * any string containing a NUL byte
    * leading or trailing whitespace
    * leading ``-`` (would be interpreted as a CLI flag by downstream tools)

    All other Unicode is accepted; this is a security boundary, not a
    cosmetic slugifier. Callers that want forgiving behaviour should slugify
    *before* calling this function.

    Args:
        name: The candidate path segment.
        kind: A human label used in the error message (e.g. ``"case id"``,
            ``"skill name"``).

    Returns:
        The input ``name`` unchanged.

    Raises:
        ValueError: If ``name`` is not safe as a single path segment.
    """
    if not isinstance(name, str):
        raise ValueError(f"{kind} must be a string, got {type(name).__name__}")
    if name == "":
        raise ValueError(f"{kind} must not be empty")
    if name in (".", ".."):
        raise ValueError(f"{kind} must not be '.' or '..' (got {name!r})")
    if "/" in name or "\\" in name:
        raise ValueError(
            f"{kind} must not contain path separators (got {name!r})"
        )
    if "\x00" in name:
        raise ValueError(f"{kind} must not contain NUL bytes (got {name!r})")
    if name != name.strip():
        raise ValueError(
            f"{kind} must not have leading or trailing whitespace (got {name!r})"
        )
    if name.startswith("-"):
        raise ValueError(
            f"{kind} must not start with '-' (got {name!r}); "
            "would be misread as a CLI flag"
        )
    return name


def assert_within(child: Path, root: Path) -> Path:
    """Resolve both paths and assert ``child`` is under ``root``.

    Uses :meth:`Path.resolve` so symlinks are followed and ``..`` segments
    collapsed before the containment check. Returns the resolved child.

    Args:
        child: A path that should be inside ``root``.
        root: The directory ``child`` must not escape.

    Returns:
        The resolved ``child`` path.

    Raises:
        ValueError: If the resolved ``child`` is not under the resolved
            ``root``.
    """
    resolved_root = root.resolve()
    resolved_child = child.resolve()
    try:
        resolved_child.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"path {child} resolves to {resolved_child}, "
            f"which is outside {resolved_root}"
        ) from exc
    return resolved_child


# ── Symlink defense ──────────────────────────────────────────────────


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
    """Yield direct children of *directory*, skipping symlinks with a warning."""
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
    descended into.
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
    """``shutil.copytree`` ``ignore=`` callback that drops every symlink."""
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
