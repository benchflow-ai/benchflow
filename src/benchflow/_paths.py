"""Path safety helpers — reject (don't slugify) unsafe path inputs.

BenchFlow accepts user-controlled strings (case ids, skill names, trace file
paths) and uses them as filesystem path segments. A name like ``../escape``
or ``a/b`` traverses outside the intended tree, so we validate each segment
before it is used as a path component.

These helpers **reject** invalid names rather than slugifying them. The reasoning:

* A silent slug rewrites the identifier the caller passed in; two distinct
  inputs (``a/b`` and ``a-b``) collapse to the same directory, which can
  shadow data or make GEPA / job-result lookups fall back to the wrong case.
* Slugification can also mask programmer error — a typo that produces ``..``
  silently maps to a usable name instead of surfacing the bug.
* Callers that genuinely need a forgiving UX (e.g. user-facing tooling that
  derives a slug from a free-form title) can slugify upstream, then call
  :func:`safe_path_segment` on the slug as a final guard.

For containment checks across multiple segments (or once a full path has been
constructed), use :func:`assert_within`, which resolves symlinks and verifies
the resulting path stays under a known root.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["safe_path_segment", "assert_within"]


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

    Defense-in-depth helper for sites that already validated each path
    segment but want to be sure no later code (e.g. ``shutil.copytree``,
    ``Path.write_text``) escaped via an unexpected route.

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
