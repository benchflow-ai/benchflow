"""Failure scanning for verifier logs.

Extracted from ``benchflow.task.verifier`` as a pure leaf cluster. Streams the
verifier ``test-stdout.txt`` off disk to detect dependency-install failures
without ever returning or persisting the scanned (secret-bearing) text, and
detects a pytest that could not load the plugin guard hardening injected.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from benchflow._utils.scoring import (
    VERIFIER_DEP_INSTALL_MARKERS,
    contains_verifier_dep_install_marker,
)

# The safe, fixed diagnostic surfaced into ``verifier_error`` on a detected
# dep-install failure. Contains a marker the classifier recognises; carries no
# stdout content. Points operators at the (private) log artifact for detail.
_DEP_INSTALL_DIAGNOSTIC = (
    "dependency install failed (see verifier/test-stdout.txt in the run "
    "artifacts for resolver output)"
)

# Size of each fixed chunk streamed off disk while scanning for markers. The
# whole file is scanned (dep-install runs at the START of test.sh, so the marker
# can be buried under arbitrarily many trailing lines — see PR #572), but only
# one bounded chunk is ever held in memory at a time, so a verifier emitting a
# single huge line can't bloat memory. We never persist any of the scanned text.
_SCAN_CHUNK_BYTES = 64 * 1024


def _has_dep_install_failure(path: Path) -> bool:
    """True if *path* (test-stdout.txt) shows a dependency-install failure.

    Only the boolean verdict leaves this function — the scanned text is never
    returned or persisted, so no secret-bearing stdout can reach result
    metadata (PR #572).

    The ENTIRE file is scanned from the start in fixed ``_SCAN_CHUNK_BYTES``
    chunks (with a small overlap so a marker straddling a chunk boundary is
    still caught), short-circuiting on the first marker. uv/pip install runs at
    the START of ``test.sh``, so its failure marker may be followed by many
    lines of trailing output (fallback attempts, cleanup, partial tests); a
    tail-only scan would silently drop it (PR #572). Memory stays bounded — at
    most one chunk plus the overlap is held at a time, regardless of file size.
    """
    return _stream_contains(
        path,
        contains_verifier_dep_install_marker,
        max(len(m) for m in VERIFIER_DEP_INSTALL_MARKERS),
    )


def _has_guard_load_failure(verifier_dir: Path, guard: str) -> bool:
    """True if a verifier log shows pytest could not load the guard *guard*.

    pytest wraps any import error of a ``-p`` plugin as ``Error importing
    plugin "<name>"`` (pytest 3 through 9); pluggy names the plugin as
    ``Plugin '<name>'`` when it refuses its hooks. The guard name carries a
    random suffix chosen after the solver stopped, so a solution cannot
    pre-plant these lines. Every top-level file is scanned because test.sh may
    send pytest output to its own log instead of stdout.
    """
    markers = (
        f'Error importing plugin "{guard}"',
        f"No module named '{guard}'",
        f"Plugin '{guard}'",
    )
    try:
        paths = sorted(p for p in verifier_dir.iterdir() if p.is_file())
    except OSError:
        return False
    return any(
        _stream_contains(
            path,
            lambda text: any(marker in text for marker in markers),
            max(len(m) for m in markers),
        )
        for path in paths
    )


def _stream_contains(
    path: Path, found: Callable[[str], bool], longest_marker: int
) -> bool:
    """Stream *path* in bounded chunks until *found* matches a window."""
    # Longest marker minus one byte: enough overlap to catch a marker split
    # across two reads without rescanning whole chunks.
    overlap = longest_marker - 1
    try:
        with path.open(errors="replace") as f:
            carry = ""
            while True:
                chunk = f.read(_SCAN_CHUNK_BYTES)
                if not chunk:
                    return False
                window = carry + chunk
                if found(window):
                    return True
                # Keep the tail of this chunk so a boundary-spanning marker is
                # found on the next read; bound it to the overlap size.
                carry = chunk[-overlap:] if overlap else ""
    except OSError:
        return False
