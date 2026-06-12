"""Dep-install failure scanning for verifier test stdout.

Extracted from ``benchflow.task.verifier`` as a pure leaf cluster. Streams the
verifier ``test-stdout.txt`` off disk to detect dependency-install failures
without ever returning or persisting the scanned (secret-bearing) text.
"""

from __future__ import annotations

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
    # Longest marker minus one byte: enough overlap to catch a marker split
    # across two reads without rescanning whole chunks.
    overlap = max(len(m) for m in VERIFIER_DEP_INSTALL_MARKERS) - 1
    try:
        with path.open(errors="replace") as f:
            carry = ""
            while True:
                chunk = f.read(_SCAN_CHUNK_BYTES)
                if not chunk:
                    return False
                window = carry + chunk
                if contains_verifier_dep_install_marker(window):
                    return True
                # Keep the tail of this chunk so a boundary-spanning marker is
                # found on the next read; bound it to the overlap size.
                carry = chunk[-overlap:] if overlap else ""
    except OSError:
        return False
