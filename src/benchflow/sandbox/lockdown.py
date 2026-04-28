"""Path lockdown for the sandbox user.

Owns:
    - Defaults and validation for locked paths (/solution, /tests, …)
    - Resolving caller overrides into an effective locked-path list
    - Locking paths at runtime (chown → chmod, reject symlinks)

Does not own:
    - Creating the sandbox user — see benchflow.sandbox.user
    - Verifier hardening — see benchflow.sandbox.verifier_harden (pending)
"""

from __future__ import annotations

import os
import re

_DEFAULT_LOCKED = ["/solution", "/tests"]
_SAFE_PATH_RE = re.compile(r"^/[a-zA-Z0-9_./*?\-]+(/[a-zA-Z0-9_./*?\-]+)*$")


def _validate_locked_path(p: str) -> None:
    """Reject injection and traversal in a locked path."""
    p_norm = os.path.normpath(p)
    if p_norm != p:
        raise ValueError(
            f"Invalid locked path {p!r}: normalizes to {p_norm!r} — "
            f"use the normalized form directly"
        )
    if any(c == ".." for c in p.split("/")):
        raise ValueError(f"Invalid locked path {p!r}: '..' component not allowed")
    if not _SAFE_PATH_RE.match(p):
        raise ValueError(
            f"Invalid locked path {p!r}: must be absolute, "
            f"alphanumeric with /-_.*? only"
        )
    if p.endswith("/") and p != "/":
        raise ValueError(
            f"Invalid locked path {p!r}: trailing slash not allowed "
            f"(chown on '/dir/' may have unintended scope)"
        )


def _resolve_locked_paths(
    sandbox_user: str | None,
    sandbox_locked_paths: list[str] | None,
) -> list[str]:
    """Resolve effective locked paths.

    - sandbox_user=None → [] (no lockdown)
    - sandbox_user set, paths=None → defaults (/solution, /tests)
    - sandbox_user set, paths=[] → [] (explicit opt-out)
    - sandbox_user set, paths=[...] → union of defaults + caller paths
    """
    if not sandbox_user:
        if sandbox_locked_paths:
            raise ValueError("sandbox_locked_paths requires sandbox_user")
        return []
    if sandbox_locked_paths is None:
        return list(_DEFAULT_LOCKED)
    if not sandbox_locked_paths:
        return []  # explicit opt-out
    return list(dict.fromkeys(_DEFAULT_LOCKED + sandbox_locked_paths))


async def lockdown_paths(env, paths: list[str]) -> None:
    """Lock directories so the sandbox user cannot access them.

    Runs after root-level setup but before agent launch.
    Uses chown-then-chmod ordering to prevent TOCTOU window.
    Rejects symlinks and validates path patterns against injection.
    """
    if not paths:
        return

    for p in paths:
        _validate_locked_path(p)

    # Build shell command: reject symlinks, chown before chmod
    parts = []
    for p in paths:
        parts.append(
            f"for d in {p}; do "
            f'  [ -L "$d" ] && echo "WARN: skipping symlink $d" >&2 && continue; '
            f'  [ -e "$d" ] || continue; '
            f'  chown root:root "$d" && chmod 700 "$d"; '
            f"done"
        )
    cmd = " && ".join(parts)
    await env.exec(cmd, timeout_sec=30)


__all__ = [
    "_DEFAULT_LOCKED",
    "_SAFE_PATH_RE",
    "_resolve_locked_paths",
    "_validate_locked_path",
    "lockdown_paths",
]
