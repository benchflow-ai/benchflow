"""Dependency-neutral redaction policy for persisted execution configuration.

Both solver and reviewer artifacts keep nonsecret execution settings so that
scoring-only resume preserves the inference protocol without persisting auth.
"""

from __future__ import annotations

from benchflow.trajectories.types import redact_trajectory_text

# Substrings that flag an env var name as secret-bearing for ``config.json``
# redaction. Matching is case-insensitive (callers ``.upper()`` the key first)
# and uses substring containment so derived names like ``MY_AUTH_HEADER``,
# ``SESSION_COOKIE``, or ``GH_TOKEN`` are caught. This stays a denylist (rather
# than an allowlist of safe keys) because agent env varies per agent — the
# union of safe keys is not knowable here — but the list now covers the common
# auth-bearing names that issue #410 called out (COOKIE, AUTHORIZATION, AUTH,
# BEARER, SESSION) on top of the original KEY/TOKEN/SECRET/PASSWORD/CREDENTIALS.
_SECRET_ENV_SUBSTRINGS: tuple[str, ...] = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIALS",
    "COOKIE",
    "AUTHORIZATION",
    "AUTH",
    "BEARER",
    "SESSION",
)
_SECRET_URL_PATH_MARKERS: tuple[str, ...] = ("/__benchflow/",)


def _is_secret_env_key(name: str) -> bool:
    """Return True if *name* looks like it carries a secret value.

    Case-insensitive substring match against :data:`_SECRET_ENV_SUBSTRINGS`.
    Used by :func:`_write_config` to drop secret-bearing entries before
    persisting ``agent_env`` to the rollout's ``config.json``.
    """
    upper = name.upper()
    return any(s in upper for s in _SECRET_ENV_SUBSTRINGS)


def _is_secret_env_value(name: str, value: str) -> bool:
    """Return True if a normally public env value embeds a runtime secret."""
    # Inline harness JSON/TOML can contain credentials under a harmless env
    # name. Drop the whole value; redacted placeholders cannot safely replay.
    return any(marker in value for marker in _SECRET_URL_PATH_MARKERS) or (
        redact_trajectory_text(value) != value
    )


def _should_record_env_entry(name: str, value: str) -> bool:
    return not _is_secret_env_key(name) and not _is_secret_env_value(name, value)
