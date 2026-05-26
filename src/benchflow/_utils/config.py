"""Shared configuration normalization helpers."""

from __future__ import annotations

from benchflow.agents.registry import resolve_agent_key


def normalize_agent_name(agent: str) -> str:
    """Return a stable registry key for an agent spec.

    For plain ACP agents this is the canonical agent name. For ``acpx/<agent>``
    specs the acpx-wrapped config is registered into the agent registry and its
    stable runtime key is returned, so the Rollout/Evaluation path resolves
    acpx install/launch commands instead of the literal spec string.

    Unknown specs are returned unchanged.
    """
    return resolve_agent_key(agent)


def normalize_sandbox_user(sandbox_user: str | None) -> str | None:
    """Map text root-user sentinels to ``None``."""
    if sandbox_user is None:
        return None
    if sandbox_user.lower() in {"none", "null"}:
        return None
    return sandbox_user


DEFAULT_AGENT_IDLE_TIMEOUT_SEC = 600


def normalize_agent_idle_timeout(timeout: object) -> int | None:
    """Normalize ACP idle timeout config.

    ``None`` and ``0`` both disable idle detection. Positive integer values
    enable the watchdog for that many idle seconds. Numeric strings are accepted
    at config boundaries, but booleans and floats are rejected to avoid silent
    integer coercion.
    """
    if timeout is None:
        return None
    if isinstance(timeout, str):
        value = timeout.strip().lower()
        if value in {"", "none", "null"}:
            return None
        unsigned = value[1:] if value.startswith("-") else value
        if not unsigned.isdecimal():
            raise ValueError("agent_idle_timeout must be null or integer seconds")
        timeout = int(value)
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise ValueError("agent_idle_timeout must be null or integer seconds")
    normalized = timeout
    if normalized < 0:
        raise ValueError("agent_idle_timeout must be >= 0")
    return None if normalized == 0 else normalized
