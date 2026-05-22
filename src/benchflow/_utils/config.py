"""Shared configuration normalization helpers."""

from __future__ import annotations

from benchflow.agents.registry import parse_agent_spec


def normalize_agent_name(agent: str) -> str:
    """Return the canonical registry name for an ACP agent alias."""
    protocol, canonical = parse_agent_spec(agent)
    if protocol == "acp":
        return canonical
    return agent


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
    enable the watchdog for that many idle seconds.
    """
    if timeout is None:
        return None
    if isinstance(timeout, str):
        value = timeout.strip().lower()
        if value in {"", "none", "null"}:
            return None
        timeout = int(value)
    normalized = int(timeout)
    if normalized < 0:
        raise ValueError("agent_idle_timeout must be >= 0")
    return None if normalized == 0 else normalized
