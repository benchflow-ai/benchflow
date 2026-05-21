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
