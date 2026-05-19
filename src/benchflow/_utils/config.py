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
