"""Shared configuration normalization helpers."""

from __future__ import annotations

from benchflow.agents.registry import resolve_agent_key


def normalize_agent_name(agent: str) -> str:
    """Return a stable registry key for an agent spec.

    For plain ACP agents this is the canonical agent name. For ``acpx/<agent>``
    specs the acpx-wrapped config is registered into the agent registry and its
    stable runtime key is returned, so the Rollout/Evaluation path resolves
    acpx install/launch commands instead of the literal spec string.

    Unknown bare IDs fail closed. Explicit raw commands pass through unchanged.
    A close misspelling of a registered or scripted agent (``claud-agent-acp``,
    ``oracel``) raises the pre-run check's ``ValueError`` with a suggestion;
    any other unknown bare ID raises the registry's ``KeyError``.
    """
    try:
        return resolve_agent_key(agent)
    except KeyError:
        # Checked after the lookup, so an agent the catalog provides is never
        # mistaken for a misspelling of a built-in.
        from benchflow.runtime import check_agent_names

        check_agent_names([agent])
        raise


def normalize_sandbox_user(sandbox_user: str | None) -> str | None:
    """Map text root-user sentinels to ``None``, and root itself (``root``
    or uid ``0``) too: an agent that runs as root is one case everywhere
    (the firewall-power check, the cloud metadata block), whichever way the
    run spelled it."""
    if sandbox_user is None:
        return None
    if sandbox_user.lower() in {"none", "null"}:
        return None
    if sandbox_user.strip() in {"root", "0"}:
        return None
    return sandbox_user


# Documented effort levels accepted across vendors (none → minimal → … → max).
# Vendors that expose a reasoning/thinking effort map these onto their own
# scales; values outside the set are rejected so a typo cannot silently launch
# a run with no effort applied.
_REASONING_EFFORT_VALUES = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)


def normalize_reasoning_effort(effort: object) -> str | None:
    """Normalize model reasoning/thinking effort at config boundaries."""
    if effort is None:
        return None
    if not isinstance(effort, str):
        raise ValueError("reasoning_effort must be a string")
    normalized = effort.strip().lower()
    if not normalized:
        return None
    if normalized not in _REASONING_EFFORT_VALUES:
        raise ValueError(
            "reasoning_effort must be one of: "
            + ", ".join(sorted(_REASONING_EFFORT_VALUES))
        )
    return normalized


DEFAULT_AGENT_IDLE_TIMEOUT_SEC = 600
#: BenchFlow's idle limit at the efforts where a model can reason for a quarter
#: of an hour without a word: a published run's model at max effort answered
#: after 12 to 15 minutes of reasoning that streams no summary, and the 600 s
#: default stopped it. Twice that, with room to spare.
LONG_THINKING_IDLE_TIMEOUT_SEC = 1800
_LONG_THINKING_EFFORTS = frozenset({"xhigh", "max"})


def default_agent_idle_timeout(effort: str | None) -> tuple[int, str]:
    """BenchFlow's idle limit for a run that sets none, and why: by its
    reasoning effort (``xhigh`` and ``max`` think longest). A run's own
    ``--agent-idle-timeout`` always wins."""
    if effort in _LONG_THINKING_EFFORTS:
        return LONG_THINKING_IDLE_TIMEOUT_SEC, f"effort_{effort}"
    return DEFAULT_AGENT_IDLE_TIMEOUT_SEC, "default"


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
