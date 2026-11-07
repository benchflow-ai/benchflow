"""Neutral names at the agent process boundary; host aliases remain accepted."""

from __future__ import annotations


def visible_name(name: str) -> str:
    return name.replace("BENCHFLOW_", "RUNTIME_")


def visible_env(env: dict[str, str]) -> dict[str, str]:
    """Translate only names, never credentials or user-authored values."""
    return {
        visible_name(name): value
        for name, value in env.items()
        if name
        not in {
            "_BENCHFLOW_CREDENTIAL_POLICY",
            "_BENCHFLOW_SUBSCRIPTION_AUTH",
            "BENCHFLOW_LOGIN_LABEL",
        }
    }


def visible_script(source: str) -> str:
    """Translate our deployed launcher source, before encoding it for upload."""
    return source.replace("BENCHFLOW_", "RUNTIME_")


def visible_launch(command: str) -> str:
    """Remove image-inherited legacy aliases inside the agent's identity.

    Root transport variables are needed until the privilege drop finishes.
    Only then remove them, before the agent and its subprocesses start.
    """
    return (
        "for runtime_name in $(env | sed -n 's/^\\([^=]*BENCHFLOW_[^=]*\\)=.*/\\1/p'); "
        'do unset "$runtime_name"; done; unset runtime_name; ' + visible_script(command)
    )


def host_env_aliases(env: dict[str, str]) -> dict[str, str]:
    """Accept neutral spelling from SDK callers without changing host consumers."""
    return {
        name.replace("RUNTIME_", "BENCHFLOW_", 1)
        if name.startswith("RUNTIME_")
        else name: value
        for name, value in env.items()
    }
