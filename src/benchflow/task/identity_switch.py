"""Accounts a task lets its agent switch to: ``x-agent-identity-switch``.

Agent code runs under no_new_privs (K1-R11-1), so it keeps the agent user's
uid and the kills, which find processes by uid, stop all of it. A task whose
agent must run commands as another account (its image grants ``sudo -u``
to it) names those accounts under ``[metadata] x-agent-identity-switch`` in
task.toml, or ``metadata: x-agent-identity-switch:`` in task.md front matter:
a list of user names. Every kill of the agent user's processes stops those
accounts' processes too. root is never accepted.

Such a task runs only on a backend whose agent container is created with
no_new_privs (Docker, remote Docker, Daytona with a compose file),
where no process can gain another identity. Elsewhere it is refused before
any sandbox starts (:func:`backend_refusal`): the accounts are cleaned one
after another, and a process under one not yet cleaned could restart a
process under one already cleaned, which would survive into grading
(K1-R15-2). Agent commands keep no_new_privs on every backend.
"""

from __future__ import annotations

import re
from typing import Any

KEY = "x-agent-identity-switch"
#: At most this many accounts.
MAX_ACCOUNTS = 8
_NAME = re.compile(r"[a-z_][a-z0-9_-]{0,31}")


def read(value: Any) -> tuple[str, ...]:
    """The accounts a ``x-agent-identity-switch`` value names.

    Raises :class:`ValueError` saying what is wrong when it is not a list of
    at most :data:`MAX_ACCOUNTS` user names other than root.
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"{KEY} must be a non-empty list of user names")
    if len(value) > MAX_ACCOUNTS:
        raise ValueError(f"{KEY} names at most {MAX_ACCOUNTS} accounts")
    names = []
    for name in value:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError(f"{KEY}: {name!r} is not a user name")
        if name == "root":
            raise ValueError(f"{KEY}: root cannot be named")
        if name not in names:
            names.append(name)
    return tuple(names)


def for_task(task: Any) -> tuple[str, ...]:
    """The accounts ``task`` declares, or ``()``."""
    metadata = getattr(getattr(task, "config", None), "metadata", None)
    if not isinstance(metadata, dict) or KEY not in metadata:
        return ()
    return read(metadata[KEY])


def declared(config: Any) -> bool:
    """Whether a task config names the key at all (valid or not)."""
    metadata = getattr(config, "metadata", None)
    return isinstance(metadata, dict) and KEY in metadata


def backend_holds(sandbox: str, *, compose: bool) -> bool:
    """Whether ``sandbox`` creates the agent's container with no_new_privs
    for this task (``compose``: the task has a compose file)."""
    from benchflow.sandbox.providers import PROVIDERS_BY_NAME

    provider = PROVIDERS_BY_NAME.get(sandbox)
    holds = getattr(provider, "agent_container_no_new_privs", "never")
    return holds == "always" or (holds == "compose" and compose)


def refusal(sandbox: str) -> str:
    """Why an identity-switch task cannot run on ``sandbox``, and where it can."""
    return (
        f"the {sandbox} sandbox cannot stop the processes of every account the "
        "agent may switch to together before grading: they are cleaned one "
        "after another, so a process under one account could restart a "
        "process under an account already cleaned, and it would survive into "
        "grading. Run this task with --sandbox docker (or "
        "remote-docker, or daytona with a docker-compose file), whose agent "
        "container is created with no_new_privs; there every agent command "
        f"keeps the agent's own account. Or remove {KEY}"
    )


def backend_refusal(sandbox: str, *, compose: bool) -> str | None:
    """:func:`refusal` when ``sandbox`` lacks the barrier for this task."""
    return None if backend_holds(sandbox, compose=compose) else refusal(sandbox)


def require_barrier(sandbox: Any, *, backend: str | None = None) -> None:
    """Require the sandbox's own container no_new_privs capability.

    SDK admission checks this before start; install_agent repeats it before
    any agent code. Missing or unknown capabilities fail closed, including
    a custom plane whose sandbox has not yet been made.
    """
    if getattr(sandbox, "agent_container_no_new_privileges", False) is True:
        return
    from benchflow.task.runtime_capabilities import (
        IDENTITY_SWITCH_ISSUE_PATH,
        UnsupportedTaskFeature,
        UnsupportedTaskFeatureError,
    )

    name = backend or type(sandbox).__name__
    raise UnsupportedTaskFeatureError(
        [
            UnsupportedTaskFeature(
                path=IDENTITY_SWITCH_ISSUE_PATH, reason=refusal(name), sandbox=name
            )
        ]
    )
