"""A Modal sandbox's lifetime, shared by task admission and the launcher.

Modal ends a sandbox when its ``timeout`` runs out, whatever is running in it.
BenchFlow creates each one with ``BENCHFLOW_MODAL_SANDBOX_TIMEOUT`` seconds,
and the plan gate refuses a task whose budgets do not fit in that.

Keep this module independent of the optional provider SDK: checking a task
must not import Modal.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

#: Each Modal sandbox's lifetime in whole seconds.
SANDBOX_TIMEOUT_ENV = "BENCHFLOW_MODAL_SANDBOX_TIMEOUT"
#: Modal's longest sandbox lifetime (24 h), and the default.
MODAL_SANDBOX_MAX_SEC = 86400
#: The shortest lifetime the setting takes.
MODAL_SANDBOX_MIN_SEC = 60
#: The attribute (True) of the task view a separate verifier's sandbox is
#: launched with (``rollout._separate_verifier._verifier_task``): that
#: sandbox runs no agent, so its lifetime check counts no agent budget.
VERIFIER_SANDBOX_VIEW = "_benchflow_verifier_sandbox"


def modal_sandbox_timeout(environ: Mapping[str, str] | None = None) -> int:
    """The lifetime a Modal sandbox is created with, in seconds.

    ``BENCHFLOW_MODAL_SANDBOX_TIMEOUT`` (whole seconds, 60 to 86400), else
    86400. ValueError for any other value, so no sandbox is created with it.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(SANDBOX_TIMEOUT_ENV) or "").strip()
    if not raw:
        return MODAL_SANDBOX_MAX_SEC
    if raw.isascii() and raw.isdigit():
        value = int(raw)
        if MODAL_SANDBOX_MIN_SEC <= value <= MODAL_SANDBOX_MAX_SEC:
            return value
    raise ValueError(
        f"{SANDBOX_TIMEOUT_ENV}={raw!r}: give a Modal sandbox's lifetime in whole "
        f"seconds, {MODAL_SANDBOX_MIN_SEC} to {MODAL_SANDBOX_MAX_SEC} (Modal's longest)"
    )


def modal_lifetime_issue(
    config: Any, limit_sec: int, *, verifier_sandbox: bool = False
) -> str | None:
    """Why one Modal sandbox for the task in *config* (a TaskConfig) cannot
    run in *limit_sec* seconds, or None when it fits.

    What the agent's sandbox must outlive: the agent's ``timeout_sec`` (a
    task without one counts the share the rollout's deadline gives it), the
    verifier's ``timeout_sec`` unless the verifier has a sandbox of its own,
    and the setup budgets (the sandbox user setup, ``setup_timeout_sec``, and
    each setup command's ``timeout_sec``). *verifier_sandbox*: *config* is the
    view a separate verifier's sandbox is launched with, which runs no agent:
    its verifier and setup budgets only.
    """
    from benchflow.task.config import (
        AGENT_DEADLINE_SHARE_SEC,
        effective_agent_timeout_sec,
        sandbox_user_setup_timeout_sec,
    )
    from benchflow.task.verifier_sandbox import separate_verifier_requested

    if verifier_sandbox:
        agent = 0.0
        verifier = float(config.verifier.timeout_sec)
    else:
        agent = float(
            effective_agent_timeout_sec(config.agent) or AGENT_DEADLINE_SHARE_SEC
        )
        verifier = (
            0.0
            if separate_verifier_requested(config)
            else float(config.verifier.timeout_sec)
        )
    sandbox = config.sandbox
    setup = float(sandbox_user_setup_timeout_sec(None, sandbox)) + sum(
        float(command.timeout_sec) for command in sandbox.setup_commands
    )
    need = agent + verifier + setup
    if need <= limit_sec:
        return None
    lives = (
        f"a Modal sandbox lives {limit_sec} s ({SANDBOX_TIMEOUT_ENV}, at most "
        f"{MODAL_SANDBOX_MAX_SEC})"
    )
    if verifier_sandbox:
        return (
            f"{lives}, and this task's separate verifier sandbox needs its "
            f"verifier ({verifier:.0f} s) and setup ({setup:.0f} s) budgets, "
            f"{need:.0f} s"
        )
    return (
        f"{lives}, and this task's agent ({agent:.0f} s), "
        f"verifier ({verifier:.0f} s) and setup ({setup:.0f} s) budgets add up "
        f"to {need:.0f} s"
    )
