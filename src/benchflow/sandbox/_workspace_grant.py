"""The workspace the sandbox user's grant made root's and 1777, around root's own scripts and the agent's next turn
(#1249). The commands are ``_runtime_tree``'s; these run them as root, read as receipts, and check what they show.

Outside ``lockdown`` so the rollout kernel can call them (it imports no concrete plane).
"""

from __future__ import annotations

import contextlib
from typing import Any

from benchflow.sandbox._exec_receipt import receipt_exec


def _return_code(result: Any) -> int | None:
    code = getattr(result, "return_code", None)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


async def close_granted_workspaces(env: Any, *, before: str) -> None:
    """Before root's own script *before* (a root ``solve.sh``): take group and
    other write from each workspace the grant made root's and sticky
    (``_runtime_tree.close_granted_command``), as root, and check it took;
    the next step that hands the workspace back
    (:func:`reopen_granted_workspaces`) gives the grant's 1777 back. Does
    nothing (no exec) without a recorded workspace. A step that does not take
    raises: root's script would meet a workspace where it can't rewrite the
    agent's files (#1249)."""
    from benchflow._runtime_tree import (
        WORKSPACE_CLOSED_ATTR,
        WORKSPACE_MODE_ATTR,
        close_granted_command,
        granted_closed,
        shown_modes_problem,
    )

    modes = getattr(env, WORKSPACE_MODE_ATTR, None)
    if not isinstance(modes, dict) or not modes:
        return
    with contextlib.suppress(AttributeError):
        # Handed back whatever happens next.
        setattr(env, WORKSPACE_CLOSED_ATTR, dict(modes))
    read = await receipt_exec(
        env, close_granted_command(modes), user="root", timeout_sec=60
    )
    code = _return_code(read)
    problem = (
        f"exited with rc={code}"
        if code != 0
        else shown_modes_problem(
            getattr(read, "stdout", "") or "", modes, granted_closed
        )
    )
    if problem is not None:
        raise RuntimeError(
            "Verifier hardening failed: taking group and other write from the "
            f"workspace before {before} ({problem})"
        )


async def reopen_granted_workspaces(env: Any) -> None:
    """Before another agent turn in a sandbox whose workspace a step took
    group and other write from (#1249 (b): a multi-step task's next step, a
    ``connect`` after ``verify``, an in-place branch child; and after a root
    ``solve.sh``): each workspace still root's and sticky gets the grant's
    1777 back, as root, and its mode is checked. Does nothing (no exec)
    unless such a step ran here (``_runtime_tree.WORKSPACE_CLOSED_ATTR``). A
    re-open that does not take raises: the agent would start without the
    workspace it had."""
    from benchflow._runtime_tree import (
        WORKSPACE_CLOSED_ATTR,
        granted_reopened,
        reopen_granted_command,
        shown_modes_problem,
    )

    closed = getattr(env, WORKSPACE_CLOSED_ATTR, None)
    if not isinstance(closed, dict) or not closed:
        return
    read = await receipt_exec(
        env, reopen_granted_command(closed), user="root", timeout_sec=60
    )
    code = _return_code(read)
    problem = (
        f"exited with rc={code}"
        if code != 0
        else shown_modes_problem(
            getattr(read, "stdout", "") or "", closed, granted_reopened
        )
    )
    if problem is not None:
        raise RuntimeError(
            "Re-opening the workspace for the agent's next turn failed "
            f"({problem}); the agent would start without the workspace the "
            "grant gave it"
        )
    with contextlib.suppress(AttributeError):
        setattr(env, WORKSPACE_CLOSED_ATTR, None)
