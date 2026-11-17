"""A stand-in for BenchFlow's bfx on the unit tests' fake sandboxes.

Since the dev cycle, a trial in off or audit is refused before the agent's
turn where bfx cannot run (``_verifier_links.walker_unavailable``), and the
link walks and ``read_back`` use bfx alone. A unit test's fake sandbox (a
class of the test's own, a SimpleNamespace, a Mock) has no bfx, so every
model-agent trial on one was refused. This gives such a fake what a real bfx
would answer for an empty, untouched sandbox: one mount at /, no entries
(so no links, and nothing for read_back to refuse), and a removal that
succeeds. Real backends (any class from the ``benchflow`` package) are never
touched; the Docker matrix keeps the real binary. A test that asserts the
refusal itself is marked ``bfx_refusal`` and gets no stand-in.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any
from unittest import mock

#: The delivery the stand-in reports: bfx reached through ``bfx_call`` only,
#: nothing at /.bfx/bfx (so root's scripts keep their forms without bfx).
DELIVERY = "per-call"


async def fake_bfx_call(cmd: str, args: Any = None, *, timeout_sec: Any = None) -> dict:
    if cmd == "scan" and isinstance(args, dict) and args.get("mounts"):
        return {"entries": [], "mounts": [{"mount_point": "/"}]}
    return {"entries": []}


def is_unit_fake(env: Any) -> bool:
    """A sandbox the unit tests made, never one of BenchFlow's backends."""
    if env is None:
        return False
    if isinstance(env, mock.Mock):
        return True
    module = type(env).__module__ or ""
    if not (module == "benchflow" or module.startswith("benchflow.")):
        return True
    return _half_built_docker(env)


def _half_built_docker(env: Any) -> bool:
    """A DockerSandbox a unit test made with ``object.__new__``, never
    initialised: Docker's own ``__init__`` always sets ``bfx_delivery``."""
    from benchflow.sandbox.docker import DockerSandbox

    state = getattr(env, "__dict__", None)
    return (
        isinstance(env, DockerSandbox)
        and isinstance(state, dict)
        and "bfx_delivery" not in state
    )


def give(env: Any) -> None:
    """Give a unit fake the stand-in where it has no working bfx of its own."""
    call = getattr(env, "bfx_call", None)
    has_own = (
        callable(call)
        and not (isinstance(call, mock.Mock) and not isinstance(call, mock.AsyncMock))
        and not _half_built_docker(env)
    )
    try:
        if not has_own:
            env.bfx_call = fake_bfx_call
        delivery = getattr(env, "bfx_delivery", None)
        if delivery is None or isinstance(delivery, mock.Mock):
            env.bfx_delivery = DELIVERY
    except (AttributeError, TypeError):
        return
    answer_the_close(env)
    # A unit test that drives verify() on its own never ran the turn's start,
    # so its fake has no record of the links before the turn: an empty one,
    # as the stand-in's walk would have made.
    from benchflow.sandbox import _verifier_links

    if not isinstance(
        getattr(env, _verifier_links.INVENTORY, None), _verifier_links.Walked
    ):
        _verifier_links.inherit_record(env, _verifier_links.Walked("bfx", links={}))


#: What the output folder's close echoes once its chown and chmod succeeded
#: (task/verifier_core.py ``_OUTPUTS_CLOSED``).
CLOSE_MARKER = "benchflow-outputs-folder-closed"


def answer_the_close(env: Any) -> None:
    """A unit fake answers every exec with its own canned result, so the
    close's ``echo`` never shows; a real shell prints it after the chown and
    chmod succeed. A fake whose answer says the command succeeded (status 0)
    gets the marker added, as a shell would print it; a failed one keeps its
    answer."""
    run = getattr(env, "exec", None)
    if not callable(run) or getattr(run, "_answers_the_close", False):
        return

    async def exec_(command, *args, **kwargs):
        result = await run(command, *args, **kwargs)
        if (
            isinstance(command, str)
            and f"echo {CLOSE_MARKER}" in command
            and getattr(result, "return_code", None) == 0
            and CLOSE_MARKER not in str(getattr(result, "stdout", "") or "")
        ):
            result = copy.copy(result)
            result.stdout = f"{getattr(result, 'stdout', '') or ''}\n{CLOSE_MARKER}\n"
        return result

    exec_._answers_the_close = True
    with contextlib.suppress(AttributeError, TypeError):
        env.exec = exec_
