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
from types import SimpleNamespace
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
    answer_the_pause(env)
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


#: The walk pause (sandbox/_walk_pause.py) in an empty sandbox: the uid read,
#: no process of the agent's to stop, the resume's look clean; and rule S's
#: question (sandbox/_agent_alive.py): none alive.
_PAUSE_ANSWERS = (
    ("BF pause still", "BF pause uid 1000\nBF pause at 1.0\nBF pause still 1\n"),
    ("BF resume done", "BF resume at 2.0\nBF resume done\n"),
    # Rule S (sandbox/_agent_alive.py): no process of the agent's alive.
    (
        "BF alive done",
        "BF alive uid 1000\nBF alive at 1.0\n"
        "BF alive by getpriority none found:0 none\nBF alive done\n",
    ),
)


def answer_the_pause(env: Any) -> None:
    """A unit fake's canned answer can't be the walk pause's: it gets an
    empty sandbox's. The fake's exec still runs first, so one that raises
    still fails the pause. A Mock's exec stays the same Mock, its answers
    routed through here, so a test still reads every call from it
    (``await_args_list``), the ones before this too."""
    run = getattr(env, "exec", None)
    if not callable(run) or getattr(run, "_answers_the_pause", False) is True:
        return
    if isinstance(run, mock.Mock):
        before = run.side_effect

        async def fake(*args, **kwargs):
            # The Mock's own answer, as it gave it before this.
            shadow = mock.AsyncMock(side_effect=before, return_value=run.return_value)
            return await shadow(*args, **kwargs)
    else:
        fake = run

    async def exec_(command, *args, **kwargs):
        result = await fake(command, *args, **kwargs)
        for said, answer in _PAUSE_ANSWERS:
            if isinstance(command, str) and said in command:
                return SimpleNamespace(return_code=0, stdout=answer, stderr="")
        return result

    with contextlib.suppress(AttributeError, TypeError):
        if isinstance(run, mock.Mock):
            run.side_effect = exec_
            run._answers_the_pause = True
        else:
            exec_._answers_the_pause = True
            env.exec = exec_
