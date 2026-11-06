"""The kill before grading runs as ``bfx kill`` where bfx is delivered
(TRUSTED-RUNTIME v2 §4, K1 row), with the same receipt and failure
semantics as the shell kill, which still runs everywhere else."""

from __future__ import annotations

from typing import Any

import pytest

from benchflow._bfx.client import BfxAmbiguous, BfxError, BfxNotStarted
from benchflow._bfx.locate import BfxUnavailable
from benchflow.sandbox import lockdown
from benchflow.sandbox._identities import Identities

PASSWD = "root:x:0:0::/root:/bin/sh\nagent:x:1000:1000::/home/agent:/bin/sh\n"
GROUP = "root:x:0:\nagent:x:1000:\n"


class Box:
    """A sandbox with bfx delivered and the pre-agent identities measured."""

    bfx_delivery: str | None = "bind"

    def __init__(self, answer: Any = None, *, identities: bool = True) -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict]] = []
        self.shell_commands: list[str] = []
        if identities:
            self._benchflow_identities = Identities(PASSWD, GROUP)

    async def bfx_call(self, cmd: str, args: dict, *, timeout_sec: int) -> dict:
        self.calls.append((cmd, args))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


@pytest.fixture(autouse=True)
def shell_kill(monkeypatch):
    """The shell kill's exec, recorded on the box; it finds nothing to kill."""
    from benchflow.sandbox.protocol import ExecResult

    async def read_exec(env, command, **kwargs):
        env.shell_commands.append(command)
        return ExecResult(return_code=0, stdout="BF killed 0 end\n", stderr="")

    monkeypatch.setattr(lockdown, "_read_exec", read_exec)


async def _kill(box: Box, user: str = "agent") -> int:
    return await lockdown._kill_sandbox_user_procs(box, user, record=False)


async def test_bfx_kill_uses_the_pre_agent_numbers_and_its_receipt():
    box = Box({"killed": 3, "receipt": "BF killed 3 end\n"})
    assert await _kill(box) == 3
    assert box.calls == [("kill", {"accounts": [{"uid": 1000, "gid": 1000}]})]
    assert box.shell_commands == []


@pytest.mark.parametrize(
    "receipt",
    ["", "BF killed 3\n", "BF killed 1 end\nBF killed 2 end\n", None],
)
async def test_bfx_receipt_without_one_whole_count_fails(receipt):
    box = Box({"killed": 1, "receipt": receipt})
    with pytest.raises(RuntimeError, match="quiescence"):
        await _kill(box)


async def test_root_account_is_refused_before_bfx_runs():
    box = Box({"receipt": "BF killed 0 end\n"})
    with pytest.raises(RuntimeError, match="identity check"):
        await _kill(box, "root")
    assert box.calls == []


@pytest.mark.parametrize(
    ("kind", "what"),
    [("survivors", "quiescence"), ("unobservable", "quiescence"),
     ("invalid-account", "identity")],
)  # fmt: skip
async def test_bfx_refusal_fails_with_the_shell_wording(kind, what):
    box = Box(BfxError(kind, "a message"))
    with pytest.raises(RuntimeError, match=f"process {what} check"):
        await _kill(box)
    assert box.shell_commands == []


async def test_a_kill_that_may_have_run_is_not_sent_again():
    box = Box(BfxAmbiguous("channel died", 1, True))
    with pytest.raises(RuntimeError, match="no bfx answer"):
        await _kill(box)
    assert len(box.calls) == 1
    assert box.shell_commands == []


@pytest.mark.parametrize(
    "answer", [BfxNotStarted("not-started", "withdrawn"), BfxUnavailable("gone")]
)
async def test_a_kill_bfx_did_not_start_runs_as_the_shell_kill(answer):
    box = Box(answer)
    assert await _kill(box) == 0
    assert len(box.shell_commands) == 1


async def test_without_bfx_the_shell_kill_runs():
    box = Box({"receipt": "BF killed 9 end\n"})
    box.bfx_delivery = None
    assert await _kill(box) == 0
    assert box.calls == []
    assert len(box.shell_commands) == 1


async def test_without_pre_agent_identities_the_shell_kill_runs():
    box = Box({"receipt": "BF killed 9 end\n"}, identities=False)
    assert await _kill(box) == 0
    assert box.calls == []


async def test_a_stand_in_proc_root_keeps_the_shell_kill():
    box = Box({"receipt": "BF killed 9 end\n"})
    killed = await lockdown._kill_sandbox_user_procs(
        box, "agent", proc_root="/tmp/fake-proc", record=False
    )
    assert killed == 0
    assert box.calls == []


async def test_switched_accounts_are_killed_through_bfx_too():
    box = Box({"receipt": "BF killed 2 end\n"})
    box._benchflow_identities = Identities(
        PASSWD + "worker:x:1001:1001::/home/worker:/bin/sh\n",
        GROUP + "worker:x:1001:\n",
    )
    box.agent_identity_switch = ("worker",)
    assert await lockdown.kill_switched_accounts(box) == 2
    assert box.calls == [("kill", {"accounts": [{"uid": 1001, "gid": 1001}]})]
