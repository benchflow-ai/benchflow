"""rows M4 and A3: a new bfx route serves the walker and read_back
alone.

Where bfx sits at /.bfx (a bind, a volume, an upload), Docker's root scripts
start it too: every bounded root exec through ``bfx launch``, the ``bfx_sh``
steps, strict's K1 kill. Modal's volume and AgentCore's upload put it there
as well, but those uses have never run on either backend, and strict runs on
Modal today: ``bfx_in_scripts`` is False there, so their root scripts and
strict's kill stay as they were. Docker keeps them (the control).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._bounded_exec import bfx_delivered

pytestmark = pytest.mark.bfx_refusal


def _box(cls, mode):
    box = object.__new__(cls)
    box.bfx_delivery = mode
    return box


def _kill_reaches_bfx(monkeypatch, box) -> list:
    """Whether ``_bfx_kill_receipt`` would send ``bfx kill``: an identity
    table that resolves to a non-root account is in place."""
    calls = []

    async def bfx_call(cmd, args, *, timeout_sec):
        calls.append(cmd)
        return {"receipt": "r"}

    monkeypatch.setattr(lockdown, "Identities", object)
    monkeypatch.setattr(
        lockdown,
        "resolve_identity",
        lambda table, user: SimpleNamespace(uid=1000, gid=1000),
    )
    monkeypatch.setattr(lockdown, "_init_may_be", lambda table, identity: False)
    vars(box)["_benchflow_identities"] = object()
    box.bfx_call = bfx_call
    return calls


@pytest.mark.parametrize(
    "backend,mode",
    [("modal", "volume"), ("agentcore", "upload")],
)
async def test_a_new_route_leaves_root_scripts_and_strict_kill_alone(
    monkeypatch, backend, mode
):
    """M4 (Modal) and A3 (AgentCore). Mutant: ``bfx_in_scripts`` ignored."""
    if backend == "modal":
        from benchflow.sandbox.modal_impl import ModalSandbox as cls
    else:
        from benchflow.sandbox.agentcore import AgentCoreSandbox as cls
    assert cls.bfx_in_scripts is False
    box = _box(cls, mode)
    assert bfx_delivered(box) is False
    assert lockdown._bfx_in_scripts(box) is False
    calls = _kill_reaches_bfx(monkeypatch, box)
    assert await lockdown._bfx_kill_receipt(box, "agent") is None
    assert calls == []


async def test_docker_keeps_bfx_in_its_root_scripts(monkeypatch):
    """The control: Docker's bind is in its root scripts and its K1 kill."""
    from benchflow.sandbox.docker import DockerSandbox

    box = _box(DockerSandbox, "bind")
    assert bfx_delivered(box) is True
    calls = _kill_reaches_bfx(monkeypatch, box)
    assert await lockdown._bfx_kill_receipt(box, "agent") == "r"
    assert calls == ["kill"]
