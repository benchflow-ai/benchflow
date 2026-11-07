"""Guard d7736754: ancestry approval must precede verifier construction."""

import shlex
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import ExecResult
from benchflow.task import keep_services
from benchflow.task.keep_services import KeepService, KeptServices
from tests._root_python_fakes import root_python_parts
from tests.test_sandbox_hardening import _make_env, _make_task


@pytest.mark.parametrize("workspace", ["/srv/job/submission", "/tmp/quote' line\n"])
@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("status", [0, 1, 127, "timeout"])
async def test_ancestry_is_checked_after_closure_before_verifier(
    monkeypatch, workspace, restore, status
):
    events = []

    async def freeze(*args, **kwargs):
        events.append("freeze")

    async def restore_state(*args):
        events.append("restore")

    async def build(*args, **kwargs):
        events.append("build")
        return {}

    def execute(command, **kwargs):
        if "bf-root-python" in command:
            # The judge accepts the image's interpreter.
            return ExecResult(stdout="A /usr/bin/python3\n", stderr="", return_code=0)
        if lockdown._close_workspace_cmd(workspace) in command:
            events.append("close")
        if "Cannot protect workspace resolution" in command:
            events.append("ancestry")
            assert events == (["restore"] if restore else []) + [
                "freeze",
                "close",
                "ancestry",
            ]
            assert kwargs["user"] == "root"
            # Root's helper: the kit's Python through root_python_command with
            # a kit, else the interpreter V3 judged; no kit here.
            parts = root_python_parts(command)
            if parts is not None:
                assert parts[1][-1] == workspace
            else:
                # (the fake env has taken the route's launch text off)
                assert command.removeprefix(lockdown._keep_route_path("")).startswith(
                    "cd / && /usr/bin/python3 -I -S -c "
                )
                assert shlex.split(command)[-1] == workspace
            if status == "timeout":
                raise TimeoutError("dummy incomplete ancestry inspection")
            return ExecResult(stdout="", stderr="dummy receipt", return_code=status)
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(lockdown, "_freeze_workspace", freeze)
    monkeypatch.setattr(lockdown, "_restore_workspace_state", restore_state)
    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    monkeypatch.setattr(lockdown, "_reclaim_disk", AsyncMock())
    monkeypatch.setattr(lockdown, "_record_kept", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    monkeypatch.setattr(
        keep_services,
        "trial_keep",
        lambda *args: KeptServices((KeepService(5000),), "trusted task"),
    )
    monkeypatch.setattr(
        lockdown,
        "_kill_sandbox_user_procs_keeping",
        AsyncMock(return_value=lockdown.KeepingKill(0, (), 0.0)),
    )
    call = lockdown.harden_before_verify(
        _make_env(execute),
        _make_task(),
        sandbox_user="agent",
        workspace=workspace,
        restore_workspace=restore,
    )
    if status == 0:
        await call
    else:
        with pytest.raises((RuntimeError, TimeoutError)):
            await call
    expected = (["restore"] if restore else []) + ["freeze", "close", "ancestry"]
    assert events == expected + (["build"] if status == 0 else [])


async def test_without_kept_processes_no_ancestry_interpreter_is_needed(monkeypatch):
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        return ExecResult(stdout="", stderr="", return_code=0)

    for name in ("_freeze_workspace", "_reclaim_disk", "_kill_sandbox_user_procs"):
        monkeypatch.setattr(lockdown, name, AsyncMock())
    monkeypatch.setattr(keep_services, "trial_keep", lambda *args: None)
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    build = AsyncMock(return_value={})
    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    await lockdown.harden_before_verify(
        _make_env(execute), _make_task(), sandbox_user="agent", workspace="/app"
    )
    build.assert_awaited_once()
    assert all("Cannot protect workspace resolution" not in cmd for cmd in commands)
