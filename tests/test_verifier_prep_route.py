"""Verifier preparation runs on one root route.

U5a runs the preparation steps as one script on Docker and Daytona, each step
in its own exec elsewhere. V3 moved root's plumbing onto the system route
(``with_system_path``: loader and startup variables unset, PATH the system
folders). These hold both shapes, the soft-verify clean-up and the kept-service
ancestry check to that one route, with the clean-up still finding the
directories and the interpreter it needs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox._base import SYSTEM_PATH, ExecResult
from benchflow.task import keep_services
from benchflow.task.keep_services import KeepService, KeptServices
from tests.test_sandbox_hardening import _make_env, _make_task
from tests.test_verifier_prep_script import ShellSandbox, _step

# What a step sees: the PATH it runs on and the clean-up's PYTHONPATH input.
_PROBE = 'printf "%s\\n%s\\n" "$PATH" "${__bf_hook_pythonpath-unset}"'
_HOOKS = "/hooks-in"


def _run(command: str, path: str | None = None) -> subprocess.CompletedProcess:
    """Run *command* in ``sh`` as a sandbox would, with the image's PYTHONPATH."""
    return subprocess.run(
        ["sh", "-c", command],
        capture_output=True,
        text=True,
        env={"PATH": path or os.environ["PATH"], "PYTHONPATH": _HOOKS},
    )


@pytest.mark.asyncio
async def test_script_steps_run_on_the_root_route(tmp_path, monkeypatch):
    """A script-capable backend runs every step on the same PATH as the
    step-by-step helpers, the image's PYTHONPATH kept for the clean-up."""
    monkeypatch.setenv("PYTHONPATH", _HOOKS)
    out = tmp_path / "out"
    steps = [_step("probe", f"{_PROBE} > {shlex.quote(str(out))}", label="probe")]
    await lockdown._run_prep_steps(ShellSandbox(tmp_path), steps)
    assert out.read_text().splitlines() == [SYSTEM_PATH, _HOOKS]


@pytest.mark.asyncio
async def test_step_by_step_clean_up_keeps_its_inputs_on_the_route(monkeypatch):
    """One exec per step (Modal, Apple Container, kept services): the clean-up
    is on the route and still sees the PYTHONPATH it must purge (U5a-CONF-5)."""
    seen: list[list[str]] = []

    def execute(command, **kwargs):
        if _PROBE in command:
            seen.append(_run(command).stdout.splitlines())
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(lockdown, "_build_cleanup_cmd", lambda *a, **k: _PROBE)
    monkeypatch.setattr(lockdown, "_reclaim_disk", AsyncMock())
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    monkeypatch.setattr(lockdown, "_build_verifier_env", AsyncMock(return_value={}))
    await lockdown.harden_before_verify(
        _make_env(execute), _make_task(), sandbox_user=None, workspace=None
    )
    assert seen == [[SYSTEM_PATH, _HOOKS]]


@pytest.mark.asyncio
async def test_soft_verify_clean_up_keeps_its_inputs_on_the_route(monkeypatch):
    seen: list[list[str]] = []

    def execute(command, **kwargs):
        if _PROBE in command:
            seen.append(_run(command).stdout.splitlines())
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(lockdown, "_build_cleanup_cmd", lambda *a, **k: _PROBE)
    await lockdown.cleanup_verifier_python_hooks(_make_env(execute), None)
    assert seen == [[SYSTEM_PATH, _HOOKS]]


def test_clean_up_python_is_the_judged_interpreter(tmp_path):
    """The purge runs the interpreter the host judged, the one
    the route names, with the saved PYTHONPATH (as PT-R16's _BF_HOOK_PYTHONPATH
    data, merge wave 32b); never a python3 by name (the route's system PATH
    has no /usr/local/bin, a python:* image's only one)."""
    marker = tmp_path / "marker"
    stub = tmp_path / "judged" / "python3"
    stub.parent.mkdir()
    stub.write_text(f'#!/bin/sh\nprintf "%s" "$_BF_HOOK_PYTHONPATH" > {marker}\n')
    stub.chmod(0o755)
    command = lockdown._build_cleanup_cmd({"cleanup_conftests": False})
    # Only the hook purge: the /tmp sweep before it deletes files.
    sweep, _, purge = command.partition("; ")
    assert sweep == "find /tmp /var/tmp -name '*.py' -delete 2>/dev/null"
    token = lockdown._root_python.set(str(stub))
    try:
        routed = lockdown._prep_on_path(purge)
    finally:
        lockdown._root_python.reset(token)
    done = subprocess.run(
        ["sh", "-c", routed],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            "PATH": os.environ["PATH"],
            "PYTHONPATH": _HOOKS,
            "HOME": str(tmp_path),
            "PYTHONNOUSERSITE": "1",
        },
    )
    assert done.returncode == 0, done.stderr
    assert marker.exists(), "the purge did not run the judged python3"
    assert marker.read_text() == _HOOKS


@pytest.mark.asyncio
async def test_kept_service_ancestry_check_runs_on_the_root_route(
    tmp_path, monkeypatch
):
    """The kept-service workspace-ancestry check never runs a ``python3`` the
    image PATH names; it still refuses a replaceable ancestor."""
    planted = tmp_path / "planted"
    planted.mkdir()
    marker = tmp_path / "planted-ran"
    stub = planted / "python3"
    stub.write_text(f"#!/bin/sh\n: > {marker}\n")
    stub.chmod(0o755)
    workspace = tmp_path / "job" / "submission"
    workspace.mkdir(parents=True)
    workspace.parent.chmod(0o777)

    def execute(command, **kwargs):
        if "Cannot protect workspace resolution" in command:
            done = _run(command, f"{planted}:{os.environ['PATH']}")
            return ExecResult(
                stdout=done.stdout, stderr=done.stderr, return_code=done.returncode
            )
        return ExecResult(stdout="", stderr="", return_code=0)

    for name in ("_freeze_workspace", "_reclaim_disk"):
        monkeypatch.setattr(lockdown, name, AsyncMock())
    monkeypatch.setattr(lockdown, "_record_kept", lambda *a, **kw: None)
    monkeypatch.setattr(
        keep_services,
        "trial_keep",
        lambda *a: KeptServices((KeepService(5000),), "the task"),
    )
    monkeypatch.setattr(
        lockdown, "_trusted_verifier_bash_env", AsyncMock(return_value="")
    )
    monkeypatch.setattr(
        lockdown,
        "_kill_sandbox_user_procs_keeping",
        AsyncMock(return_value=lockdown.KeepingKill(0, (), 0.0)),
    )
    build = AsyncMock(return_value={})
    monkeypatch.setattr(lockdown, "_build_verifier_env", build)
    with pytest.raises(RuntimeError, match="workspace ancestors"):
        await lockdown.harden_before_verify(
            _make_env(execute),
            _make_task(),
            sandbox_user="agent",
            workspace=str(workspace),
        )
    assert not Path(marker).exists()
    build.assert_not_awaited()
