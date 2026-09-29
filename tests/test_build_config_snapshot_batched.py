"""The pre-agent build-config snapshot takes two execs, not one per file.

Every sandbox exec is a network round trip on a remote sandbox, and
_snapshot_build_config's per-file loop (mkdir, one probe per file, manifest
write) ran at the start of every trial. It now probes and copies all files in one exec and writes the
manifest in a second; the manifest contract is unchanged.
"""

from __future__ import annotations

import json
import shlex
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox.lockdown import _BUILD_CONFIG_FILES, _snapshot_build_config


@pytest.mark.asyncio
async def test_two_execs_and_the_same_manifest():
    env = MagicMock()
    env.exec = AsyncMock(
        side_effect=[
            MagicMock(stdout="setup.py=present\nMakefile=absent\n"),
            MagicMock(stdout=""),
        ]
    )
    await _snapshot_build_config(env, workspace="/app")
    assert env.exec.await_count == 2
    probe = env.exec.await_args_list[0].args[0]
    for fname in _BUILD_CONFIG_FILES:
        assert f"/app/{fname}" in probe
    write = env.exec.await_args_list[1].args[0]
    manifest = json.loads(shlex.split(write)[1])
    assert manifest == {fname: fname == "setup.py" for fname in _BUILD_CONFIG_FILES}
    assert all(c.kwargs.get("user") == "root" for c in env.exec.await_args_list)


@pytest.mark.asyncio
async def test_a_failed_copy_is_recorded_absent():
    env = MagicMock()
    # cp failed for setup.py: no line for it at all.
    env.exec = AsyncMock(side_effect=[MagicMock(stdout=""), MagicMock(stdout="")])
    await _snapshot_build_config(env, workspace="/app")
    write = env.exec.await_args_list[1].args[0]
    assert json.loads(shlex.split(write)[1])["setup.py"] is False


@pytest.mark.asyncio
async def test_log_dirs_are_prepared_in_one_exec():
    """A branch child's /logs ownership: one exec, same commands."""
    from benchflow.sandbox.lockdown import _prepare_log_dirs

    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout=""))
    await _prepare_log_dirs(env, sandbox_user="agent")
    assert env.exec.await_count == 1
    cmd = env.exec.await_args.args[0]
    assert "chown root:root /logs && chmod 755 /logs" in cmd
    assert "chown agent:agent /logs/agent /logs/artifacts" in cmd
    assert env.exec.await_args.kwargs["user"] == "root"
