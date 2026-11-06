"""The pre-agent build-config snapshot takes one exec, not one per file.

Every sandbox exec is a network round trip on a remote sandbox, and
_snapshot_build_config's per-file loop (mkdir, one probe per file, manifest
write) ran at the start of every trial. It probed and copied all files in one
exec and wrote the manifest in a second; since PY8 U12 the same exec writes
the manifest too. The manifest contract is unchanged.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.sandbox.lockdown import _BUILD_CONFIG_FILES, _snapshot_build_config
from tests.exec_diet_fakes import ScriptedSandbox


@pytest.mark.asyncio
async def test_one_exec_and_the_same_manifest(tmp_path):
    """The real shell, with the snapshot folder and the workspace mapped
    into the test's directory: one root exec copies the present files and
    writes the manifest json.dumps writes."""
    app = tmp_path / "app"
    app.mkdir()
    (app / "setup.py").write_text("s")
    (app / "Makefile").write_text("m")
    snapshot = tmp_path / "snapshot"
    calls: list = []

    async def exec_(command, **kwargs):
        calls.append((command, kwargs))
        # The plugin guard target keeps the copy at a fixed root-only path
        # (V2-R12-CONF-5); map whichever one this build uses.
        mapped = command.replace(lockdown._SNAPSHOT_DIR, str(snapshot)).replace(
            "/app/", f"{app}/"
        )
        done = subprocess.run(
            ["/bin/sh", "-c", mapped], capture_output=True, text=True, timeout=60
        )
        return MagicMock(stdout=done.stdout, stderr=done.stderr, return_code=0)

    env = MagicMock()
    env.exec = exec_
    await _snapshot_build_config(env, workspace="/app")
    assert len(calls) == 1
    probe, kwargs = calls[0]
    for fname in _BUILD_CONFIG_FILES:
        assert f"/app/{fname}" in probe
    assert kwargs.get("user") == "root"
    manifest = {
        fname: fname in ("setup.py", "Makefile") for fname in _BUILD_CONFIG_FILES
    }
    assert (snapshot / "manifest.json").read_text() == json.dumps(manifest) + "\n"
    assert sorted(p.name for p in snapshot.iterdir()) == [
        "Makefile",
        "manifest.json",
        "setup.py",
    ]


@pytest.mark.asyncio
async def test_a_failed_copy_is_recorded_absent():
    # cp failed for setup.py: no line for it at all; the script's own write
    # failed too, so the host writes the manifest.
    def handler(command, user):
        return (0, "") if "cp --preserve=all" in command else (1, "")

    env = ScriptedSandbox(handler)
    await _snapshot_build_config(env, workspace="/app")
    write = env.execs[1].command
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
