"""Guards the staging correction against the L08 defect at fcc733f466.

The real upload command runs in a disposable local tree. Only its absolute
/opt prefix and transport are substituted. A root-owned chmod wrapper gives
UID 1000 a deterministic opportunity to rename the staging root immediately
before chmod; the later extraction proof is deliberately left intact.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.modal_impl import ModalSandbox
from benchflow.task.config import SandboxConfig
from tests._trust_admission import admitted

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0,
    reason="requires a disposable Linux container with root and UID switching",
)


@pytest.fixture
def image_tree():
    # A separate top-level temporary folder lets UID 1000 traverse the route
    # without changing the permissions of pytest's own shared temporary root.
    with tempfile.TemporaryDirectory(prefix="bf-modal-stage-") as folder:
        root = Path(folder)
        root.chmod(0o755)
        (root / "Dockerfile").write_text("FROM scratch\n")
        (root / "opt" / "benchflow").mkdir(parents=True)
        (root / "sentinel").mkdir(mode=0o751)
        (root / "source").mkdir()
        (root / "source" / "answer").write_text("ordinary upload\n")
        yield root


def _sandbox(root: Path, monkeypatch, *, attack: bool):
    tools = root / "tools"
    tools.mkdir()
    run = root / "opt" / "benchflow" / "run"
    sentinel = root / "sentinel"
    outcomes = root / "swap-results"
    helper = tools / "swap.py"
    helper.write_text(
        "import os, sys\n"
        f"run = {str(run)!r}\n"
        f"sentinel = {str(sentinel)!r}\n"
        "read_fd, write_fd = os.pipe()\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.close(read_fd)\n"
        "    os.setgroups([])\n"
        "    os.setgid(1000)\n"
        "    os.setuid(1000)\n"
        "    try:\n"
        "        os.rename(run, run + '-held')\n"
        "        os.symlink(sentinel, run)\n"
        "    except PermissionError:\n"
        "        outcome = b'denied\\n'\n"
        "    else:\n"
        "        outcome = b'swapped\\n'\n"
        "    os.write(write_fd, outcome)\n"
        "    os._exit(0)\n"
        "os.close(write_fd)\n"
        "outcome = os.read(read_fd, 128)\n"
        "os.close(read_fd)\n"
        "_, status = os.waitpid(pid, 0)\n"
        "assert status == 0 and outcome\n"
        f"with open({str(outcomes)!r}, 'ab') as output:\n"
        "    output.write(outcome)\n"
    )
    chmod = shutil.which("chmod")
    assert chmod is not None
    wrapper = tools / "chmod"
    intervention = (
        f'if [ "$2" = {shlex.quote(str(run))} ]; then '
        f"{shlex.quote(sys.executable)} {shlex.quote(str(helper))} || exit 1; fi\n"
        if attack
        else ""
    )
    wrapper.write_text(f'#!/bin/sh\n{intervention}exec {shlex.quote(chmod)} "$@"\n')
    wrapper.chmod(0o755)
    path = f"{tools}:/usr/bin:/bin"
    sandbox = ModalSandbox(
        environment_dir=root,
        environment_name="stage-custody",
        session_id="stage-custody",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    sandbox._sandbox = MagicMock()
    monkeypatch.setattr(
        sandbox,
        "exec_route",
        AsyncMock(return_value=SimpleNamespace(shell="/bin/sh", pin=path)),
    )

    def mapped(value: str) -> str:
        return value.replace("/opt", str(root / "opt"))

    async def execute(command, **kwargs):
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            f"PATH={shlex.quote(path)}; export PATH; {mapped(command)}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
        finally:
            if process.returncode is None:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=2)
        assert process.returncode is not None
        return ExecResult(
            stdout=stdout.decode(),
            stderr=stderr.decode(),
            return_code=process.returncode,
        )

    async def upload(source, target, **kwargs):
        shutil.copyfile(source, mapped(target))

    monkeypatch.setattr(sandbox, "exec", AsyncMock(side_effect=execute))
    monkeypatch.setattr(sandbox, "upload_file", AsyncMock(side_effect=upload))
    return sandbox, outcomes


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_kind", ["agent_owned", "world_writable", "protected"])
@pytest.mark.parametrize("attack", [True, False])
async def test_upload_cannot_chmod_a_swapped_staging_target(
    image_tree, parent_kind, attack, monkeypatch
):
    """Late extraction refusal at fcc733f466 cannot undo chmod."""
    root = image_tree
    parent = root / "opt" / "benchflow"
    if parent_kind == "agent_owned":
        os.chown(parent, 1000, 1000)
    elif parent_kind == "world_writable":
        parent.chmod(0o777)
    sentinel = root / "sentinel"
    original_mode = stat.S_IMODE(sentinel.stat().st_mode)
    sandbox, outcomes = _sandbox(root, monkeypatch, attack=attack)
    target = root / "uploaded"
    error = None
    try:
        await sandbox.upload_dir(
            root / "source", str(target), admission=admitted(sandbox)
        )
    except RuntimeError as exc:
        error = exc
    # Check the side effect even when the later proof refused the upload.
    assert stat.S_IMODE(sentinel.stat().st_mode) == original_mode
    assert error is None, str(error)
    assert (target / "answer").read_text() == "ordinary upload\n"
    assert sandbox.upload_file.await_count == 1
    assert not (parent / "run").is_symlink()
    if attack:
        attempts = outcomes.read_text().splitlines()
        assert attempts and set(attempts) == {"denied"}


@pytest.mark.asyncio
@pytest.mark.parametrize("link_at", ["parent", "staging_root"])
async def test_existing_stage_link_refuses_before_upload(
    image_tree, link_at, monkeypatch
):
    """The L08 correction preserves no-link refusal and the sentinel's permissions."""
    root = image_tree
    parent = root / "opt" / "benchflow"
    sentinel = root / "sentinel"
    if link_at == "parent":
        parent.rmdir()
        parent.symlink_to(sentinel, target_is_directory=True)
    else:
        (parent / "run").symlink_to(sentinel, target_is_directory=True)
    original_mode = stat.S_IMODE(sentinel.stat().st_mode)
    sandbox, _ = _sandbox(root, monkeypatch, attack=False)
    with pytest.raises(RuntimeError, match="Could not make Modal"):
        await sandbox.upload_dir(
            root / "source", str(root / "uploaded"), admission=admitted(sandbox)
        )
    sandbox.upload_file.assert_not_awaited()
    assert stat.S_IMODE(sentinel.stat().st_mode) == original_mode
    assert not list(sentinel.iterdir())
