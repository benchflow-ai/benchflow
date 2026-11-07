"""Guards PR #1127's fix for shell-only capture failures introduced by PR #1126."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.review.evidence import EvidenceError, capture_workspace
from benchflow.review.evidence_runtime import ensure_evidence_python
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox.protocol import ExecResult


class ShellOnlyTransport:
    """Execute real capture code with Python absent from the initial PATH."""

    def __init__(self, binary_dir: Path):
        self.binary_dir = binary_dir

    async def exec(self, cmd, *, user="root", timeout_sec=30):
        assert user == "root"
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            cmd,
            env={**os.environ, "PATH": str(self.binary_dir)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_sec)
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def download_file(self, src, dst):
        shutil.copyfile(src, dst)


def shell_image(tmp_path, *, has_python=False, package_manager=False):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    for name in ("sh", "rm"):
        (binary_dir / name).symlink_to(shutil.which(name))
    python = binary_dir / "python3"
    marker = tmp_path / "package-manager-called"
    if has_python:
        python.symlink_to(sys.executable)
    if package_manager:
        manager = binary_dir / "apt-get"
        manager.write_text(
            "#!/bin/sh\n"
            f": > {shlex.quote(str(marker))}\n"
            'if [ "$1" = install ]; then\n'
            f"  /bin/ln -s {shlex.quote(sys.executable)} {shlex.quote(str(python))}\n"
            "fi\n"
        )
        manager.chmod(0o755)
    return ShellOnlyTransport(binary_dir), marker


@pytest.mark.asyncio
async def test_shell_only_task_can_capture_after_trusted_setup(tmp_path):
    """Guards PR #1127: reproduce missing Python, then capture real evidence."""
    transport, marker = shell_image(tmp_path, package_manager=True)
    source = tmp_path / "workspace"
    source.mkdir()
    (source / "hello.txt").write_text("Hello, world!\n")
    with pytest.raises(EvidenceError, match=r"python3:.*not found"):
        await capture_workspace(transport, str(source), tmp_path / "before")

    await ensure_evidence_python(transport)
    manifest = await capture_workspace(transport, str(source), tmp_path / "after")

    assert marker.exists()
    assert [entry.path for entry in manifest.entries] == ["hello.txt"]
    assert (tmp_path / "after/workspace/hello.txt").read_text() == "Hello, world!\n"


@pytest.mark.asyncio
async def test_existing_python_does_not_install_packages(tmp_path):
    """Guards PR #1127 against changing already compatible task images."""
    transport, marker = shell_image(tmp_path, has_python=True, package_manager=True)
    await ensure_evidence_python(transport)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_missing_capture_runtime_fails_before_agent_setup(tmp_path, monkeypatch):
    """Guards PR #1127 against spending solver budget before a capture failure."""
    transport, _ = shell_image(tmp_path)
    rollout = Rollout(RolloutConfig(task_path=tmp_path, agent="oracle"))
    rollout._env = transport
    rollout._task = Mock()
    rollout._rollout_dir = tmp_path
    rollout._review_plan = object()
    rollout._planes.setup_sandbox_user = AsyncMock()
    monkeypatch.setattr(
        "benchflow.rollout._resolve_agent_cwd", AsyncMock(return_value="/app")
    )
    with pytest.raises(EvidenceError, match="no supported package manager"):
        await rollout.install_agent()
    rollout._planes.setup_sandbox_user.assert_not_awaited()
