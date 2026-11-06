"""Guards FF-sw-daytona safe directory-upload modes at 363d815b."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._transport import STAGE_ROOT
from benchflow.sandbox.daytona import (
    DaytonaSandbox,
    _DaytonaDinD,
    _DaytonaDirect,
    _load_daytona_sdk,
)
from tests._archive_fake import unpack_archive


@pytest.mark.parametrize("tar", [True, False], ids=["archive", "no-tar"])
@pytest.mark.parametrize("backend", ["direct", "dind"])
@pytest.mark.parametrize("mode", [0o755, 0o700, 0o711, 0o4755, 0o777, 0o644, 0o666])
async def test_upload_preserves_safe_executable_intent(tmp_path, backend, mode, tar):
    """Guard 363d815b: verifier/oracle helpers survive both upload paths, as
    one archive (DAYTONA-GAPS rule 1) and file by file where the sandbox has
    no tar."""
    _load_daytona_sdk()
    source = tmp_path / "source"
    source.mkdir()
    helper = source / "helper ' name"
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(mode)
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)

    def path(remote: str) -> Path:
        # Root's staging folder under vm/; every other folder as it is.
        if remote.startswith(STAGE_ROOT + "/"):
            return tmp_path / "vm" / remote.lstrip("/")
        return Path(remote)

    async def upload_file(source_path, destination):
        target = path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source_path).read_bytes())

    async def upload(*, files):
        for entry in files:
            target = Path(entry.destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(entry.source).read_bytes())
            target.chmod(0o644)

    async def execute(command, **kwargs):
        if "tar -x -f" in command:
            if not tar:
                return ExecResult(return_code=127, stdout="", stderr="")
            assert unpack_archive(command, path)
            return ExecResult(return_code=0, stdout="", stderr="")
        # Reproduce mode restoration as the sandbox uid; ownership is already
        # that uid in this mount-free transport fixture.
        command = command.replace("chown 0:0", "true")
        result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
        return ExecResult(
            return_code=result.returncode, stdout=result.stdout, stderr=result.stderr
        )

    sandbox._sandbox = SimpleNamespace(
        fs=SimpleNamespace(upload_file=upload_file, upload_files=upload)
    )
    sandbox._sandbox_exec = execute
    target = tmp_path / "target"
    if backend == "direct":
        strategy = _DaytonaDirect(sandbox)
    else:
        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        strategy._env = sandbox
        strategy.exec = AsyncMock(
            return_value=ExecResult(return_code=0, stdout="", stderr="")
        )
        strategy._vm_exec = execute

        async def compose(args, **kwargs):
            shutil.copytree(args[1][:-2], target, dirs_exist_ok=True)
            return ExecResult(return_code=0, stdout="", stderr="")

        strategy._compose_exec = compose
    await strategy.upload_dir(source, str(target))
    expected = 0o755 if mode & 0o111 else 0o644
    assert (target / helper.name).stat().st_mode & 0o7777 == expected


async def test_unpack_failure_aborts_upload(tmp_path):
    """DAYTONA-GAPS rule 1: unpacking the archive is a required transfer step."""
    _load_daytona_sdk()
    (tmp_path / "helper").write_text("dummy")
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(fs=SimpleNamespace(upload_file=AsyncMock()))
    sandbox._sandbox_exec = AsyncMock(
        return_value=ExecResult(return_code=1, stdout="", stderr="dummy")
    )
    with pytest.raises(RuntimeError, match="unpack"):
        await sandbox._sdk_upload_dir(tmp_path, "/tmp/dummy-upload")


async def test_mode_failure_aborts_upload(tmp_path):
    """Guard 363d815b: permission restoration is a required transfer step
    (the file-by-file path, where the sandbox has no tar)."""
    _load_daytona_sdk()
    (tmp_path / "helper").write_text("dummy")
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox._sandbox = SimpleNamespace(
        fs=SimpleNamespace(upload_file=AsyncMock(), upload_files=AsyncMock())
    )
    sandbox._sandbox_exec = AsyncMock(
        side_effect=[
            ExecResult(return_code=127, stdout="", stderr=""),
            ExecResult(return_code=1, stdout="", stderr="dummy"),
        ]
    )
    with pytest.raises(RuntimeError, match="permissions"):
        await sandbox._sdk_upload_dir(tmp_path, "/tmp/dummy-upload")
