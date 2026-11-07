from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox.docker import DockerSandbox
from tests._trust_admission import admitted


@pytest.mark.asyncio
async def test_docker_upload_dir_creates_target_before_compose_cp() -> None:
    """Guards the v0.5 edit-pdf Docker failure where /app did not exist."""
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.exec = AsyncMock(
        return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
    )
    sandbox._run_docker_compose_command = AsyncMock()

    await sandbox.upload_dir("local/skills", "/app/skills", admission=admitted(sandbox))

    # upload_dir threads the compose `service` selector through to mkdir (#248);
    # root's mkdir starts /bin/sh by path, never one on the image's PATH (J1-R3-A-1).
    sandbox.exec.assert_awaited_once_with(
        "mkdir -p /app/skills", user="root", service="main", system_shell=True
    )
    sandbox._run_docker_compose_command.assert_awaited_once_with(
        ["cp", "local/skills/.", "main:/app/skills"],
        check=True,
    )


@pytest.mark.asyncio
async def test_docker_upload_file_creates_parent_before_compose_cp() -> None:
    """Guards uploads into task images whose Dockerfile uses a non-/app WORKDIR."""
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.exec = AsyncMock()
    sandbox._run_docker_compose_command = AsyncMock()

    await sandbox.upload_file(
        Path("instruction.md"), "/app/instruction.md", admission=admitted(sandbox)
    )

    sandbox.exec.assert_awaited_once_with(
        "mkdir -p /app", user="root", system_shell=True
    )
    sandbox._run_docker_compose_command.assert_awaited_once_with(
        ["cp", "instruction.md", "main:/app/instruction.md"],
        check=True,
    )


@pytest.mark.asyncio
async def test_docker_upload_file_applies_requested_mode() -> None:
    """Guards PR #942: Docker must honor the upload-file mode protocol."""

    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.exec = AsyncMock(
        return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
    )
    sandbox._run_docker_compose_command = AsyncMock()

    await sandbox.upload_file(
        Path("secret.json"), "/app/secret.json", mode="600", admission=admitted(sandbox)
    )

    assert sandbox.exec.await_args_list[-1].args[0] == "chmod 600 /app/secret.json"
    assert sandbox.exec.await_args_list[-1].kwargs == {
        "user": "root",
        "timeout_sec": 30,
    }
