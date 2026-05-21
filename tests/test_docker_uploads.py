from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox.docker import DockerSandbox


@pytest.mark.asyncio
async def test_docker_upload_dir_creates_target_before_compose_cp() -> None:
    """Guards the v0.5 edit-pdf Docker failure where /app did not exist."""
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.exec = AsyncMock()
    sandbox._run_docker_compose_command = AsyncMock()

    await sandbox.upload_dir("local/skills", "/app/skills")

    sandbox.exec.assert_awaited_once_with("mkdir -p /app/skills", user="root")
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

    await sandbox.upload_file(Path("instruction.md"), "/app/instruction.md")

    sandbox.exec.assert_awaited_once_with("mkdir -p /app", user="root")
    sandbox._run_docker_compose_command.assert_awaited_once_with(
        ["cp", "instruction.md", "main:/app/instruction.md"],
        check=True,
    )
