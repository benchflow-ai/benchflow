from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox


def test_docker_logs_mount_fast_path_enabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("BENCHFLOW_DOCKER_LOGS_HOST_MOUNTED", raising=False)
    sandbox = DockerSandbox.__new__(DockerSandbox)

    assert sandbox.is_mounted is True


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off"])
def test_docker_logs_mount_fast_path_can_be_disabled(monkeypatch, value: str) -> None:
    monkeypatch.setenv("BENCHFLOW_DOCKER_LOGS_HOST_MOUNTED", value)
    sandbox = DockerSandbox.__new__(DockerSandbox)

    assert sandbox.is_mounted is False


@pytest.mark.asyncio
async def test_prebuilt_stop_does_not_remove_images() -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox._keep_containers = False
    sandbox._use_prebuilt = True
    sandbox._chown_to_host_user = AsyncMock()
    sandbox._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(stdout="", stderr=None, return_code=0)
    )

    await sandbox.stop(delete=True)

    sandbox._run_docker_compose_command.assert_awaited_once_with(
        ["down", "--volumes", "--remove-orphans"]
    )


@pytest.mark.asyncio
async def test_docker_upload_dir_creates_target_before_compose_cp() -> None:
    """Guards the v0.5 edit-pdf Docker failure where /app did not exist."""
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.exec = AsyncMock()
    sandbox._run_docker_compose_command = AsyncMock()

    await sandbox.upload_dir("local/skills", "/app/skills")

    # upload_dir threads the compose `service` selector through to mkdir (#248).
    sandbox.exec.assert_awaited_once_with(
        "mkdir -p /app/skills", user="root", service="main"
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

    await sandbox.upload_file(Path("instruction.md"), "/app/instruction.md")

    sandbox.exec.assert_awaited_once_with("mkdir -p /app", user="root")
    sandbox._run_docker_compose_command.assert_awaited_once_with(
        ["cp", "instruction.md", "main:/app/instruction.md"],
        check=True,
    )
