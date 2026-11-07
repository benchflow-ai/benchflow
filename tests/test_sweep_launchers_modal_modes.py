"""Guards FF-sw-launchers against lost upload modes at 363d815b."""

import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.modal_impl import ModalSandbox
from benchflow.task.config import SandboxConfig
from tests._trust_admission import admitted


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [0o600, 0o644, 0o700, 0o751, 0o755])
async def test_upload_preserves_permissions(tmp_path, mode):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    source = tmp_path / "tests"
    source.mkdir()
    helper = source / "check.sh"
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(mode)
    env = ModalSandbox(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="s",
        rollout_paths=None,
        task_env_config=SandboxConfig(),
    )
    env._sandbox = MagicMock()
    env.exec = AsyncMock(return_value=ExecResult(stdout="", stderr="", return_code=0))
    env.exec_route = AsyncMock(
        return_value=SimpleNamespace(shell="/bin/sh", pin="/usr/bin:/bin")
    )
    members: dict[str, int] = {}

    async def upload(source, target, **kwargs):
        # One archive (DAYTONA-GAPS rule 1): its members carry the modes.
        with tarfile.open(source) as tar:
            members.update({m.name: m.mode for m in tar.getmembers()})

    env.upload_file = AsyncMock(side_effect=upload)
    await env.upload_dir(source, "/tests", admission=admitted(env))
    assert members == {"check.sh": mode}
    assert env.upload_file.await_count == 1
