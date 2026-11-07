"""Credential destinations never receive a privileged upload or recursive chown."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.agents.credentials import upload_credential


def environment():
    return SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout="", stderr="")
        ),
        upload_file=AsyncMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        ".config/gcloud/adc.json",
        ".codex/auth.json",
        ".gemini/settings.json",
        ".aws/credentials",
        ".claude/login.json",
    ],
)
async def test_credentials_upload_only_into_private_staging(path):
    env = environment()
    dest = f"/tmp/agent-home/{path}"
    await upload_credential(env, dest, "dummy-marker", owner="agent")
    uploaded = env.upload_file.await_args.args[1]
    assert not uploaded.startswith("/tmp/agent-home/")
    for call in env.exec.await_args_list:
        if (
            dest in call.args[0]
            or f"/tmp/agent-home/{path.rsplit('/', 1)[0]}" in call.args[0]
        ):
            assert call.kwargs.get("user") == "agent"
    assert not any("chown -R" in call.args[0] for call in env.exec.await_args_list)


@pytest.mark.asyncio
async def test_credential_install_failure_is_not_accepted():
    env = environment()
    env.exec.return_value.return_code = 1
    with pytest.raises(RuntimeError, match="credential"):
        await upload_credential(
            env, "/tmp/agent-home/.codex/auth.json", "dummy", owner="agent"
        )
    env.upload_file.assert_not_awaited()
