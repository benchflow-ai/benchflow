import asyncio
import base64
import io
import os
import tarfile
from types import SimpleNamespace

import pytest

from benchflow._dotenv import load_dotenv_env
from benchflow.rollout import _run_environment_setup_commands


@pytest.mark.asyncio
async def test_environment_setup_host_lock_serializes_commands(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("BENCHFLOW_SETUP_LOCK_DIR", str(tmp_path))

    active = 0
    max_active = 0

    class Env:
        async def exec(self, *args, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    command = SimpleNamespace(
        command="echo setup",
        cwd="/workspace",
        env={},
        timeout_sec=30.0,
        user=None,
        service="main",
        host_lock="shared-notion-oauth",
    )
    task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(setup_commands=[command]))
    )

    await asyncio.gather(
        _run_environment_setup_commands(Env(), task),
        _run_environment_setup_commands(Env(), task),
    )

    assert max_active == 1


@pytest.mark.asyncio
async def test_environment_setup_command_captures_directory_to_env(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("CAPTURED_AUTH_B64", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text('OTHER=1\nCAPTURED_AUTH_B64="old"\n')
    monkeypatch.setenv("CAPTURE_DOTENV_PATH", str(dotenv_path))

    class Env:
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def download_dir(self, source_dir, target_dir, service="main"):
            assert source_dir == "/workspace/configs/.mcp-auth"
            assert service == "main"
            target_dir.mkdir(parents=True)
            (target_dir / "oauth.json").write_text('{"refresh_token":"rotated"}')

    command = SimpleNamespace(
        command="echo setup",
        cwd="/workspace",
        env={},
        timeout_sec=30.0,
        user=None,
        service="main",
        host_lock=None,
        capture_dir="/workspace/configs/.mcp-auth",
        capture_dir_b64_env="CAPTURED_AUTH_B64",
        capture_dir_b64_env_file_var="CAPTURE_DOTENV_PATH",
    )
    task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(setup_commands=[command]))
    )

    await _run_environment_setup_commands(Env(), task)

    captured = os.environ["CAPTURED_AUTH_B64"]
    assert load_dotenv_env(dotenv_path)["CAPTURED_AUTH_B64"] == captured

    encoded = base64.b64decode(captured)
    with tarfile.open(fileobj=io.BytesIO(encoded), mode="r:gz") as tar:
        member = tar.extractfile("oauth.json")
        assert member is not None
        assert member.read().decode() == '{"refresh_token":"rotated"}'


@pytest.mark.asyncio
async def test_environment_setup_host_lock_covers_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_SETUP_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.delenv("LOCKED_CAPTURED_AUTH_B64", raising=False)

    capture_active = False
    exec_started_during_capture = False

    class Env:
        async def exec(self, *args, **kwargs):
            nonlocal exec_started_during_capture
            if capture_active:
                exec_started_during_capture = True
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        async def download_dir(self, source_dir, target_dir, service="main"):
            nonlocal capture_active
            assert source_dir == "/workspace/configs/.mcp-auth"
            capture_active = True
            target_dir.mkdir(parents=True)
            (target_dir / "oauth.json").write_text('{"refresh_token":"rotated"}')
            await asyncio.sleep(0.01)
            capture_active = False

    command = SimpleNamespace(
        command="echo setup",
        cwd="/workspace",
        env={},
        timeout_sec=30.0,
        user=None,
        service="main",
        host_lock="shared-notion-oauth",
        capture_dir="/workspace/configs/.mcp-auth",
        capture_dir_b64_env="LOCKED_CAPTURED_AUTH_B64",
        capture_dir_b64_env_file_var=None,
    )
    task = SimpleNamespace(
        config=SimpleNamespace(environment=SimpleNamespace(setup_commands=[command]))
    )

    await asyncio.gather(
        _run_environment_setup_commands(Env(), task),
        _run_environment_setup_commands(Env(), task),
    )

    assert not exec_started_during_capture
