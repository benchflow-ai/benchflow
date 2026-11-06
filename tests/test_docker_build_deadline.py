"""Guards the fix at integration commit 363d815b."""

import time
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import docker as d
from benchflow.task.config import SandboxConfig
from tests.test_docker_build_identity import sandbox


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_retry_attempts_share_one_build_budget(tmp_path, monkeypatch, backend):
    """Guards 363d815b: a retry cannot restart the configured deadline."""
    env = sandbox(tmp_path, backend, task_env_config=SandboxConfig(build_timeout_sec=5))
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    seen = []

    async def command(args, check=True, timeout_sec=None):
        seen.append(timeout_sec)
        now[0] += 1
        if len(seen) < 2:
            raise RuntimeError("connection timed out")

    async def sleep(delay):
        now[0] += delay

    monkeypatch.setattr(d.asyncio, "sleep", sleep)
    env._run_docker_compose_command = command
    await env._run_docker_compose_build()
    assert seen == [5, 2]


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_backoff_cannot_outlive_build_budget(tmp_path, monkeypatch, backend):
    """Guards 363d815b: do not sleep/retry beyond the build deadline."""
    env = sandbox(
        tmp_path, backend, task_env_config=SandboxConfig(build_timeout_sec=0.01)
    )
    env._run_docker_compose_command = AsyncMock(
        side_effect=RuntimeError("connection timed out")
    )
    sleep = AsyncMock()
    monkeypatch.setattr(d.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match=r"(?i)build.*timed out"):
        await env._run_docker_compose_build()
    sleep.assert_not_awaited()


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_up_cannot_start_a_second_build(tmp_path, backend):
    """Guards 363d815b: only the explicitly budgeted build may build images."""
    env = sandbox(tmp_path, backend)
    env._run_docker_compose_command = AsyncMock()
    await env._run_docker_compose_up()
    command = env._run_docker_compose_command.await_args.args[0]
    assert "--no-build" in command


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_prebuilt_main_builds_sidecars_before_up_with_the_same_deadline(
    tmp_path, backend
):
    """Guards 363d815b's sibling: sidecar builds also use the explicit budget."""
    env = sandbox(
        tmp_path, backend, task_env_config=SandboxConfig(docker_image="trusted:task")
    )
    (env.environment_dir / "docker-compose.yaml").write_text(
        "services:\n  sidecar:\n    build: .\n"
    )
    built = []

    async def build():
        built.append(True)

    async def up():
        assert built == [True], "sidecar build was left to an unbounded up"

    env._run_docker_compose_build = build
    env._run_docker_compose_up = up
    env._prepare_compose_builds = AsyncMock()
    env._absent_task_image = AsyncMock(return_value=None)
    env._note_task_image_pull = AsyncMock()
    env._choose_subnets = AsyncMock()
    env._name_trial_volumes = AsyncMock()
    env._learn_main_container = AsyncMock()
    env._prepare_log_dirs_after_up = AsyncMock()
    await d.DockerSandbox.start(env, False)
