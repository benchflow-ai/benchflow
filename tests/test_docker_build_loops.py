"""Guards the fix at integration commit 363d815b."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.sandbox import docker as d
from tests.test_docker_build_identity import sandbox


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
def test_repeated_contended_sdk_batches_have_loop_local_locks(
    tmp_path, monkeypatch, backend
):
    """Guards 363d815b: repeated asyncio.run batches may contend each time."""
    monkeypatch.setattr(d.DockerSandbox, "_image_build_locks", {})
    for batch in range(2):

        async def run_batch(batch=batch):
            entered = asyncio.Event()

            async def build():
                entered.set()
                await asyncio.sleep(0.02)

            envs = [sandbox(tmp_path / f"{batch}-{i}", backend) for i in range(2)]
            for env in envs:
                env._run_docker_compose_build = build
                env._start_from_layers = AsyncMock()
                env._run_pre_compose_hook = AsyncMock()
                env._docker_subnet_key = Mock(
                    return_value=SimpleNamespace(
                        identify=AsyncMock(
                            return_value=SimpleNamespace(server_id="test-daemon")
                        )
                    )
                )
                env._choose_subnets = AsyncMock()
                env._name_trial_volumes = AsyncMock()
                env._run_docker_compose_up = AsyncMock()
                env._learn_main_container = AsyncMock()
                env._prepare_log_dirs_after_up = AsyncMock()
            await asyncio.gather(*(d.DockerSandbox.start(env, False) for env in envs))

        asyncio.run(run_batch())


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
def test_configured_build_admission_is_loop_local(tmp_path, monkeypatch, backend):
    """Guards 363d815b's sibling: the startup semaphore also binds on contention."""
    monkeypatch.setattr(d.DockerSandbox, "_build_concurrency", None, raising=False)
    monkeypatch.setattr(d.DockerSandbox, "_build_semaphores", {}, raising=False)
    monkeypatch.setattr(d.DockerSandbox, "_image_build_locks", {})
    d.DockerSandbox.set_build_concurrency(1)
    for batch in range(2):

        async def run_batch(batch=batch):
            async def build():
                await asyncio.sleep(0.01)

            envs = [sandbox(tmp_path / f"admit-{batch}-{i}", backend) for i in range(2)]
            for i, env in enumerate(envs):
                env.environment_name = f"different-task-{i}"
                env._run_docker_compose_build = build
                env._start_from_layers = AsyncMock()
                env._run_pre_compose_hook = AsyncMock()
                env._docker_subnet_key = Mock(
                    return_value=SimpleNamespace(
                        identify=AsyncMock(
                            return_value=SimpleNamespace(server_id="test-daemon")
                        )
                    )
                )
                env._choose_subnets = AsyncMock()
                env._name_trial_volumes = AsyncMock()
                env._run_docker_compose_up = AsyncMock()
                env._learn_main_container = AsyncMock()
                env._prepare_log_dirs_after_up = AsyncMock()
            await asyncio.gather(*(d.DockerSandbox.start(env, False) for env in envs))

        asyncio.run(run_batch())
