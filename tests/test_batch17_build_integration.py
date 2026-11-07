"""Guards the batch 17 merge of c51c88dc and 7ca37968: cached build budgets."""

import time
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import docker as d
from benchflow.sandbox._compose_build import ComposeBuild
from benchflow.task.config import SandboxConfig
from tests.test_docker_build_identity import sandbox


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_certified_build_retries_keep_frozen_inputs_and_one_deadline(
    tmp_path, monkeypatch, backend
):
    """Guards c51c88dc + 7ca37968: cache retries cannot reset a task's budget."""
    env = sandbox(tmp_path, backend, task_env_config=SandboxConfig(build_timeout_sec=5))
    invocation = ComposeBuild(
        argv=("/tmp/certified-compose", "build"),
        environment=(("TASK_BUILD_INPUT", "frozen"),),
        cwd=str(tmp_path),
        uncertainty=None,
    )
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    seen = []

    async def process(argv, **kwargs):
        assert argv == list(invocation.argv)
        assert kwargs["environment"] == dict(invocation.environment)
        assert kwargs["cwd"] == invocation.cwd
        seen.append(kwargs["timeout_sec"])
        now[0] += 1
        if len(seen) == 1:
            raise RuntimeError("connection timed out")

    async def sleep(delay):
        now[0] += delay

    monkeypatch.setattr(d.asyncio, "sleep", sleep)
    env._run_docker_process = process
    env._run_docker_compose_command = AsyncMock()
    await env._run_docker_compose_build(invocation=invocation)
    assert seen == [5, 2]
    env._run_docker_compose_command.assert_not_awaited()


@pytest.mark.parametrize("backend", ["docker", "remote-docker"])
async def test_certified_build_backoff_cannot_exceed_the_remaining_budget(
    tmp_path, monkeypatch, backend
):
    """Guards c51c88dc + 7ca37968: certified builds obey the ordinary backoff cap."""
    env = sandbox(
        tmp_path, backend, task_env_config=SandboxConfig(build_timeout_sec=0.01)
    )
    invocation = ComposeBuild(
        argv=("/tmp/certified-compose", "build"),
        environment=(),
        cwd=str(tmp_path),
        uncertainty=None,
    )
    env._run_docker_process = AsyncMock(
        side_effect=RuntimeError("connection timed out")
    )
    sleep = AsyncMock()
    monkeypatch.setattr(d.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match=r"(?i)build.*timed out"):
        await env._run_docker_compose_build(invocation=invocation)
    sleep.assert_not_awaited()
