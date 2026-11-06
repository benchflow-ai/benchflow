"""Guards FF-sw-launchers against the runtime identity regression at 363d815b."""

from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import agentcore_provisioning as provisioning
from benchflow.sandbox.agentcore import AgentCoreSandbox
from benchflow.task.config import SandboxConfig


@pytest.mark.asyncio
async def test_rebuild_does_not_rebind_another_evaluators_runtime(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("BENCHFLOW_AGENTCORE_ROLE_ARN", "arn:aws:iam::1:role/rt")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
    provisioning.reset_cache()
    names = []

    async def create(name, image):
        names.append(name)
        return name, name

    def sandbox():
        env = AgentCoreSandbox(
            environment_dir=tmp_path,
            environment_name="task",
            session_id="s",
            rollout_paths=None,
            task_env_config=SandboxConfig(),
        )
        env._create_or_adopt_runtime = AsyncMock(side_effect=create)
        return env

    a, b = sandbox(), sandbox()
    first = await a._ensure_runtime("repo@sha256:111")
    second = await b._ensure_runtime("repo@sha256:222")
    assert first != second
    assert await a._ensure_runtime("repo@sha256:111") == first
    assert len(names) == 2
    provisioning.reset_cache()
