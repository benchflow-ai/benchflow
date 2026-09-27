"""A Docker snapshot image must not claim the compose project of its container.

``docker commit`` copies the container's labels onto the image, including
``com.docker.compose.project``. Docker Compose then treats the
snapshot as one of the project's images, and the sandbox teardown
``docker compose down --rmi all`` deletes it with the project. A kept
``--checkpoints`` snapshot was gone before ``--retry-from-checkpoint`` could
start from it ("Unable to find image 'bf-snap-…'"). The deterministic integration
tier exercises this path.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


@pytest.fixture
def docker_sandbox(tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="labels",
        session_id="bf-labels",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )


async def test_snapshot_image_drops_the_compose_project_labels(
    docker_sandbox, monkeypatch
):
    calls: list[tuple[str, ...]] = []
    docker_sandbox._main_container_id = AsyncMock(return_value="c0ffee")

    async def no_credentials(ops):
        return []

    monkeypatch.setattr("benchflow.sandbox.docker.scrub_credentials", no_credentials)

    class Proc:
        returncode = 0

        async def communicate(self):
            return b"sha256:" + b"0" * 64, b""

    async def fake_exec(*args, **kwargs):
        calls.append(args)
        return Proc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
    image = await docker_sandbox.snapshot()

    (args,) = calls
    assert args[:2] == ("docker", "commit")
    assert args[-2:] == ("c0ffee", image.ref)
    changes = [args[i + 1] for i, a in enumerate(args) if a == "--change"]
    assert "LABEL com.docker.compose.project=" in changes
    assert "LABEL com.docker.compose.service=" in changes
