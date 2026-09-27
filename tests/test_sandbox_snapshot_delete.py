"""Docker and Daytona-direct sandboxes delete the snapshots a branch created.

Regression test: no sandbox could delete a container snapshot, so a Daytona
provider snapshot outlived its run and a Docker ``bf-snap-*`` image survived
unless ``compose down --rmi all`` happened to take it. A snapshot that is
still in use (the restored parent runs from it) is deleted when the sandbox
stops instead.

Unit tests with a mocked Docker CLI and a fake Daytona client.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import daytona as daytona_mod
from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import (
    ExecResult,
    SandboxImage,
    SandboxSnapshotNotSupported,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

IN_USE = (
    "Error response from daemon: conflict: unable to remove repository "
    'reference "bf-snap-x" (must force) - container 1234 is using its '
    "referenced image"
)


@pytest.fixture
def docker_sandbox(tmp_path):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name="snapshot-delete",
        session_id="bf-snapshot-delete",
        rollout_paths=paths,
        task_env_config=SandboxConfig(),
    )


def _cli(results: dict[str, list[ExecResult]]):
    calls: list[list[str]] = []

    async def cli(args, check=True):
        calls.append(list(args))
        queue = results.get(" ".join(args[:2]), [])
        return queue.pop(0) if queue else ExecResult(0, "", "")

    return cli, calls


async def test_docker_deletes_an_unused_snapshot_image(docker_sandbox):
    cli, calls = _cli({})
    docker_sandbox._docker_cli = cli
    image = SandboxImage(provider="docker", ref="bf-snap-x")
    assert await docker_sandbox.delete_snapshot(image) is True
    assert calls == [["image", "rm", "bf-snap-x"]]


async def test_docker_treats_a_missing_image_as_deleted(docker_sandbox):
    cli, _ = _cli({"image rm": [ExecResult(1, "", "Error: No such image: bf-snap-x")]})
    docker_sandbox._docker_cli = cli
    image = SandboxImage(provider="docker", ref="bf-snap-x")
    assert await docker_sandbox.delete_snapshot(image) is True


async def test_docker_defers_an_image_in_use_until_stop(docker_sandbox):
    cli, calls = _cli({"image rm": [ExecResult(1, "", IN_USE)]})
    docker_sandbox._docker_cli = cli
    docker_sandbox._chown_to_host_user = AsyncMock()
    docker_sandbox._run_docker_compose_command = AsyncMock(
        return_value=ExecResult(0, "", "")
    )
    image = SandboxImage(provider="docker", ref="bf-snap-x")

    assert await docker_sandbox.delete_snapshot(image) is False
    await docker_sandbox.stop(delete=True)

    # compose down ran first, then the deferred image was removed.
    down = docker_sandbox._run_docker_compose_command.call_args_list[0].args[0]
    assert down[0] == "down"
    assert calls == [["image", "rm", "bf-snap-x"], ["image", "rm", "bf-snap-x"]]
    calls.clear()
    await docker_sandbox.stop(delete=True)
    assert calls == []  # nothing left to delete


async def test_docker_rejects_another_providers_snapshot(docker_sandbox):
    with pytest.raises(SandboxSnapshotNotSupported):
        await docker_sandbox.delete_snapshot(
            SandboxImage(provider="daytona", ref="bf-snap-x")
        )


class _FakeSnapshots:
    def __init__(self, names, *, fail_deletes=0):
        self.names = set(names)
        self.fail_deletes = fail_deletes
        self.deleted: list[str] = []

    async def get(self, name):
        if name not in self.names:
            raise daytona_mod.DaytonaNotFoundError(f"Snapshot {name} not found")
        return SimpleNamespace(name=name, id=f"id-{name}")

    async def delete(self, snapshot):
        if self.fail_deletes:
            self.fail_deletes -= 1
            raise RuntimeError("snapshot is in use by a sandbox")
        self.names.discard(snapshot.name)
        self.deleted.append(snapshot.name)


@pytest.fixture
def daytona_direct(monkeypatch):
    pytest.importorskip("daytona")  # sandbox-daytona optional dependency
    daytona_mod._load_daytona_sdk()
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.logger = daytona_mod.logger
    sandbox._sandbox = None
    sandbox._client_manager = None
    strategy = _DaytonaDirect(sandbox)
    sandbox._strategy = strategy
    snapshots = _FakeSnapshots({"bf-snap-x"})
    client = SimpleNamespace(snapshot=snapshots)
    manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    monkeypatch.setattr(
        daytona_mod.DaytonaClientManager,
        "get_instance",
        AsyncMock(return_value=manager),
    )
    monkeypatch.setattr(
        "benchflow.sandbox.daytona_strategies._DELETE_RETRY_DELAYS", (0, 0)
    )
    return sandbox, snapshots


async def test_daytona_deletes_its_provider_snapshot(daytona_direct):
    sandbox, snapshots = daytona_direct
    image = SandboxImage(provider="daytona", ref="bf-snap-x")
    assert await sandbox.delete_snapshot(image) is True
    assert snapshots.deleted == ["bf-snap-x"]
    # Deleting again is a no-op, not an error.
    assert await sandbox.delete_snapshot(image) is True


async def test_daytona_defers_a_snapshot_in_use_until_stop(daytona_direct):
    sandbox, snapshots = daytona_direct
    snapshots.fail_deletes = 1
    image = SandboxImage(provider="daytona", ref="bf-snap-x")
    assert await sandbox.delete_snapshot(image) is False
    assert snapshots.deleted == []
    await sandbox.stop(delete=True)
    assert snapshots.deleted == ["bf-snap-x"]
