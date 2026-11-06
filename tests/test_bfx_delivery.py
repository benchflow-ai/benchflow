"""The host-side delivery check (TRUSTED-RUNTIME v3,
Done-when 4): a configuration test of our own launch, run entirely from the
host.

For a sandbox BenchFlow started, ``docker inspect`` of ``main`` must show:
- exactly one mount at ``/.bfx`` and none beneath it, with ``RW: false``;
- its source equal to the verified file or volume: the host folder that
  holds only the pinned binary (bind), or this start's helper volume, which
  carries the pin's label (volume);
- ``CAP_SYS_ADMIN`` absent from ``HostConfig.CapAdd`` (and ``ALL`` too), and
  the container not privileged;
- no ``seccomp=unconfined`` in ``HostConfig.SecurityOpt``.

Then ``docker cp`` of ``/.bfx/bfx`` out of the container must equal the pin.

No step runs inside the container. The forms checked are the bind on the
local daemon, the volume on the local daemon (by choice), and the volume
on a second, remote daemon through ``remote-docker``
(``tests/_bfx_daemons.py``; skipped without one).
"""

from __future__ import annotations

import json
import platform
import re
import uuid
from pathlib import Path

import pytest

from benchflow._bfx import delivery, pins
from benchflow._bfx.locate import SANDBOX_DIR, SANDBOX_PATH, BfxUnavailable
from benchflow.sandbox._docker_sweep import release_project
from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests._bfx_daemons import (
    Daemon,
    local_daemon,
    own_volume_name,
    point_remote_docker_at,
    remote_daemon,
)

# A main image BenchFlow's launch accepts without a build.
MAIN_IMAGE = "ubuntu:24.04"


def _arch_for(daemon: Daemon) -> str:
    if daemon.env is None:
        return platform.machine()
    return daemon.ok(["info", "--format", "{{.Architecture}}"]).decode().strip()


def _require_pin(arch: str) -> None:
    from benchflow._bfx.locate import verified_binary

    try:
        verified_binary(arch)
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx for {arch}: {exc}")


def _sandbox(tmp_path: Path, cls=DockerSandbox) -> DockerSandbox:
    environment = tmp_path / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(f"FROM {MAIN_IMAGE}\n")
    (environment / "docker-compose.yaml").write_text(
        json.dumps({"services": {"main": {}}})
    )
    paths = RolloutPaths(rollout_dir=tmp_path / "rollout")
    paths.mkdir()
    name = f"bf-test-bfxdeliv-{uuid.uuid4().hex[:8]}"
    return cls(
        environment_dir=environment,
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(docker_image=MAIN_IMAGE, cpus=1, memory_mb=256),
    )


_SYS_ADMIN = {"SYS_ADMIN", "CAP_SYS_ADMIN", "ALL", "CAP_ALL"}
_UNCONFINED = re.compile(r"^seccomp[=:]unconfined$", re.IGNORECASE)


def check_launch(daemon: Daemon, container: str, *, mode: str, source: str, arch: str):
    """The delivery check, from the host, on a started *container*."""
    info = daemon.inspect("container", container)
    assert info["State"]["Running"] is True
    mounts = info["Mounts"]
    at = [m for m in mounts if m["Destination"] == SANDBOX_DIR]
    beneath = [m for m in mounts if m["Destination"].startswith(SANDBOX_DIR + "/")]
    assert len(at) == 1, mounts
    assert beneath == [], beneath
    [mount] = at
    assert mount["RW"] is False, mount
    if mode == delivery.BIND:
        assert mount["Type"] == "bind", mount
        assert mount["Source"] == source, mount
        folder = Path(source)
        assert [p.name for p in folder.iterdir()] == ["bfx"]
        assert delivery.sha256_of(folder / "bfx") == pins.SHA256[arch]
    else:
        assert mount["Type"] == "volume", mount
        assert mount["Name"] == source, mount
        volume = daemon.inspect("volume", source)
        assert volume["Labels"].get(delivery.VOLUME_LABEL) == pins.SHA256[arch]
        assert volume["Driver"] == "local" and not volume.get("Options"), volume
    host = info["HostConfig"]
    cap_add = {c.upper() for c in host.get("CapAdd") or []}
    assert not cap_add & _SYS_ADMIN, cap_add
    assert host.get("Privileged") is False
    for opt in host.get("SecurityOpt") or []:
        assert not _UNCONFINED.match(opt.replace(" ", "")), opt
    # The binary itself, copied out by the daemon (no exec in the sandbox).
    assert daemon.file_sha256(container, SANDBOX_PATH) == pins.SHA256[arch]


async def _started(sandbox: DockerSandbox) -> str:
    await sandbox.start(force_build=False)
    container = await sandbox._main_container_id()
    assert container
    return container


async def _stop(sandbox: DockerSandbox) -> None:
    try:
        await sandbox.stop(delete=True)
    finally:
        release_project(sandbox.compose_project_name)


@pytest.fixture
def local():
    daemon = local_daemon()
    yield daemon
    daemon.cleanup()


@pytest.fixture
def remote():
    daemon = remote_daemon()
    yield daemon
    daemon.cleanup()


@pytest.mark.asyncio
async def test_local_bind_delivery_is_a_read_only_bind_of_the_verified_folder(
    tmp_path, monkeypatch, local
):
    arch = _arch_for(local)
    _require_pin(arch)
    monkeypatch.setenv(delivery.DELIVERY_ENV, "bind")
    sandbox = _sandbox(tmp_path)
    try:
        container = await _started(sandbox)
        assert sandbox.bfx_delivery == delivery.BIND
        check_launch(
            local,
            container,
            mode=delivery.BIND,
            source=str(delivery.host_dir(arch)),
            arch=arch,
        )
    finally:
        await _stop(sandbox)


@pytest.mark.asyncio
async def test_local_volume_delivery_is_the_read_only_pinned_volume(
    tmp_path, monkeypatch, local
):
    arch = _arch_for(local)
    _require_pin(arch)
    monkeypatch.setenv(delivery.DELIVERY_ENV, "volume")
    volume = own_volume_name(monkeypatch, local)
    sandbox = _sandbox(tmp_path)
    try:
        container = await _started(sandbox)
        assert sandbox.bfx_delivery == delivery.VOLUME
        check_launch(local, container, mode=delivery.VOLUME, source=volume, arch=arch)
    finally:
        await _stop(sandbox)


@pytest.mark.asyncio
async def test_remote_docker_volume_delivery_is_the_read_only_pinned_volume(
    tmp_path, monkeypatch, remote
):
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    arch = _arch_for(remote)
    _require_pin(arch)
    point_remote_docker_at(monkeypatch)
    monkeypatch.delenv(delivery.DELIVERY_ENV, raising=False)
    volume = own_volume_name(monkeypatch, remote)
    remote.ok(["pull", "-q", MAIN_IMAGE], timeout=600)
    sandbox = _sandbox(tmp_path, RemoteDockerSandbox)
    try:
        container = await _started(sandbox)
        # auto on a remote daemon is the volume.
        assert sandbox.bfx_delivery == delivery.VOLUME
        check_launch(remote, container, mode=delivery.VOLUME, source=volume, arch=arch)
    finally:
        await _stop(sandbox)


# M-1: a task's own mount at or beneath /.bfx never starts (the Claude
# review's reproducer: the task binds its own program over /.bfx/bfx).
_BENEATH = {
    "bind-over-bfx": {
        "volumes": [
            {
                "type": "bind",
                "source": "./tool",
                "target": "/.bfx/bfx",
                "read_only": True,
            }
        ]
    },
    "tmpfs-beneath": {"tmpfs": ["/.bfx/sub"]},
    "volume-beneath": {
        "volumes": [{"type": "volume", "source": "data", "target": "/.bfx/sub"}]
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("how", sorted(_BENEATH))
async def test_a_task_mount_beneath_bfx_is_refused_and_nothing_starts(
    tmp_path, monkeypatch, local, how
):
    from benchflow.task.env import TaskEnvNotAllowedError

    arch = _arch_for(local)
    _require_pin(arch)
    monkeypatch.setenv(delivery.DELIVERY_ENV, "bind")
    sandbox = _sandbox(tmp_path)
    environment = tmp_path / "environment"
    tool = environment / "tool"
    tool.write_text('#!/bin/sh\necho task-supplied-program "$@"\n')
    tool.chmod(0o755)
    compose = {"services": {"main": _BENEATH[how]}}
    if how == "volume-beneath":
        compose["volumes"] = {"data": {}}
    (environment / "docker-compose.yaml").write_text(json.dumps(compose))
    try:
        with pytest.raises(TaskEnvNotAllowedError, match="at or beneath"):
            await sandbox.start(force_build=False)
        assert await sandbox._main_container_id() is None
    finally:
        await _stop(sandbox)


# The check itself, on containers made by hand: each one changes a single
# fact of the launch, and the check must refuse it (the control passes).
_FAULTS = {
    "control": [],
    "writable": ["rw"],
    "sys-admin": ["--cap-add", "SYS_ADMIN"],
    "cap-all": ["--cap-add", "ALL"],
    "unconfined": ["--security-opt", "seccomp=unconfined"],
    "privileged": ["--privileged"],
    "other-bytes": ["other-bytes"],
}


@pytest.mark.parametrize("fault", sorted(_FAULTS))
def test_the_check_refuses_each_unsafe_launch(tmp_path, local, fault):
    arch = _arch_for(local)
    _require_pin(arch)
    source = delivery.host_dir(arch)
    flags = list(_FAULTS[fault])
    mount = f"type=bind,src={source},dst={SANDBOX_DIR},readonly"
    if flags == ["rw"]:
        flags, mount = [], mount.removesuffix(",readonly")
    if flags == ["other-bytes"]:
        other = tmp_path / "other"
        other.mkdir()
        (other / "bfx").write_bytes(b"not the pinned bfx")
        flags, mount = [], mount.replace(f"src={source}", f"src={other}")
    name = f"bf-test-bfxcheck-{uuid.uuid4().hex[:10]}"
    local.containers.append(name)
    local.ok(
        [
            "run", "-d", "--name", name, "--network", "none",
            "--mount", mount, *flags, MAIN_IMAGE, "sleep", "600",
        ]
    )  # fmt: skip
    check = lambda: check_launch(  # noqa: E731
        local, name, mode=delivery.BIND, source=str(source), arch=arch
    )
    if fault == "control":
        check()
    else:
        with pytest.raises(AssertionError):
            check()
