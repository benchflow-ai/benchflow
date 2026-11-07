"""Guards U14-C6 against daemon-wide client fallbacks introduced at efe8893d."""

from __future__ import annotations

import asyncio
import ipaddress
import json

import pytest

from benchflow.sandbox import _docker_subnets as subnets
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests._docker_daemon_fakes import server_id


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    subnets.reset_allocators()
    for name in (*subnets._CLIENT_SETTINGS, subnets.ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(subnets, "_read_daemon_id", lambda key: "same-daemon")
    # Model the endpoints selected by these fixed aliases; the real inspection
    # boundary and context/host precedence have their own regression matrix.
    monkeypatch.setattr(
        subnets,
        "_uses_local_socket",
        lambda key: (
            not key.endpoint_hint
            and (
                bool(key.env.get("DOCKER_CONTEXT"))
                or not key.env.get("DOCKER_HOST")
                or key.env["DOCKER_HOST"].startswith("unix:///")
            )
        ),
    )
    monkeypatch.setattr(subnets, "daemon_on_this_host", lambda name: True)
    monkeypatch.setattr(subnets, "host_reserved", lambda: [])
    yield
    subnets.reset_allocators()


class Daemon:
    def __init__(self):
        self.info_calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def __call__(self, args):
        if args[0] == "info":
            self.info_calls += 1
            self.entered.set()
            await self.release.wait()
            stdout = json.dumps({"name": "here", "pools": []})
        else:
            stdout = ""
        return ExecResult(stdout=stdout, stderr="", return_code=0)


@pytest.mark.parametrize(
    "remote_host",
    [
        "ssh://dummy@localhost",
        "tcp://localhost:2376",
        "http://localhost:2376",
        "https://localhost:2376",
        "  SSH://dummy@localhost",
    ],
)
@pytest.mark.parametrize(
    "local_env",
    [
        {},
        {"DOCKER_HOST": "unix:///var/run/docker.sock"},
        {"DOCKER_CONTEXT": "local-alias"},
        {"DOCKER_CONFIG": "/tmp/dummy-local-context"},
    ],
)
async def test_remote_fallback_does_not_disable_a_local_alias(remote_host, local_env):
    """Guards U14-C6/efe8893d across URL spellings and local client selectors."""
    remote = subnets.daemon_key({"DOCKER_HOST": remote_host})
    local = subnets.daemon_key(local_env)
    daemon = Daemon()
    assert await subnets.allocator_for(remote, daemon) is None
    assert await subnets.allocator_for(remote, daemon) is None
    assert daemon.info_calls == 1  # The fallback is cached for this client only.
    allocator = await subnets.allocator_for(local, daemon)
    assert allocator is not None
    # A positive result supersedes even an earlier fallback for an alias.
    assert await subnets.allocator_for(remote, daemon) is allocator
    nets = [
        allocator.allocate([subnets.NetworkNeed("default")])["default"]
        for _ in range(40)
    ]
    assert len(set(nets)) == 40
    assert all(net.prefixlen == 28 for net in nets)


@pytest.mark.parametrize("first", ["remote", "local"])
@pytest.mark.parametrize("finish_first", ["remote", "local"])
async def test_pending_fallback_cannot_capture_another_clients_build(
    first, finish_first
):
    """Guards U14-C6/efe8893d for both launch and completion orders."""
    keys = {
        "remote": subnets.daemon_key({"DOCKER_HOST": "ssh://dummy@localhost"}),
        "local": subnets.daemon_key({}),
    }
    runners = {name: Daemon() for name in keys}
    for runner in runners.values():
        runner.release.clear()
    other = "local" if first == "remote" else "remote"
    tasks = {
        first: asyncio.create_task(subnets.allocator_for(keys[first], runners[first]))
    }
    await runners[first].entered.wait()
    tasks[other] = asyncio.create_task(
        subnets.allocator_for(keys[other], runners[other])
    )
    try:
        # Each client must inspect independently while neither has a success.
        await asyncio.wait_for(runners[other].entered.wait(), timeout=1)
        runners[finish_first].release.set()
        await tasks[finish_first]
    finally:
        for runner in runners.values():
            runner.release.set()
        await asyncio.gather(*tasks.values())
    allocator = tasks["local"].result()
    assert allocator is not None
    assert await subnets.allocator_for(keys["remote"], runners["remote"]) is allocator


@pytest.mark.parametrize("same_daemon", [True, False])
async def test_concurrent_successes_share_only_the_same_daemons_leases(
    monkeypatch, same_daemon
):
    """Guards U14-C6/efe8893d without reintroducing U14-C5's split reservations."""
    monkeypatch.setenv(subnets.ENV_VAR, "172.20.0.0/16")
    if not same_daemon:
        monkeypatch.setattr(subnets, "_read_daemon_id", server_id)
    keys = [
        subnets.daemon_key({}),
        subnets.daemon_key({"DOCKER_HOST": "ssh://dummy@localhost"}),
    ]
    pin = ipaddress.IPv4Network("172.20.0.0/28")
    subnets.pin_subnets([pin], key=keys[0])
    daemon = Daemon()
    daemon.release.clear()
    tasks = [asyncio.create_task(subnets.allocator_for(key, daemon)) for key in keys]
    await daemon.entered.wait()
    daemon.release.set()
    allocators = await asyncio.gather(*tasks)
    assert all(a is not None for a in allocators)
    assert (allocators[0] is allocators[1]) == same_daemon
    if same_daemon:
        nets = [
            a.allocate([subnets.NetworkNeed("default")])["default"] for a in allocators
        ]
        assert not nets[0].overlaps(nets[1])
        assert all(not net.overlaps(pin) for net in nets)


async def test_same_client_coalesces_and_reset_forgets_its_fallback():
    """Guards U14-C6/efe8893d: splitting clients must preserve per-client caching."""
    key = subnets.daemon_key({"DOCKER_HOST": "ssh://dummy@localhost"})
    daemon = Daemon()
    assert (
        await asyncio.gather(*(subnets.allocator_for(key, daemon) for _ in range(4)))
        == [None] * 4
    )
    assert daemon.info_calls == 1
    subnets.reset_allocators()
    assert await subnets.allocator_for(key, daemon) is None
    assert daemon.info_calls == 2


@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_unsuccessful_remote_wait_does_not_capture_local_build(failure):
    """Guards U14-C6/efe8893d on error and cancelled-wait paths."""
    remote = subnets.daemon_key({"DOCKER_HOST": "ssh://dummy@localhost"})
    local = subnets.daemon_key({})
    daemon = Daemon()
    daemon.release.clear()

    async def failing(args):
        await daemon(args)
        raise RuntimeError("dummy unavailable endpoint")

    task = asyncio.create_task(
        subnets.allocator_for(remote, failing if failure == "error" else daemon)
    )
    await daemon.entered.wait()
    if failure == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    daemon.release.set()
    if failure == "error":
        assert await task is None
    assert await subnets.allocator_for(local, Daemon()) is not None


async def test_remote_then_local_backends_emit_small_network_overrides(
    tmp_path, monkeypatch
):
    """Guards U14-C6/efe8893d through the shared backend trial/retry path."""
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy@localhost")
    daemon = Daemon()
    monkeypatch.setattr(
        DockerSandbox, "_docker_cli_unchecked", lambda self, args: daemon(args)
    )
    sandboxes = []

    def sandbox(cls, name):
        environment = tmp_path / name / "environment"
        environment.mkdir(parents=True)
        (environment / "Dockerfile").write_text("FROM fixture\n")
        paths = RolloutPaths(rollout_dir=tmp_path / name / "rollout")
        paths.mkdir()
        box = cls(
            environment_dir=environment,
            environment_name=name,
            session_id=name,
            rollout_paths=paths,
            task_env_config=SandboxConfig(),
        )
        sandboxes.append(box)
        return box

    try:
        remote = sandbox(RemoteDockerSandbox, "remote")
        await remote._choose_subnets()
        assert remote._subnets_compose_path is None
        locals_ = [sandbox(DockerSandbox, f"local-{i}") for i in range(40)]
        await asyncio.gather(*(box._choose_subnets() for box in locals_))
        nets = []
        for box in locals_:
            assert box._subnets_compose_path is not None
            doc = json.loads(box._subnets_compose_path.read_text())
            nets.append(
                ipaddress.IPv4Network(
                    doc["networks"]["default"]["ipam"]["config"][0]["subnet"]
                )
            )
        assert len(set(nets)) == 40
        assert all(net.prefixlen == 28 for net in nets)
        # Remote startup/retry also sees the successful shared allocator.
        await remote._choose_subnets(refresh=True)
        assert remote._subnet_allocator is locals_[0]._subnet_allocator
        assert remote._subnets["default"] not in nets
    finally:
        for box in sandboxes:
            box._release_subnets()
