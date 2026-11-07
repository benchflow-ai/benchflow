"""Docker trial networks get small subnets BenchFlow chooses (U14).

A stock daemon hands each compose network a whole pool chunk (a /16 from
172.17-31, a /20 from 192.168), about 30 networks in all, so above about 30
live trials ``compose up`` failed with "all predefined address pools have
been fully subnetted". BenchFlow now gives each trial network a /28 carved
from the daemon's own pools, retries a subnet the daemon refuses, and says
what to do when the range runs out.

Unit tests here need no Docker; the real-daemon tests (two processes, and a
network made between the read and the create) are in the deterministic tier
and skip where no Docker daemon on this host answers.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import textwrap
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _docker_subnets as subnets
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._docker_subnets import (
    ENV_VAR,
    NetworkNeed,
    SubnetAllocator,
    SubnetRangeExhausted,
    daemon_pools,
    host_reserved,
    network_subnets,
    override_document,
    parse_range,
    plan_networks,
    prefix_for,
)
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests._docker_daemon_fakes import server_id


@pytest.fixture(autouse=True)
def _without_bfx(monkeypatch):
    # These check the task's own mounts and Compose files; bfx's read-only
    # overlay (present when the pinned binaries are built) is checked in
    # tests/test_bfx_*.py.
    monkeypatch.setenv("BENCHFLOW_BFX_DELIVERY", "off")


IPv4 = ipaddress.IPv4Network
DEFAULT_POOLS = [IPv4(p) for p in subnets.DOCKER_DEFAULT_POOLS]
DOCKER0 = IPv4("172.17.0.0/16")


@pytest.fixture(autouse=True)
def _fresh_allocators(monkeypatch):
    monkeypatch.setattr(subnets, "_read_daemon_id", server_id)
    # These daemon fixtures use the default socket unless a host is explicit.
    # Endpoint inspection/precedence itself is covered in test_docker_subnet_endpoint.
    monkeypatch.setattr(
        subnets,
        "_uses_local_socket",
        lambda key: (
            not key.env.get("DOCKER_HOST")
            or key.env["DOCKER_HOST"].startswith("unix:///")
        ),
    )
    monkeypatch.delenv(ENV_VAR, raising=False)
    subnets.reset_allocators()
    yield
    subnets.reset_allocators()


def _overlap(a: IPv4, b: IPv4) -> bool:
    return a.overlaps(b)


def _in_pools(net: IPv4, pools: list[IPv4]) -> bool:
    return any(net.subnet_of(p) for p in pools)


# --- the allocator ----------------------------------------------------------


def test_a_stock_daemons_pools_hold_128_trial_networks_and_more():
    """128 trials (and far more) each get their own /28 from Docker's own
    pools, none overlapping the daemon's bridge or each other."""
    allocator = SubnetAllocator(DEFAULT_POOLS, start=0)
    allocator.set_taken([DOCKER0])
    got = [allocator.allocate([NetworkNeed("default")])["default"] for _ in range(4096)]
    assert len(set(got)) == 4096
    assert all(n.prefixlen == 28 for n in got)
    assert all(_in_pools(n, DEFAULT_POOLS) for n in got)
    assert not any(_overlap(n, DOCKER0) for n in got)
    ordered = sorted(got)
    assert all(not a.overlaps(b) for a, b in pairwise(ordered))


def test_the_default_range_is_the_daemons_pools_never_new_private_space():
    assert daemon_pools(None) == DEFAULT_POOLS
    assert daemon_pools([]) == DEFAULT_POOLS
    configured = [
        {"Base": "172.80.0.0/16", "Size": 24},
        {"Base": "fd00::/8", "Size": 64},
    ]
    assert daemon_pools(configured) == [IPv4("172.80.0.0/16")]
    ten = IPv4("10.0.0.0/8")
    assert not any(p.overlaps(ten) for p in DEFAULT_POOLS)
    # A process at any start point stays inside the pools.
    for start in (0, 17, 4095, 50_000, 65_535):
        allocator = SubnetAllocator(DEFAULT_POOLS, start=start)
        for _ in range(64):
            net = allocator.allocate([NetworkNeed("default")])["default"]
            assert _in_pools(net, DEFAULT_POOLS) and not net.overlaps(ten)


def test_taken_networks_and_host_routes_are_skipped():
    reserved = [IPv4("172.31.0.0/16"), IPv4("192.168.1.1/32")]
    allocator = SubnetAllocator(
        [IPv4("172.30.0.0/15"), IPv4("192.168.0.0/20")],
        reserved=lambda: reserved,
        start=0,
    )
    taken = [IPv4("172.30.0.0/17"), IPv4("172.30.200.0/28")]
    allocator.set_taken(taken)
    # Free: 172.30.128.0/17 but one /28 (2,047), 192.168.0.0/20 but the
    # nameserver's /28 (255); 172.31.0.0/16 is a host route.
    got = [allocator.allocate([NetworkNeed("n")])["n"] for _ in range(2047 + 255)]
    for net in got:
        assert not any(net.overlaps(t) for t in [*taken, *reserved]), net
    assert len(set(got)) == len(got)
    with pytest.raises(SubnetRangeExhausted):
        allocator.allocate([NetworkNeed("n")])


def test_released_subnets_come_back_and_exhaustion_says_what_to_do():
    allocator = SubnetAllocator([IPv4("192.168.50.0/26")], start=0)
    first = [allocator.allocate([NetworkNeed("default")])["default"] for _ in range(4)]
    assert sorted(first) == list(IPv4("192.168.50.0/26").subnets(new_prefix=28))
    with pytest.raises(SubnetRangeExhausted) as exc:
        allocator.allocate([NetworkNeed("default")])
    message = str(exc.value)
    assert ENV_VAR in message and "--concurrency" in message
    assert "192.168.50.0/26" in message
    allocator.release([first[2]])
    assert allocator.allocate([NetworkNeed("default")])["default"] == first[2]


def test_a_trial_with_two_networks_gets_two_or_none():
    allocator = SubnetAllocator([IPv4("192.168.50.0/27")], start=0)
    allocator.allocate([NetworkNeed("default")])
    with pytest.raises(SubnetRangeExhausted):
        allocator.allocate([NetworkNeed("default"), NetworkNeed("back")])
    # The half-made lease was handed back: one /28 is still free.
    assert len(allocator.live) == 1
    allocator.allocate([NetworkNeed("default")])


def test_the_walk_wraps_and_moves_forward():
    pools = [IPv4("192.168.0.0/27"), IPv4("192.168.10.0/28")]
    allocator = SubnetAllocator(pools, start=2)  # the second pool's only /28
    a = allocator.allocate([NetworkNeed("x")])["x"]
    b = allocator.allocate([NetworkNeed("x")])["x"]
    c = allocator.allocate([NetworkNeed("x")])["x"]
    assert [a, b, c] == [
        IPv4("192.168.10.0/28"),
        IPv4("192.168.0.0/28"),
        IPv4("192.168.0.16/28"),
    ]
    allocator.release([a])
    # The cursor moves on; the freed /28 is reached again after a wrap.
    assert allocator.allocate([NetworkNeed("x")])["x"] == a


def test_two_processes_start_at_different_points():
    starts = {
        SubnetAllocator(DEFAULT_POOLS).allocate([NetworkNeed("d")])["d"]
        for _ in range(20)
    }
    assert len(starts) > 15


def test_bigger_networks_when_a_task_needs_them():
    assert prefix_for(1) == 28
    assert prefix_for(11) == 28  # 13 addresses, 2 spare
    assert prefix_for(12) == 27
    assert prefix_for(4091) == 20
    assert prefix_for(4092) is None
    allocator = SubnetAllocator([IPv4("192.168.0.0/24")], start=1)
    got = allocator.allocate([NetworkNeed("a", 28), NetworkNeed("b", 26)])
    assert got["a"].prefixlen == 28 and got["b"].prefixlen == 26
    assert int(got["b"].network_address) % 64 == 0
    assert not got["a"].overlaps(got["b"])


# --- reading the range and the host -----------------------------------------


def test_parse_range():
    assert parse_range("172.20.0.0/14, 192.168.0.0/16") == [
        IPv4("172.20.0.0/14"),
        IPv4("192.168.0.0/16"),
    ]
    for bad in ("fd00::/8", "nonsense", "192.168.0.0/29", " "):
        with pytest.raises(ValueError, match=ENV_VAR):
            parse_range(bad)


def test_host_reserved_reads_routes_and_nameservers(tmp_path):
    route = tmp_path / "route"
    route.write_text(
        "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
        # default route: never reserved
        "eth0\t00000000\t0100800A\t0003\t0\t0\t100\t00000000\t0\t0\t0\n"
        # 172.31.0.0/16 (an AWS default VPC)
        "eth0\t00001FAC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n"
        # 192.168.5.0/24 (a VM's user network)
        "eth1\t0005A8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n"
    )
    resolv = tmp_path / "resolv.conf"
    resolv.write_text("search x\nnameserver 192.168.1.1\nnameserver fe80::1%eth0\n")
    got = host_reserved(route, [resolv, tmp_path / "missing"])
    assert got == [
        IPv4("172.31.0.0/16"),
        IPv4("192.168.5.0/24"),
        IPv4("192.168.1.1/32"),
    ]


def test_network_subnets_reads_inspect_output():
    out = (
        '[{"Subnet":"172.17.0.0/16","Gateway":"172.17.0.1"}]\n'
        "[]\n"
        "null\n"
        '[{"Subnet":"192.168.0.16/28"},{"Subnet":"fd00:1::/64"}]\n'
        "Error: No such network: abc\n"
    )
    assert network_subnets(out) == [IPv4("172.17.0.0/16"), IPv4("192.168.0.16/28")]


def test_daemon_on_this_host(monkeypatch):
    monkeypatch.setattr(subnets.sys, "platform", "linux")
    monkeypatch.setattr(subnets.socket, "gethostname", lambda: "vm-1.zone.internal")
    assert subnets.daemon_on_this_host("vm-1")
    assert subnets.daemon_on_this_host("VM-1.zone.internal")
    assert not subnets.daemon_on_this_host("docker-desktop")
    assert not subnets.daemon_on_this_host("")
    monkeypatch.setattr(subnets.sys, "platform", "darwin")
    assert not subnets.daemon_on_this_host("vm-1")


# --- which networks get a subnet ---------------------------------------------


def _compose(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "docker-compose.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_a_dockerfile_task_gets_a_subnet_for_its_default_network():
    assert plan_networks(None).needs == [NetworkNeed("default", 28)]
    # No-network: main has network_mode none, Compose makes no network.
    assert plan_networks(None, main_networked=False).needs == []


def test_task_networks_without_ipam_get_one_and_explicit_ipam_is_left_alone(tmp_path):
    path = _compose(
        tmp_path,
        """
        services:
          main:
            networks: [front, back]
          db:
            image: postgres
            networks:
              back: {}
          cache:
            image: redis
            networks: [pinned]
          sidecar:
            image: busybox
            network_mode: "service:main"
          ext:
            image: busybox
            networks: [outside, named, overlay, v6]
        networks:
          front:
          back:
            internal: true
          pinned:
            ipam:
              config:
                - subnet: 10.77.0.0/24
          outside:
            external: true
            ipam:
              config:
                - subnet: 10.78.0.0/24
          named:
            name: shared-net
          overlay:
            driver: overlay
          v6:
            enable_ipv6: true
          unused: {}
        """,
    )
    plan = plan_networks(path)
    assert plan is not None
    assert plan.needs == [NetworkNeed("back", 28), NetworkNeed("front", 28)]
    # The task's own subnet, which the trial's other networks must avoid; an
    # external network's ipam is not the task's to set.
    assert plan.pinned == [IPv4("10.77.0.0/24")]


def test_the_tasks_own_default_network_settings_win(tmp_path):
    path = _compose(
        tmp_path,
        """
        services:
          main: {}
        networks:
          default:
            ipam:
              config:
                - subnet: 172.29.5.0/24
        """,
    )
    plan = plan_networks(path)
    assert plan is not None
    assert plan.needs == [] and plan.pinned == [IPv4("172.29.5.0/24")]
    path = _compose(
        tmp_path, "services:\n  main: {}\nnetworks:\n  default:\n    driver: bridge\n"
    )
    assert plan_networks(path).needs == [NetworkNeed("default", 28)]


def test_a_crowded_network_gets_a_bigger_subnet(tmp_path):
    path = _compose(
        tmp_path,
        """
        services:
          main: {}
          workers:
            image: busybox
            deploy:
              replicas: 20
        """,
    )
    assert plan_networks(path).needs == [NetworkNeed("default", 27)]


def test_a_compose_file_benchflow_cannot_read_is_left_to_the_daemon(tmp_path):
    assert plan_networks(_compose(tmp_path, "services: [\n")) is None
    assert plan_networks(_compose(tmp_path, "include:\n  - other.yaml\n")) is None
    assert plan_networks(_compose(tmp_path, "!!python/object:os.system x\n")) is None
    # Compose's own tags parse.
    path = _compose(
        tmp_path,
        "services:\n  main:\n    networks: !reset []\n  b:\n    image: x\n    networks: !override [n]\nnetworks:\n  n: {}\n",
    )
    plan = plan_networks(path)
    assert plan is not None
    assert plan.needs == [NetworkNeed("default", 28), NetworkNeed("n", 28)]


def test_override_document_names_only_ipam():
    doc = override_document(
        {"default": IPv4("172.23.0.16/28"), "back": IPv4("172.23.0.32/28")}
    )
    assert doc == {
        "networks": {
            "default": {"ipam": {"config": [{"subnet": "172.23.0.16/28"}]}},
            "back": {"ipam": {"config": [{"subnet": "172.23.0.32/28"}]}},
        }
    }


# --- the Docker sandbox ------------------------------------------------------


class _FakeDaemon:
    """``docker info`` / ``network ls`` / ``network inspect`` answers."""

    def __init__(self, name: str, networks: list[str], pools=None) -> None:
        self.name = name
        self.networks = list(networks)
        self.pools = pools
        self.calls: list[list[str]] = []

    async def __call__(
        self, args: list[str], check: bool = True, *, raw_output=False
    ) -> ExecResult:
        self.calls.append(args)
        if args[0] == "info":
            out = json.dumps({"name": self.name, "pools": self.pools})
        elif args[:2] == ["network", "ls"]:
            out = "\n".join(f"id{i}" for i in range(len(self.networks)))
        elif args[:2] == ["network", "inspect"]:
            out = "\n".join(json.dumps([{"Subnet": n}]) for n in self.networks)
        elif args[:2] == ["image", "inspect"]:
            out = "null"  # the fixture's images declare no VOLUME
        else:
            raise AssertionError(args)
        return ExecResult(stdout=out, stderr="", return_code=0)


def _sandbox(tmp_path: Path, name: str = "t", compose: str | None = None, **config):
    environment = tmp_path / name / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose is not None:
        (environment / "docker-compose.yaml").write_text(textwrap.dedent(compose))
    paths = RolloutPaths(rollout_dir=tmp_path / name / "rollout")
    paths.mkdir()
    sandbox = DockerSandbox(
        environment_dir=environment,
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(**config),
    )
    # The subnet fake has an async runner but no Compose admission client.
    sandbox.freeze_launch = lambda *args: None
    sandbox._probe_verifier_log_mount = _noop  # type: ignore[method-assign]
    sandbox._pin_or_verify_bind_targets = _noop  # type: ignore[method-assign]
    sandbox._prepare_inert_bootstrap = _noop
    sandbox._release_inert_bootstrap = _noop
    sandbox._chown_to_host_user = _noop  # type: ignore[method-assign]
    sandbox._name_trial_volumes = _noop  # type: ignore[method-assign]
    return sandbox


async def _noop(*args, **kwargs):
    return None


class _Compose:
    """A compose runner that records each command with the override it saw."""

    def __init__(self, sandbox: DockerSandbox, up_errors: list[str] | None = None):
        self.sandbox = sandbox
        self.up_errors = list(up_errors or [])
        self.calls: list[tuple[list[str], dict | None]] = []

    async def __call__(
        self,
        command,
        check=True,
        timeout_sec=None,
        *,
        raw_output=False,
        merge_stderr=True,
    ):
        path = self.sandbox._subnets_compose_path
        override = json.loads(path.read_text()) if path is not None else None
        if path is not None:
            assert path in self.sandbox._docker_compose_paths
        self.calls.append((list(command), override))
        if command == ["config", "--format", "json"]:
            return ExecResult(
                stdout='{"services": {"main": {"image": "fixture-main:latest", "build": {"context": "."}}}}',
                return_code=0,
            )
        if command[0] == "up" and self.up_errors:
            raise RuntimeError(
                "Docker compose command failed for environment t. Command: docker "
                f"compose up. Return code: 1. Stdout: {self.up_errors.pop(0)}"
            )
        return ExecResult(stdout="", stderr="", return_code=0)

    def ups(self):
        return [o for c, o in self.calls if c[0] == "up"]


def _local(monkeypatch, daemon: _FakeDaemon) -> None:
    monkeypatch.setattr(subnets, "daemon_on_this_host", lambda name: name == "here")
    monkeypatch.setattr(subnets, "host_reserved", lambda: [IPv4("192.168.0.0/24")])
    monkeypatch.setattr(
        DockerSandbox,
        "_docker_cli",
        lambda self, args, check=True, **kwargs: daemon(args, check, **kwargs),
    )


def _subnet_of(override: dict, network: str = "default") -> IPv4:
    return IPv4(override["networks"][network]["ipam"]["config"][0]["subnet"])


@pytest.mark.asyncio
async def test_start_gives_each_trial_its_own_small_subnets(tmp_path, monkeypatch):
    daemon = _FakeDaemon("here", ["172.17.0.0/16", "172.18.0.0/16"])
    _local(monkeypatch, daemon)
    compose = """
    services:
      main: {}
      db:
        image: postgres
        networks: [back, default]
    networks:
      back:
    """
    seen: list[IPv4] = []
    sandboxes = []
    for i in range(3):
        sandbox = _sandbox(tmp_path, f"t{i}", compose)
        runner = _Compose(sandbox)
        sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
        await sandbox.start(force_build=False)
        (override,) = runner.ups()
        assert set(override["networks"]) == {"default", "back"}
        for name in ("default", "back"):
            net = _subnet_of(override, name)
            assert net.prefixlen == 28 and _in_pools(net, DEFAULT_POOLS)
            assert not net.overlaps(IPv4("172.17.0.0/16"))
            assert not net.overlaps(IPv4("172.18.0.0/16"))
            assert not net.overlaps(IPv4("192.168.0.0/24"))  # a host route
            seen.append(net)
        sandboxes.append(sandbox)
    assert len(set(seen)) == 6
    # The daemon was read once for the whole process, not per trial.
    assert [c[0] for c in daemon.calls if c[0] != "ps"] == [
        "info",
        "network",
        "network",
    ]

    files = [s._subnets_compose_path for s in sandboxes]
    for sandbox in sandboxes:
        await sandbox.stop(delete=True)
    assert all(not f.exists() for f in files)
    assert all(s._subnets_compose_path is None for s in sandboxes)
    allocator = subnets._ALLOCATORS[subnets.daemon_identity(subnets.daemon_key(None))]
    assert allocator is not None and allocator.live == []


@pytest.mark.asyncio
async def test_a_refused_subnet_is_retried_with_the_next_free_one(
    tmp_path, monkeypatch
):
    daemon = _FakeDaemon("here", ["172.17.0.0/16"])
    _local(monkeypatch, daemon)
    monkeypatch.setattr(
        subnets.random,
        "SystemRandom",
        lambda: SimpleNamespace(randrange=lambda *a: 4096),
    )
    sandbox = _sandbox(tmp_path)
    overlap = (
        "failed to create network t_default: Error response from daemon: invalid "
        "pool request: Pool overlaps with other one on this address space"
    )
    runner = _Compose(sandbox, up_errors=[overlap])
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]

    # Another process made a network on the first candidate after this one
    # read the daemon.
    async def first_read_then_foreign(args, check=True):
        result = await _FakeDaemon.__call__(daemon, args, check)
        if (
            args[:2] == ["network", "inspect"]
            and "172.18.0.0/28" not in daemon.networks
        ):
            daemon.networks.append("172.18.0.0/28")
        return result

    monkeypatch.setattr(
        DockerSandbox,
        "_docker_cli",
        lambda self, args, check=True, **kwargs: first_read_then_foreign(args, check),
    )
    await sandbox.start(force_build=False)

    first, second = runner.ups()
    assert _subnet_of(first) == IPv4("172.18.0.0/28")
    assert _subnet_of(second) != IPv4("172.18.0.0/28")
    assert not _subnet_of(second).overlaps(IPv4("172.18.0.0/28"))
    # `ps` finds the main container after `up` (6227141a).
    commands = [c[0] for c, _ in runner.calls if c[0] not in ("exec", "ps")]
    # The half-made project is taken down before the second try (a first
    # start has nothing to take down, U10).
    assert commands == ["build", "up", "down", "up"]
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_a_subnet_refused_every_time_ends_the_trial_with_a_reason(
    tmp_path, monkeypatch
):
    _local(monkeypatch, _FakeDaemon("here", []))
    sandbox = _sandbox(tmp_path)
    overlap = "Error response from daemon: invalid pool request: Pool overlaps with other one on this address space"
    runner = _Compose(sandbox, up_errors=[overlap] * 10)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    with pytest.raises(SandboxStartupError, match="--concurrency"):
        await sandbox.start(force_build=False)
    assert len(runner.ups()) == 5
    assert len({_subnet_of(o) for o in runner.ups()}) == 5
    path = sandbox._subnets_compose_path
    await sandbox.stop(delete=True)  # what the rollout's cleanup does
    assert path is not None and not path.exists()


@pytest.mark.asyncio
async def test_an_exhausted_range_says_widen_it_or_lower_concurrency(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(ENV_VAR, "192.168.77.0/27")
    _local(monkeypatch, _FakeDaemon("here", []))
    held = []
    for i in range(2):
        sandbox = _sandbox(tmp_path, f"t{i}")
        sandbox._run_docker_compose_command = _Compose(sandbox)  # type: ignore[method-assign]
        await sandbox.start(force_build=False)
        held.append(sandbox)
    third = _sandbox(tmp_path, "t2")
    runner = _Compose(third)
    third._run_docker_compose_command = runner  # type: ignore[method-assign]
    with pytest.raises(SandboxStartupError) as exc:
        await third.start(force_build=False)
    message = str(exc.value)
    assert ENV_VAR in message and "--concurrency" in message
    assert runner.ups() == []
    # A trial that ends frees its subnet for the next one.
    await held[0].stop(delete=True)
    await third.start(force_build=False)
    assert _subnet_of(runner.ups()[0]) in list(
        IPv4("192.168.77.0/27").subnets(new_prefix=28)
    )
    for sandbox in (held[1], third):
        await sandbox.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("chooses", [True, False])
async def test_the_daemons_own_pools_running_out_says_what_to_do(
    tmp_path, monkeypatch, chooses
):
    """A network left to the daemon (or every network, when BenchFlow does not
    choose) can still exhaust its pools: the trial says what to do."""
    _local(monkeypatch, _FakeDaemon("here" if chooses else "elsewhere", []))
    sandbox = _sandbox(tmp_path)
    exhausted = (
        "failed to create network t_default: Error response from daemon: all "
        "predefined address pools have been fully subnetted"
    )
    runner = _Compose(sandbox, up_errors=[exhausted])
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    with pytest.raises(SandboxStartupError) as exc:
        await sandbox.start(force_build=False)
    message = str(exc.value)
    assert "--concurrency" in message and ENV_VAR in message
    assert (runner.ups()[0] is not None) is chooses
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_a_daemon_on_another_host_chooses_its_own_subnets(tmp_path, monkeypatch):
    """BenchFlow cannot read a VM's or a remote host's routes, so it leaves
    the choice to the daemon unless BENCHFLOW_DOCKER_SUBNETS names a range.
    Guards the fallback security overlay retained by 8ff590ca2.
    """
    from benchflow.sandbox._compose import COMPOSE_NO_NEW_PRIVILEGES_PATH

    _local(monkeypatch, _FakeDaemon("docker-desktop", []))
    sandbox = _sandbox(tmp_path)
    runner = _Compose(sandbox)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    assert runner.ups() == [None]
    assert sandbox._docker_compose_paths == [
        DockerSandbox._DOCKER_COMPOSE_BASE_PATH,
        DockerSandbox._DOCKER_COMPOSE_BUILD_PATH,
        COMPOSE_NO_NEW_PRIVILEGES_PATH,
    ]

    subnets.reset_allocators()
    monkeypatch.setenv(ENV_VAR, "172.30.0.0/16")
    other = _sandbox(tmp_path, "u")
    runner = _Compose(other)
    other._run_docker_compose_command = runner  # type: ignore[method-assign]
    await other.start(force_build=False)
    assert _subnet_of(runner.ups()[0]).subnet_of(IPv4("172.30.0.0/16"))
    await other.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["daemon", "off"])
async def test_the_setting_can_hand_the_choice_back_to_the_daemon(
    tmp_path, monkeypatch, value
):
    monkeypatch.setenv(ENV_VAR, value)
    daemon = _FakeDaemon("here", [])
    _local(monkeypatch, daemon)
    sandbox = _sandbox(tmp_path)
    runner = _Compose(sandbox)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    assert runner.ups() == [None]
    # Anonymous-volume teardown discovery is independent of subnet allocation.
    assert all(call[0] == "ps" for call in daemon.calls)


@pytest.mark.asyncio
async def test_a_malformed_setting_fails_the_trial_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "10.0.0.0/33")
    _local(monkeypatch, _FakeDaemon("here", []))
    sandbox = _sandbox(tmp_path)
    sandbox._run_docker_compose_command = _Compose(sandbox)  # type: ignore[method-assign]
    with pytest.raises(SandboxStartupError, match=ENV_VAR):
        await sandbox.start(force_build=False)


@pytest.mark.asyncio
async def test_a_daemon_that_did_not_answer_is_asked_again(tmp_path, monkeypatch):
    """docker info failing once (a daemon still starting) must not turn
    subnet choice off for the rest of the process."""
    daemon = _FakeDaemon("here", [])
    _local(monkeypatch, daemon)
    # Not answering for the whole first trial: start() reads the daemon early
    # (beside its Compose checks) and _choose_subnets asks again (PERF-D).
    answers = iter([ExecResult(stdout="", stderr="Cannot connect", return_code=1)] * 2)

    async def flaky(args, check=True):
        if args[0] == "info":
            nxt = next(answers, None)
            if nxt is not None:
                return nxt
        return await daemon(args, check)

    monkeypatch.setattr(
        DockerSandbox,
        "_docker_cli",
        lambda self, args, check=True, **kwargs: flaky(args, check),
    )
    first = _sandbox(tmp_path, "t0")
    runner = _Compose(first)
    first._run_docker_compose_command = runner  # type: ignore[method-assign]
    await first.start(force_build=False)
    assert runner.ups() == [None]
    second = _sandbox(tmp_path, "t1")
    runner = _Compose(second)
    second._run_docker_compose_command = runner  # type: ignore[method-assign]
    await second.start(force_build=False)
    assert _subnet_of(runner.ups()[0]).prefixlen == 28
    for sandbox in (first, second):
        await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_kept_containers_keep_their_subnets(tmp_path, monkeypatch):
    _local(monkeypatch, _FakeDaemon("here", []))
    sandbox = _sandbox(tmp_path)
    sandbox._keep_containers = True
    sandbox._run_docker_compose_command = _Compose(sandbox)  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    held = dict(sandbox._subnets)
    await sandbox.stop(delete=False)
    allocator = subnets._ALLOCATORS[subnets.daemon_identity(subnets.daemon_key(None))]
    assert allocator is not None and allocator.live == list(held.values())
    assert sandbox._subnets_compose_path is not None
    assert sandbox._subnets_compose_path.exists()
    sandbox._release_subnets()


# --- whole pool chunks next to host routes (review round 1, A-1) -----------

# An AWS default-VPC host: a default route, the on-link 172.31.16.0/20,
# Docker's default bridge and one of BenchFlow's own /28 trial bridges.
_AWS_ROUTES = (
    "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
    "ens5\t00000000\t01101FAC\t0003\t0\t0\t100\t00000000\t0\t0\t0\n"
    "ens5\t00101FAC\t00000000\t0001\t0\t0\t100\t00F0FFFF\t0\t0\t0\n"
    "docker0\t000011AC\t00000000\t0001\t0\t0\t0\t0000FFFF\t0\t0\t0\n"
    "br-0123456789ab\t100012AC\t00000000\t0001\t0\t0\t0\tF0FFFFFF\t0\t0\t0\n"
)


def _start_at(address: str, pools: list[IPv4] = DEFAULT_POOLS) -> int:
    """The start index that puts a fresh allocator's cursor on *address*."""
    target = int(ipaddress.IPv4Address(address))
    index = 0
    for pool in sorted(pools):
        lo, hi = int(pool.network_address), int(pool.broadcast_address)
        if lo <= target <= hi:
            return index + (target - lo) // 16
        index += (hi - lo + 1) // 16
    raise AssertionError(address)


def _fixed_start(monkeypatch, address: str, pools: list[IPv4] = DEFAULT_POOLS) -> None:
    index = _start_at(address, pools)
    monkeypatch.setattr(
        subnets.random,
        "SystemRandom",
        lambda: SimpleNamespace(randrange=lambda *args: index),
    )


def test_the_daemons_pool_chunks():
    assert subnets.daemon_pool_chunks(None) == [
        *((p, 16) for p in DEFAULT_POOLS[:-1]),
        (IPv4("192.168.0.0/16"), 20),
    ]
    configured = [
        {"Base": "172.80.0.0/16", "Size": 24},
        {"Base": "10.200.0.0/16"},  # no Size: the whole base
        {"Base": "10.201.0.0/16", "Size": 12},  # bigger than its base
        {"Base": "fd00::/8", "Size": 64},
    ]
    assert subnets.daemon_pool_chunks(configured) == [
        (IPv4("172.80.0.0/16"), 24),
        (IPv4("10.200.0.0/16"), 16),
        (IPv4("10.201.0.0/16"), 16),
    ]


def test_routes_on_dockers_own_bridges_are_the_daemons_networks(tmp_path):
    route = tmp_path / "route"
    route.write_text(_AWS_ROUTES)
    assert host_reserved(route, []) == [IPv4("172.31.16.0/20")]


@pytest.mark.parametrize(
    ("route", "nameserver", "start", "chunk", "first"),
    [
        # An AWS default VPC: the daemon never uses 172.31.0.0/16, so a /28
        # at 172.31.32.0 would take traffic for another AZ's subnet.
        ("172.31.16.0/20", None, "172.31.32.0", "172.31.0.0/16", "192.168.0.0/28"),
        # A LAN on 192.168.1.0/24: the daemon skips 192.168.0.0/20.
        ("192.168.1.0/24", None, "192.168.2.0", "192.168.0.0/20", "192.168.16.0/28"),
        # A nameserver blocks its chunk too.
        (None, "172.20.5.2", "172.20.6.0", "172.20.0.0/16", "172.21.0.0/28"),
    ],
)
def test_a_host_route_blocks_its_whole_pool_chunk(
    route, nameserver, start, chunk, first
):
    reserved = [IPv4(route) if route else IPv4(f"{nameserver}/32")]
    allocator = SubnetAllocator(
        DEFAULT_POOLS,
        reserved=lambda: reserved,
        chunks=subnets.daemon_pool_chunks(None),
        start=_start_at(start),
    )
    got = [allocator.allocate([NetworkNeed("d")])["d"] for _ in range(64)]
    assert got[0] == IPv4(first)
    assert not any(net.overlaps(IPv4(chunk)) for net in got)


@pytest.mark.asyncio
async def test_a_trial_on_an_aws_host_never_takes_the_vpcs_chunk(tmp_path, monkeypatch):
    """Through the sandbox: the daemon's pools and an AWS host's routes; the
    process's random start lands on 172.31.32.0, in the VPC's /16."""
    _local(monkeypatch, _FakeDaemon("here", ["172.17.0.0/16"]))
    monkeypatch.setattr(subnets, "host_reserved", lambda: [IPv4("172.31.16.0/20")])
    _fixed_start(monkeypatch, "172.31.32.0")
    sandbox = _sandbox(tmp_path)
    runner = _Compose(sandbox)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    net = _subnet_of(runner.ups()[0])
    assert not net.overlaps(IPv4("172.31.0.0/16")), net
    assert net == IPv4("192.168.0.0/28")
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_a_named_range_is_used_as_given_next_to_a_route(tmp_path, monkeypatch):
    """BENCHFLOW_DOCKER_SUBNETS names ranges the user vouches for: only the
    route's own addresses are skipped there, no pool chunks."""
    monkeypatch.setenv(ENV_VAR, "172.31.0.0/16")
    _local(monkeypatch, _FakeDaemon("here", []))
    monkeypatch.setattr(subnets, "host_reserved", lambda: [IPv4("172.31.16.0/20")])
    _fixed_start(monkeypatch, "172.31.32.0", [IPv4("172.31.0.0/16")])
    sandbox = _sandbox(tmp_path)
    runner = _Compose(sandbox)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    assert _subnet_of(runner.ups()[0]) == IPv4("172.31.32.0/28")
    await sandbox.stop(delete=True)


# --- a subnet the task pins itself (review round 1, A-2) --------------------


@pytest.mark.asyncio
async def test_the_default_network_avoids_a_subnet_the_task_pins(tmp_path, monkeypatch):
    """A cyber task pins 172.20.0.0/16 for its own network; the process's
    cursor sits inside it. The trial's default network must go elsewhere, or
    the daemon refuses one of the two on every attempt."""
    _local(monkeypatch, _FakeDaemon("here", ["172.17.0.0/16"]))
    _fixed_start(monkeypatch, "172.20.3.0")
    compose = """
    services:
      main: {}
      target:
        image: busybox
        networks: [vulnnet, default]
    networks:
      vulnnet:
        ipam:
          config:
            - subnet: 172.20.0.0/16
    """
    sandbox = _sandbox(tmp_path, "t", compose)
    runner = _Compose(sandbox)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    (override,) = runner.ups()
    assert set(override["networks"]) == {"default"}
    net = _subnet_of(override)
    assert not net.overlaps(IPv4("172.20.0.0/16")), net
    assert net == IPv4("172.21.0.0/28")
    await sandbox.stop(delete=True)


# --- a subnet a task pins, kept clear for every trial (review round 2, A-1)


def test_a_pinned_subnet_stays_blocked_for_every_later_trial():
    """Another task's trial must never get a /28 inside a subnet one task
    pins: while that /28 lives, the daemon refuses the pinned network."""
    allocator = SubnetAllocator([IPv4("192.168.50.0/26")], start=0)
    pinned = [IPv4("192.168.50.0/27")]
    got = allocator.allocate([NetworkNeed("d")], avoid=pinned)["d"]
    assert got == IPv4("192.168.50.32/28")
    # Later calls without it (other tasks' trials) still keep clear of it.
    assert allocator.allocate([NetworkNeed("d")])["d"] == IPv4("192.168.50.48/28")
    with pytest.raises(SubnetRangeExhausted):
        allocator.allocate([NetworkNeed("d")])
    assert subnets.pinned_spans() == [
        (int(IPv4("192.168.50.0/27")[0]), int(IPv4("192.168.50.0/27")[-1]))
    ]


_PINNING_COMPOSE = """
services:
  main: {}
  target:
    image: busybox
    networks: [vulnnet, default]
networks:
  vulnnet:
    ipam:
      config:
        - subnet: 172.20.0.0/16
"""


@pytest.mark.asyncio
async def test_a_job_pins_its_tasks_subnets_before_any_trial(tmp_path, monkeypatch):
    """Task X pins 172.20.0.0/16. A trial of another task starts first, with
    the process's cursor at 172.20.3.0: its /28 must fall outside, or X's
    first trial is refused while that /28 lives."""
    _local(monkeypatch, _FakeDaemon("here", ["172.17.0.0/16"]))
    _fixed_start(monkeypatch, "172.20.3.0")
    task_x = tmp_path / "x" / "environment"
    task_x.mkdir(parents=True)
    (task_x / "docker-compose.yaml").write_text(textwrap.dedent(_PINNING_COMPOSE))
    missing = tmp_path / "y" / "environment" / "docker-compose.yaml"
    assert subnets.pin_task_subnets([task_x / "docker-compose.yaml", missing]) == [
        IPv4("172.20.0.0/16")
    ]
    sandbox = _sandbox(tmp_path, "other")
    runner = _Compose(sandbox)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    net = _subnet_of(runner.ups()[0])
    assert not net.overlaps(IPv4("172.20.0.0/16")), net
    assert net == IPv4("172.21.0.0/28")
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_a_trial_pins_its_subnets_for_the_trials_after_it(tmp_path, monkeypatch):
    """Without the job's list (a trial started through the SDK), the first
    trial of the pinning task still keeps its subnet clear for later ones."""
    _local(monkeypatch, _FakeDaemon("here", ["172.17.0.0/16"]))
    _fixed_start(monkeypatch, "172.19.255.0")
    first = _sandbox(tmp_path, "x", _PINNING_COMPOSE)
    first._run_docker_compose_command = _Compose(first)  # type: ignore[method-assign]
    await first.start(force_build=False)
    seen = []
    for i in range(20):  # walks past 172.19.255.255 into 172.20/16
        sandbox = _sandbox(tmp_path, f"o{i}")
        runner = _Compose(sandbox)
        sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
        await sandbox.start(force_build=False)
        seen.append(_subnet_of(runner.ups()[0]))
    assert not any(net.overlaps(IPv4("172.20.0.0/16")) for net in seen), seen
    assert IPv4("172.21.0.0/28") in seen


@pytest.mark.asyncio
async def test_a_refused_pinned_network_ends_the_trial_at_once(tmp_path, monkeypatch):
    """The daemon refuses the network the task pins (a network made outside
    this process overlaps it): choosing other /28s for the trial's default
    network cannot help, so the trial ends on the first refusal, naming the
    pinned subnet instead of advising a lower --concurrency."""
    _local(monkeypatch, _FakeDaemon("here", []))
    sandbox = _sandbox(tmp_path, "t", _PINNING_COMPOSE)
    refused = (
        f"failed to create network {sandbox.compose_project_name}_vulnnet: Error "
        "response from daemon: invalid pool request: Pool overlaps with other "
        "one on this address space"
    )
    runner = _Compose(sandbox, up_errors=[refused] * 5)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    with pytest.raises(SandboxStartupError) as exc:
        await sandbox.start(force_build=False)
    message = str(exc.value)
    assert f"{sandbox.compose_project_name}_vulnnet" in message
    assert "172.20.0.0/16" in message
    assert "--concurrency" not in message
    assert len(runner.ups()) == 1
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_a_refused_chosen_network_beside_a_pinned_one_is_still_retried(
    tmp_path, monkeypatch
):
    _local(monkeypatch, _FakeDaemon("here", []))
    sandbox = _sandbox(tmp_path, "t", _PINNING_COMPOSE)
    refused = (
        f"failed to create network {sandbox.compose_project_name}_default: Error "
        "response from daemon: invalid pool request: Pool overlaps with other "
        "one on this address space"
    )
    runner = _Compose(sandbox, up_errors=[refused])
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    assert len(runner.ups()) == 2
    await sandbox.stop(delete=True)


def test_an_evaluation_pins_its_tasks_subnets_before_the_first_trial(
    tmp_path, monkeypatch
):
    from benchflow import evaluation
    from benchflow.review import automatic, resume

    tasks = tmp_path / "tasks"
    for name, compose in (("x", _PINNING_COMPOSE), ("y", None)):
        env = tasks / name / "environment"
        env.mkdir(parents=True)
        if compose is not None:
            (env / "docker-compose.yaml").write_text(textwrap.dedent(compose))
    job = evaluation.Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        job_name="j",
        config=evaluation.EvaluationConfig(environment="docker"),
    )
    seen = []

    async def nothing(*args, **kwargs):
        return None

    async def first_trials(remaining):
        seen.append(subnets.pinned_spans())
        raise _Stop

    monkeypatch.setattr(job, "_get_task_dirs", lambda: [tasks / "x", tasks / "y"])
    monkeypatch.setattr(job, "_get_completed_tasks", dict)
    monkeypatch.setattr(job, "_sweep_docker", nothing)
    monkeypatch.setattr(job, "_write_evaluation_record", lambda: None)
    monkeypatch.setattr(job, "_run_parallel_independent", first_trials)
    monkeypatch.setattr(resume, "resume_pending_reviews", nothing)
    monkeypatch.setattr(automatic, "prepare_review", lambda *a: None)
    with pytest.raises(_Stop):
        asyncio.run(job._run_unlocked())
    net = IPv4("172.20.0.0/16")
    assert seen == [[(int(net[0]), int(net[-1]))]]


class _Stop(Exception):
    pass


# --- a remote daemon that shares this host's name (review round 1, A-3) ----


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["ssh://user@ubuntu", "tcp://ubuntu:2376"])
async def test_a_daemon_reached_over_the_network_is_never_local(monkeypatch, host):
    """A remote host named like this one (stock names such as "ubuntu") has
    its own routes, which BenchFlow cannot read: the daemon chooses."""
    daemon = _FakeDaemon("here", [])
    _local(monkeypatch, daemon)
    assert (
        await subnets.allocator_for(subnets.daemon_key({"DOCKER_HOST": host}), daemon)
        is None
    )
    assert await subnets.allocator_for(
        subnets.daemon_key({"DOCKER_HOST": "unix:///var/run/docker.sock"}), daemon
    )
    # A named range is still used for any daemon.
    subnets.reset_allocators()
    monkeypatch.setenv(ENV_VAR, "172.30.0.0/16")
    allocator = await subnets.allocator_for(
        subnets.daemon_key({"DOCKER_HOST": host}), daemon
    )
    assert allocator is not None
    assert allocator.allocate([NetworkNeed("d")])["d"].subnet_of(IPv4("172.30.0.0/16"))


# --- advice when the task's compose file cannot be read (review round 1, B-3)


@pytest.mark.asyncio
async def test_used_up_pools_for_an_unreadable_compose_file_skip_the_setting(
    tmp_path, monkeypatch
):
    """With include, BenchFlow chooses nothing whatever the setting says, so
    the advice must not point at the setting."""
    _local(monkeypatch, _FakeDaemon("here", []))
    sandbox = _sandbox(
        tmp_path, "t", "include:\n  - other.yaml\nservices:\n  main: {}\n"
    )
    # Guards the 7b4907c8/S1 merge: the allocator cannot inspect includes,
    # but the launch gate must still resolve a valid included project.
    (sandbox.environment_dir / "other.yaml").write_text("services: {}\n")
    monkeypatch.setattr("benchflow.sandbox.docker.plan_networks", lambda *a, **k: None)
    exhausted = (
        "failed to create network t_default: Error response from daemon: all "
        "predefined address pools have been fully subnetted"
    )
    runner = _Compose(sandbox, up_errors=[exhausted])
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    with pytest.raises(SandboxStartupError) as exc:
        await sandbox.start(force_build=False)
    message = str(exc.value)
    assert "--concurrency" in message and "include" in message
    assert ENV_VAR not in message
    assert runner.ups() == [None]


# --- routes through a gateway (review round 2, B-2) ------------------------

# A host on an OpenVPN full tunnel (redirect-gateway def1) and a split
# tunnel to 172.16.0.0/12, both via the tunnel's gateway 10.8.0.1; the
# tunnel's own on-link 10.8.0.0/24 and the LAN's 192.168.1.0/24.
_VPN_ROUTES = (
    "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
    "eth0\t00000000\t0101A8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0\n"
    "tun0\t00000000\t0100080A\t0003\t0\t0\t0\t00000080\t0\t0\t0\n"
    "tun0\t00000080\t0100080A\t0003\t0\t0\t0\t00000080\t0\t0\t0\n"
    "tun0\t000010AC\t0100080A\t0003\t0\t0\t0\t0000F0FF\t0\t0\t0\n"
    "tun0\t0000080A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n"
    "eth0\t0001A8C0\t00000000\t0001\t0\t0\t100\t00FFFFFF\t0\t0\t0\n"
)


def test_routes_through_a_gateway_are_not_reserved(tmp_path):
    """Docker (25 and later) counts only on-link routes; a VPN's /1 routes
    via its gateway would otherwise block every pool."""
    route = tmp_path / "route"
    route.write_text(_VPN_ROUTES)
    assert host_reserved(route, []) == [IPv4("10.8.0.0/24"), IPv4("192.168.1.0/24")]


@pytest.mark.asyncio
async def test_a_vpn_host_still_gets_small_subnets(tmp_path, monkeypatch):
    route = tmp_path / "route"
    route.write_text(_VPN_ROUTES)
    _local(monkeypatch, _FakeDaemon("here", ["172.17.0.0/16"]))
    monkeypatch.setattr(subnets, "ROUTE_FILE", route)
    monkeypatch.setattr(subnets, "RESOLV_FILES", ())
    monkeypatch.setattr(subnets, "host_reserved", host_reserved)
    sandbox = _sandbox(tmp_path)
    runner = _Compose(sandbox)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    net = _subnet_of(runner.ups()[0])
    assert net.prefixlen == 28 and not net.overlaps(IPv4("192.168.0.0/20"))
    await sandbox.stop(delete=True)


@pytest.mark.asyncio
async def test_host_routes_over_every_pool_leave_the_choice_to_the_daemon(
    tmp_path, monkeypatch
):
    """When this host's routes leave no subnet of the daemon's pools, the
    daemon chooses, as before, instead of every trial failing at c=1."""
    _local(monkeypatch, _FakeDaemon("here", []))
    monkeypatch.setattr(
        subnets, "host_reserved", lambda: [IPv4("0.0.0.0/1"), IPv4("128.0.0.0/1")]
    )
    sandbox = _sandbox(tmp_path)
    runner = _Compose(sandbox)
    sandbox._run_docker_compose_command = runner  # type: ignore[method-assign]
    await sandbox.start(force_build=False)
    assert runner.ups() == [None]
    assert subnets.daemon_key(None).identity().connection_key in subnets._FALLBACKS
    assert subnets.daemon_identity(subnets.daemon_key(None)) not in subnets._ALLOCATORS
    await sandbox.stop(delete=True)
