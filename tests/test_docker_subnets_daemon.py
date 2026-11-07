"""Docker trial networks on a real daemon (U14): small, separate, never overlapping.

A stock daemon hands each compose network a whole pool chunk (a /16 from
172.17-31, a /20 from 192.168), about 30 networks, so above about 30 live
trials ``compose up`` failed with "all predefined address pools have been
fully subnetted" and the trial ended unscored. BenchFlow now gives each
trial network a /28 from the daemon's own pools.

Deterministic tier: these run where a Docker daemon on this host answers
(BenchFlow chooses subnets only for such a daemon) and skip with the reason
otherwise. They import nothing but the Docker sandbox, so they also run
against a BenchFlow that predates the change. Only the last test of the
module, the message when the daemon's pools run out, needs no Docker.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
import subprocess
import sys
import textwrap
import uuid
from itertools import pairwise
from pathlib import Path

import pytest

from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

IPv4 = ipaddress.IPv4Network
ENV_VAR = "BENCHFLOW_DOCKER_SUBNETS"
# Docker's built-in local address pools; the daemons these run on set none.
DEFAULT_POOLS = [
    IPv4(p)
    for p in (
        "172.17.0.0/16",
        "172.18.0.0/16",
        "172.19.0.0/16",
        "172.20.0.0/14",
        "172.24.0.0/14",
        "172.28.0.0/14",
        "192.168.0.0/16",
    )
]


def _in_pools(net: IPv4) -> bool:
    return any(net.subnet_of(p) for p in DEFAULT_POOLS)


def _sandbox(tmp_path: Path, name: str, compose: str | None = None) -> DockerSandbox:
    environment = tmp_path / name / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose is not None:
        (environment / "docker-compose.yaml").write_text(textwrap.dedent(compose))
    paths = RolloutPaths(rollout_dir=tmp_path / name / "rollout")
    paths.mkdir()
    return DockerSandbox(
        environment_dir=environment,
        environment_name=name,
        session_id=name,
        rollout_paths=paths,
        task_env_config=SandboxConfig(cpus=1, memory_mb=128),
    )


def _short(name: str) -> str:
    return name.strip().split(".", 1)[0].lower()


@pytest.fixture(autouse=True)
def _fresh_allocators():
    """Each test reads the daemon afresh (and the setting it sets)."""
    try:
        from benchflow.sandbox import _docker_subnets as subnets
    except ImportError:  # a BenchFlow from before U14
        yield
        return
    subnets.reset_allocators()
    yield
    subnets.reset_allocators()


def _docker_here() -> str | None:
    """Why the real-daemon tests cannot run here, or None."""
    from tests.integration.deterministic import harness as h

    sandbox, reason = h.select_sandbox()
    if sandbox != "docker":
        return reason if sandbox is None else f"Docker-only scenario ({sandbox})"
    if os.environ.get("DOCKER_HOST", "").startswith(("tcp://", "ssh://")):
        return "the Docker daemon is on another host"
    try:
        name = subprocess.run(
            ["docker", "info", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"docker info failed: {exc}"
    if not sys.platform.startswith("linux") or _short(name) != _short(
        socket.gethostname()
    ):
        return f"the Docker daemon ({name}) is not on this host"
    return None


@pytest.fixture
def real_docker():
    reason = _docker_here()
    if reason is not None:
        pytest.skip(reason)


def _project_subnets(project: str) -> dict[str, IPv4]:
    ids = subprocess.run(
        [
            "docker",
            "network",
            "ls",
            "-q",
            "--filter",
            f"label=com.docker.compose.project={project}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    out: dict[str, IPv4] = {}
    for nid in ids:
        info = json.loads(
            subprocess.run(
                ["docker", "network", "inspect", nid],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )[0]
        out[info["Labels"]["com.docker.compose.network"]] = IPv4(
            info["IPAM"]["Config"][0]["Subnet"]
        )
    return out


def _remove_images(names: list[str]) -> None:
    """Remove the images these tests built (never a pulled one)."""
    subprocess.run(["docker", "image", "rm", *names], capture_output=True)


_TWO_NET_COMPOSE = """\
services:
  main:
    networks: [default, back]
  peer:
    image: ubuntu:24.04
    command: ["sleep", "infinity"]
    networks: [back]
networks:
  back:
"""

_WORKER = """\
import asyncio, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

from benchflow.sandbox.docker import DockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths

try:
    import benchflow.sandbox._docker_subnets as subnets
except ImportError:  # a BenchFlow from before U14
    subnets = None
if subnets is not None:
    # Both processes start at the same point of the range, so they reach
    # for the same subnets at the same moment: only the daemon keeps them
    # apart.
    _real = subnets.random.SystemRandom()
    subnets.random.SystemRandom = lambda: SimpleNamespace(
        randrange=lambda a, b=None: int(os.environ["U14_START"]) if b is None else _real.randrange(a, b)
    )

root, label, count, go = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])

async def one(i):
    env = root / f"{label}{i}" / "environment"
    env.mkdir(parents=True)
    (env / "Dockerfile").write_text("FROM ubuntu:24.04\\n")
    (env / "docker-compose.yaml").write_text(os.environ["U14_COMPOSE"])
    paths = RolloutPaths(rollout_dir=root / f"{label}{i}" / "rollout")
    paths.mkdir()
    s = DockerSandbox(env, f"{label}{i}", f"{label}{i}", paths, SandboxConfig(cpus=1, memory_mb=128))
    await s.start(force_build=False)
    return s

async def main():
    sandboxes = await asyncio.gather(*(one(i) for i in range(count)))
    print(json.dumps({s.compose_project_name: {k: str(v) for k, v in getattr(s, "_subnets", {}).items()} for s in sandboxes}), flush=True)
    while not go.exists():
        await asyncio.sleep(0.2)
    # delete=False: --rmi all would also remove the peer's pulled ubuntu:24.04.
    await asyncio.gather(*(s.stop(delete=False) for s in sandboxes))

asyncio.run(main())
"""


@pytest.mark.deterministic
def test_two_processes_at_once_never_share_a_subnet(real_docker, tmp_path):
    """Two BenchFlow processes start trials at the same moment from the same
    point of the range; every network gets its own non-overlapping /28 (the
    daemon refuses the loser's create and it takes the next one), and
    teardown removes every network."""
    go = tmp_path / "go"
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER)
    env = {
        **os.environ,
        "U14_START": str(uuid.uuid4().int % 60000),
        "U14_COMPOSE": _TWO_NET_COMPOSE,
    }
    env.pop(ENV_VAR, None)
    tag = uuid.uuid4().hex[:6]
    labels = (f"u14a{tag}", f"u14b{tag}")
    reports: dict[str, dict[str, str]] = {}
    procs = [
        subprocess.Popen(
            [sys.executable, str(worker), str(tmp_path / label), label, "6", str(go)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=Path(__file__).resolve().parents[1],
        )
        for label in labels
    ]
    try:
        for proc in procs:
            assert proc.stdout is not None
            line = proc.stdout.readline()
            if not line:
                _, err = proc.communicate(timeout=60)
                pytest.fail(f"worker failed: {err[-3000:]}")
            reports.update(json.loads(line))
        assert len(reports) == 12
        actual = {project: _project_subnets(project) for project in reports}
        nets = [net for found in actual.values() for net in found.values()]
        assert len(nets) == 24
        assert sorted({n.prefixlen for n in nets}) == [28]
        assert all(_in_pools(n) for n in nets)
        for project, chosen in reports.items():
            assert {k: str(v) for k, v in actual[project].items()} == chosen
        ordered = sorted(nets)
        assert all(not a.overlaps(b) for a, b in pairwise(ordered))
    finally:
        go.touch()
        for proc in procs:
            proc.communicate(timeout=300)
        _remove_images([f"bf__{label}{i}" for label in labels for i in range(6)])
    assert [p.returncode for p in procs] == [0, 0]
    for project in reports:
        assert _project_subnets(project) == {}


@pytest.mark.deterministic
@pytest.mark.asyncio
async def test_a_network_made_by_someone_else_is_never_overlapped(
    real_docker, tmp_path
):
    """A network the user makes after BenchFlow read the daemon sits on the
    subnet BenchFlow would take next: the daemon refuses the create, the
    trial takes another subnet, and the user's network is untouched."""
    tag = uuid.uuid4().hex[:6]
    first = _sandbox(tmp_path, f"u14first{tag}")
    await first.start(force_build=False)
    try:
        allocator = getattr(first, "_subnet_allocator", None)
        assert allocator is not None, "BenchFlow chose no subnet for the trial"
        assert _project_subnets(first.compose_project_name)["default"].prefixlen == 28
        from benchflow.sandbox._docker_subnets import NetworkNeed

        peek = allocator.allocate([NetworkNeed("default")])["default"]
        allocator.release([peek])
        foreign = f"u14-foreign-{uuid.uuid4().hex[:8]}"
        subprocess.run(
            ["docker", "network", "create", "--subnet", str(peek), foreign],
            capture_output=True,
            check=True,
        )
        try:
            second = _sandbox(tmp_path, f"u14second{tag}")
            # Started from the same point, the second trial reaches for peek.
            allocator._cursor = int(peek.network_address)
            await second.start(force_build=False)
            got = _project_subnets(second.compose_project_name)["default"]
            assert got != peek and not got.overlaps(peek)
            assert got.prefixlen == 28
            inspect = json.loads(
                subprocess.run(
                    ["docker", "network", "inspect", foreign],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )[0]
            assert inspect["IPAM"]["Config"][0]["Subnet"] == str(peek)
            await second.stop(delete=False)
            assert _project_subnets(second.compose_project_name) == {}
        finally:
            subprocess.run(["docker", "network", "rm", foreign], capture_output=True)
    finally:
        await first.stop(delete=False)
        _remove_images([f"bf__u14first{tag}", f"bf__u14second{tag}"])
    assert _project_subnets(first.compose_project_name) == {}


@pytest.mark.deterministic
@pytest.mark.asyncio
async def test_more_trials_than_a_stock_daemon_has_pools(real_docker, tmp_path):
    """40 trials at once, each with a task network beside its default one:
    80 networks, far past the about 30 a stock daemon gives out itself.
    Every one is a /28 of the daemon's pools and teardown removes them."""
    tag = uuid.uuid4().hex[:6]
    compose = """
    services:
      main:
        networks: [default, back]
    networks:
      back:
    """
    sandboxes = [_sandbox(tmp_path, f"u14many{tag}x{i}", compose) for i in range(40)]
    try:
        results = await asyncio.gather(
            *(s.start(force_build=False) for s in sandboxes), return_exceptions=True
        )
        failed = [str(r) for r in results if isinstance(r, BaseException)]
        pools = sum("address pools" in f for f in failed)
        assert failed == [], (
            f"{len(failed)} of 40 trials failed to start, {pools} on used-up "
            f"address pools; the first: ...{failed[0][-300:]}"
        )
        nets = [
            net
            for s in sandboxes
            for net in _project_subnets(s.compose_project_name).values()
        ]
        assert len(nets) == 80
        assert sorted({n.prefixlen for n in nets}) == [28]
        assert all(_in_pools(n) for n in nets)
        ordered = sorted(nets)
        assert all(not a.overlaps(b) for a, b in pairwise(ordered))
    finally:
        await asyncio.gather(
            *(s.stop(delete=False) for s in sandboxes), return_exceptions=True
        )
        _remove_images([f"bf__u14many{tag}x{i}" for i in range(40)])
    for s in sandboxes:
        assert _project_subnets(s.compose_project_name) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("own_subnets", ["", "daemon"])
async def test_used_up_address_pools_say_widen_the_range_or_lower_concurrency(
    tmp_path, monkeypatch, own_subnets
):
    """The daemon's own pools can still run out (for a network BenchFlow
    leaves to it, or for every network when the daemon chooses): the trial
    says what to do instead of only Docker's words. No Docker needed."""
    monkeypatch.setenv(ENV_VAR, own_subnets)
    sandbox = _sandbox(tmp_path, "pools")
    sandbox._prepare_inert_bootstrap = _noop
    sandbox._release_inert_bootstrap = _noop
    sandbox._probe_verifier_log_mount = _noop  # type: ignore[method-assign]
    sandbox._chown_to_host_user = _noop  # type: ignore[method-assign]
    sandbox._name_trial_volumes = _noop  # type: ignore[method-assign]

    async def compose(
        command, check=True, timeout_sec=None, *, raw_output=False, merge_stderr=True
    ):
        if command[0] == "up":
            raise RuntimeError(
                "Docker compose command failed for environment pools. Command: "
                "docker compose up. Return code: 1. Stdout: failed to create "
                "network pools_default: Error response from daemon: all "
                "predefined address pools have been fully subnetted"
            )
        return ExecResult(stdout="", stderr="", return_code=0)

    async def docker(args, check=True):
        if args[0] == "info":
            info = {"name": socket.gethostname(), "pools": None}
            return ExecResult(stdout=json.dumps(info), stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    sandbox._run_docker_compose_command = compose  # type: ignore[method-assign]
    monkeypatch.setattr(
        DockerSandbox,
        "_docker_cli",
        lambda self, args, check=True, **kwargs: docker(args, check),
    )
    with pytest.raises(SandboxStartupError) as exc:
        await sandbox.start(force_build=False)
    message = str(exc.value)
    assert "--concurrency" in message and ENV_VAR in message
    await sandbox.stop(delete=True)


async def _noop(*args, **kwargs):
    return None
