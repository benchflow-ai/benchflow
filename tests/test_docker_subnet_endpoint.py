"""Guards endpoint classification at 849f4111, following the U14-C7 report.

The probe output is authoritative: these fixtures exercise classification,
not a second implementation of Docker's version-dependent client precedence.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _docker_subnets as subnets
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    subnets.reset_allocators()
    for name in (*subnets._CLIENT_SETTINGS, subnets.ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(subnets, "_read_daemon_id", lambda key: "same-daemon")
    monkeypatch.setattr(subnets, "daemon_on_this_host", lambda name: True)
    monkeypatch.setattr(subnets, "host_reserved", lambda: [])
    yield
    subnets.reset_allocators()


async def daemon(args):
    return ExecResult(
        stdout=json.dumps({"name": "here", "pools": []}) if args[0] == "info" else "",
        stderr="",
        return_code=0,
    )


def endpoint_probe(monkeypatch, endpoint, expected_env):
    def inspect(args, **kwargs):
        assert args == [
            "docker",
            "context",
            "inspect",
            "--format",
            "{{json .Endpoints.docker.Host}}",
        ]
        for name in subnets._CLIENT_SETTINGS:
            assert kwargs["env"].get(name, "") == expected_env.get(name, "")
        assert kwargs["check"] and kwargs["timeout"] == 30
        return SimpleNamespace(stdout=json.dumps(endpoint), returncode=0)

    monkeypatch.setattr(subprocess, "run", inspect)


@pytest.mark.parametrize(
    "host",
    [
        "ssh://dummy@localhost",
        "tcp://localhost:2376",
        "http://localhost:2376",
        "https://localhost:2376",
        "  SSH://dummy@localhost",
    ],
)
@pytest.mark.parametrize(
    "socket",
    [
        "unix:///var/run/docker.sock",
        "unix:///run/user/1234/docker.sock",
    ],
)
async def test_resolved_local_endpoint_overrides_raw_remote_hint(
    monkeypatch, host, socket
):
    """Guards 849f4111: a resolved local endpoint permits local allocation."""
    remote_env = {"DOCKER_HOST": host}
    endpoint_probe(monkeypatch, "ssh://dummy@localhost", remote_env)
    assert await subnets.allocator_for(subnets.daemon_key(remote_env), daemon) is None
    local_env = dict(remote_env, DOCKER_CONTEXT="local-alias")
    endpoint_probe(monkeypatch, socket, local_env)
    allocator = await subnets.allocator_for(subnets.daemon_key(local_env), daemon)
    assert allocator is not None
    nets = [
        allocator.allocate([subnets.NetworkNeed("default")])["default"]
        for _ in range(40)
    ]
    assert len(set(nets)) == 40 and all(net.prefixlen == 28 for net in nets)
    assert (
        await subnets.allocator_for(subnets.daemon_key(remote_env), daemon) is allocator
    )


@pytest.mark.parametrize(
    "client",
    [
        {},
        {"DOCKER_HOST": "unix:///var/run/docker.sock"},
        {"DOCKER_CONTEXT": "named-alias", "DOCKER_HOST": "unix:///ignored.sock"},
        {"DOCKER_CONTEXT": "default", "DOCKER_HOST": "tcp://localhost:2376"},
        {"DOCKER_CONFIG": "/tmp/dummy-config"},
        {"HOME": "/tmp/dummy-home"},
        {
            "DOCKER_TLS": "1",
            "DOCKER_TLS_VERIFY": "1",
            "DOCKER_CERT_PATH": "/tmp/dummy-certs",
        },
    ],
)
@pytest.mark.parametrize(
    "endpoint",
    [
        "unix:///var/run/docker.sock",
        "ssh://dummy@localhost",
        "tcp://localhost:2376",
        "http://localhost:2376",
        "https://localhost:2376",
        "npipe:////./pipe/docker_engine",
        "fd://3",
        "future://localhost",
        "",
        "unix://",
        "unix://relative",
        None,
        {},
        [],
        1,
    ],
)
async def test_only_resolved_unix_socket_can_use_host_routes(
    monkeypatch, client, endpoint
):
    """Guards U14-C7/849f4111 across selectors, transport types and bad output."""
    # Ambient selectors must not enter another backend's endpoint probe.
    monkeypatch.setenv("DOCKER_CONTEXT", "ambient-ignored")
    monkeypatch.setenv("DOCKER_HOST", "ssh://dummy@ambient.invalid")
    endpoint_probe(monkeypatch, endpoint, client)
    calls = []
    monkeypatch.setattr(subnets, "host_reserved", lambda: calls.append(True) or [])
    allocator = await subnets.allocator_for(subnets.daemon_key(client), daemon)
    local = endpoint == "unix:///var/run/docker.sock"
    assert (allocator is not None) == local
    assert bool(calls) == local


@pytest.mark.parametrize("setting", ["", "172.20.0.0/16", "off", "daemon", "none"])
@pytest.mark.parametrize("failure", ["missing", "timeout", "refused", "malformed"])
async def test_unknown_endpoint_keeps_explicit_policy(
    monkeypatch, setting, failure, caplog
):
    """Guards U14-C7/849f4111: unknown never authorizes this host's routes."""
    monkeypatch.setenv(subnets.ENV_VAR, setting)

    def inspect(args, **kwargs):
        if failure == "missing":
            raise FileNotFoundError("dummy-private-value")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 30, stderr="dummy-private-value")
        if failure == "refused":
            raise subprocess.CalledProcessError(1, args, stderr="dummy-private-value")
        return SimpleNamespace(stdout="dummy-private-value", returncode=0)

    monkeypatch.setattr(subprocess, "run", inspect)

    def no_routes():
        pytest.fail("unknown endpoint read local routes")

    monkeypatch.setattr(subnets, "host_reserved", no_routes)
    allocator = await subnets.allocator_for(subnets.daemon_key({}), daemon)
    assert (allocator is not None) == (setting == "172.20.0.0/16")
    if allocator is not None:
        assert allocator.allocate([subnets.NetworkNeed("default")])
    assert "dummy-private-value" not in caplog.text


@pytest.mark.parametrize("refresh", [False, True])
async def test_remote_then_local_context_backend_overrides(
    tmp_path, monkeypatch, refresh
):
    """Guards U14-C7/849f4111 on 40 concurrent starts and refresh/retry."""
    host = "ssh://dummy@localhost"
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", host)
    monkeypatch.setenv("DOCKER_HOST", host)
    monkeypatch.setenv("DOCKER_CONTEXT", "local-alias")
    monkeypatch.setattr(
        DockerSandbox, "_docker_cli_unchecked", lambda self, args: daemon(args)
    )
    boxes = []

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
        boxes.append(box)
        return box

    try:
        remote = sandbox(RemoteDockerSandbox, "remote")
        endpoint_probe(monkeypatch, host, {"DOCKER_HOST": host})
        await remote._choose_subnets()
        assert remote._subnets_compose_path is None
        endpoint_probe(
            monkeypatch,
            "unix:///var/run/docker.sock",
            {
                "DOCKER_HOST": host,
                "DOCKER_CONTEXT": "local-alias",
            },
        )
        locals_ = [sandbox(DockerSandbox, f"local-{i}") for i in range(40)]
        await asyncio.gather(*(box._choose_subnets(refresh=refresh) for box in locals_))
        nets = []
        for box in locals_:
            assert box._subnets_compose_path is not None
            doc = json.loads(box._subnets_compose_path.read_text())
            nets.append(
                ipaddress.IPv4Network(
                    doc["networks"]["default"]["ipam"]["config"][0]["subnet"]
                )
            )
        assert len(set(nets)) == 40 and all(net.prefixlen == 28 for net in nets)
        await remote._choose_subnets(refresh=True)
        assert remote._subnet_allocator is locals_[0]._subnet_allocator
        assert remote._subnets["default"] not in nets
    finally:
        for box in boxes:
            box._release_subnets()
