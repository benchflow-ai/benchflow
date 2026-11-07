"""Guards FF-sw-evaluation snapshot/readiness fixes at 363d815b.

The shell adapter executes real commands only in the builder's labeled container.
"""

import asyncio

import pytest

from benchflow.environment.manifest import (
    EnvironmentManifest,
    Readiness,
    ServiceSpec,
)
from benchflow.environment.manifest_env import (
    ManifestEnvironment,
)
from benchflow.sandbox.protocol import ExecResult


class ShellSandbox:
    async def exec(self, command, *, timeout_sec=120):
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_sec)
        return ExecResult(
            return_code=proc.returncode, stdout=stdout.decode(), stderr=stderr.decode()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("owns", [False, True])
@pytest.mark.parametrize("probe", ["http", "tcp", "both"])
async def test_explicit_readiness_probes_are_mandatory(owns, probe):
    calls = []

    class Sandbox:
        async def exec(self, command, **kwargs):
            calls.append(command)
            return ExecResult(return_code=1, stdout="", stderr="not ready")

    services = [] if owns else [ServiceSpec(name="svc", command="svc", port=9001)]
    manifest = EnvironmentManifest(
        name="fixture",
        image="fixture",
        owns_lifecycle=owns,
        services=services,
        readiness=Readiness(
            http=["http://localhost:9001/ready"] if probe != "tcp" else [],
            tcp=[9002] if probe != "http" else [],
            timeout_sec=0,
        ),
    )
    env = ManifestEnvironment(manifest, sandbox=Sandbox())
    env._started = services
    result = await env.readiness()
    assert result.ready is False
    assert result.checked[0] == (
        "tcp://localhost:9002" if probe == "tcp" else "http://localhost:9001/ready"
    )


@pytest.mark.asyncio
async def test_tcp_probe_observes_actual_listener():
    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    manifest = EnvironmentManifest(
        name="fixture", image="fixture", readiness=Readiness(tcp=[port], timeout_sec=0)
    )
    env = ManifestEnvironment(manifest, sandbox=ShellSandbox())
    async with server:
        assert (await env.readiness()).ready
    result = await env.readiness()
    assert not result.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("owns", [False, True])
async def test_http_success_cannot_hide_failed_tcp_probe(owns):
    class Sandbox:
        async def exec(self, command, **kwargs):
            return ExecResult(
                return_code=0 if "curl" in command else 1, stdout="", stderr=""
            )

    services = [] if owns else [ServiceSpec(name="svc", command="svc", port=9001)]
    manifest = EnvironmentManifest(
        name="fixture",
        image="fixture",
        owns_lifecycle=owns,
        services=services,
        readiness=Readiness(
            http=["http://localhost:9001/ready"], tcp=[9002], timeout_sec=0
        ),
    )
    env = ManifestEnvironment(manifest, sandbox=Sandbox())
    env._started = services
    result = await env.readiness()
    assert not result.ready
    assert result.checked == ["http://localhost:9001/ready", "tcp://localhost:9002"]
