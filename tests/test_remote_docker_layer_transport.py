"""Guards RD-CONF-1/2/3 at 39b56cbc: layers use the provider's transport boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.sandbox import env_layers
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerHost, RemoteDockerSandbox
from tests.test_env_layers_agent import RECIPE as AGENT_RECIPE
from tests.test_env_layers_proxy import RECIPE as PROXY_RECIPE
from tests.test_remote_docker_isolation import _certs, _sandbox


def layers_for(sandbox):
    # The reviewed archive predates the factory. Exercise its actual construction
    # there too, so the before proof fails on disclosure, never a missing API.
    factory = getattr(sandbox, "_docker_layers", None)
    return (
        factory()
        if factory
        else env_layers.DockerLayers(env=sandbox._docker_client_env())
    )


@pytest.mark.parametrize("scheme", ["ssh", "tcp"])
@pytest.mark.parametrize("show", [False, True])
@pytest.mark.parametrize("code", [0, 1, 125])
@pytest.mark.parametrize("command", ["pull", "build", "inspect", "info", "run", "tag"])
async def test_layer_runner_redacts_diagnostics_before_consumers(
    monkeypatch, scheme, show, code, command
):
    """Guards RD-CONF-1: all layer commands share one diagnostic boundary."""
    host = RemoteDockerHost(f"{scheme}://builder.internal.test", show_address=show)
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = host
    if scheme == "tcp":
        diagnostic = 'Get "https://builder.internal.test:2376/v1/info": tls: failed to verify certificate: x509: private.internal.test\n'
    else:
        diagnostic = "command [ssh dummy] has exited: failed to open the raw stream connection: private.internal.test\nunknown continuation private.internal.test\n"
    data = json.dumps({"Name": "builder.internal.test", "ID": "private.internal.test"})
    stdout = data if code == 0 else diagnostic
    proc = SimpleNamespace(
        returncode=code,
        communicate=AsyncMock(return_value=(stdout.encode(), diagnostic.encode())),
    )
    monkeypatch.setattr(
        env_layers.asyncio, "create_subprocess_exec", AsyncMock(return_value=proc)
    )
    result = await layers_for(sandbox)._docker(command)
    assert result.stdout == (data if code == 0 else host.redact(diagnostic))
    assert result.stderr == host.redact(diagnostic)
    if not show:
        assert "private.internal.test" not in result.stderr
        if code:
            assert "private.internal.test" not in result.stdout


@pytest.mark.parametrize("scheme", ["ssh", "tcp"])
@pytest.mark.parametrize("kind", ["agent", "proxy"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
async def test_layer_fallback_logs_and_records_redacted_reason(
    tmp_path, monkeypatch, caplog, scheme, kind, stream
):
    """Guards RD-CONF-1: redaction precedes pull helpers' tail slicing."""
    monkeypatch.setenv(
        "BENCHFLOW_REMOTE_DOCKER_HOST", f"{scheme}://builder.internal.test"
    )
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS", "0")
    if scheme == "tcp":
        monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
        monkeypatch.setenv("DOCKER_CERT_PATH", str(_certs(tmp_path / "dummy-certs")))
    sandbox = _sandbox(tmp_path, docker_image="dummy/task:latest")
    sandbox._use_prebuilt = True
    if kind == "agent":
        sandbox.use_agent_layer(AGENT_RECIPE)
    else:
        sandbox.use_proxy_layer(PROXY_RECIPE)
    diagnostic = (
        'Get "https://builder.internal.test:2376/v1/info": tls: failed to verify certificate: x509: '
        if scheme == "tcp"
        else "command [ssh dummy] has exited: "
    ) + "private.internal.test " * 100

    async def spawn(*args, **kwargs):
        text = "No such image: dummy/task:latest" if args[1] == "image" else diagnostic
        out, err = (text, "") if stream == "stdout" else ("", text)
        return SimpleNamespace(
            returncode=1,
            communicate=AsyncMock(return_value=(out.encode(), err.encode())),
        )

    monkeypatch.setattr(env_layers.asyncio, "create_subprocess_exec", spawn)
    await sandbox._start_from_layers()
    record = sandbox.layer_records[kind]
    assert record["status"] == "fresh"
    assert "private.internal.test" not in record["reason"]
    assert "private.internal.test" not in caplog.text
    assert sandbox.docker_host.display in record["reason"]


async def test_local_layer_output_keeps_original_data_and_diagnostics(monkeypatch):
    """Guards RD-CONF-1: ordinary Docker keeps the caller's diagnostics."""
    text = b'{"Name":"builder.internal.test"}'
    proc = SimpleNamespace(
        returncode=1, communicate=AsyncMock(return_value=(text, text))
    )
    monkeypatch.setattr(
        env_layers.asyncio, "create_subprocess_exec", AsyncMock(return_value=proc)
    )
    result = await layers_for(object.__new__(DockerSandbox))._docker("info")
    assert result.stdout == result.stderr == text.decode()


@pytest.mark.parametrize(
    "first,second",
    [
        (a, b)
        for a in ("local", "ssh", "tcp")
        for b in ("local", "ssh", "tcp")
        if (a, b) != ("local", "local")
    ],
)
@pytest.mark.parametrize("same_daemon", [False, True])
async def test_layer_daemon_cache_is_scoped_by_effective_connection(
    monkeypatch, first, second, same_daemon
):
    """Guards RD-CONF-2: shim environments cannot merge distinct endpoints."""
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"):
        monkeypatch.delenv(name, raising=False)

    def client(scheme, endpoint):
        if scheme == "local":
            sandbox = object.__new__(DockerSandbox)
        else:
            sandbox = object.__new__(RemoteDockerSandbox)
            sandbox.docker_host = RemoteDockerHost(
                f"{scheme}://{endpoint}.internal.test"
            )
        return layers_for(sandbox)

    a, b = client(first, "first"), client(second, "second")
    b_id = "daemon-A" if same_daemon else "daemon-B"
    a._docker = AsyncMock(return_value=env_layers._Run("daemon-A\n", "", 0))
    b._docker = AsyncMock(return_value=env_layers._Run(b_id + "\n", "", 0))
    assert await a.daemon_id() == "daemon-A"
    assert await b.daemon_id() == b_id
    # Aliases of one daemon query independently, then use the same server ID
    # for layer locks and pull provenance. Their cache entries stay separate.
    b._docker.assert_awaited_once()
    assert await a.daemon_id() == "daemon-A"
    a._docker.assert_awaited_once()


@pytest.mark.parametrize(
    "setting", ["DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "DOCKER_TLS", "HOME"]
)
async def test_layer_daemon_cache_includes_connection_configuration(setting):
    """Guards RD-CONF-2: certificates and the config home can select another daemon."""
    a = env_layers.DockerLayers(env={setting: "dummy-a"})
    b = env_layers.DockerLayers(env={setting: "dummy-b"})
    a._docker = AsyncMock(return_value=env_layers._Run("daemon-A\n", "", 0))
    b._docker = AsyncMock(return_value=env_layers._Run("daemon-B\n", "", 0))
    assert await a.daemon_id() == "daemon-A"
    assert await b.daemon_id() == "daemon-B"
    b._docker.assert_awaited_once()


@pytest.mark.parametrize("scheme", ["ssh", "tcp"])
async def test_remote_disk_floor_never_reads_client_filesystem(monkeypatch, scheme):
    """Guards RD-CONF-3: hostname equality cannot establish transport locality."""
    from benchflow.sandbox import _docker_subnets as subnets

    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = RemoteDockerHost(f"{scheme}://same-host")
    layers = layers_for(sandbox)

    async def docker(*args, **kwargs):
        return env_layers._Run(
            "dummy-daemon" if args[-1] == "{{.ID}}" else "/tmp/dummy-root\tsame-host\n",
            "",
            0,
        )

    layers._docker = AsyncMock(side_effect=docker)
    monkeypatch.setattr(env_layers.socket, "gethostname", lambda: "same-host")
    monkeypatch.setattr(env_layers.os.path, "isdir", lambda path: True)
    disk = Mock(return_value=SimpleNamespace(free=1))
    monkeypatch.setattr(env_layers.shutil, "disk_usage", disk)
    monkeypatch.setattr(
        subnets.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout='"unix:///tmp/dummy.sock"'),
    )
    assert await layers._free_bytes() is None
    disk.assert_not_called()


@pytest.mark.parametrize("context", ["", "default", "selected"])
@pytest.mark.parametrize(
    "endpoint",
    [
        "unix:///tmp/dummy.sock",
        "ssh://dummy@same-host",
        "tcp://same-host:2376",
        "https://same-host",
        "npipe://dummy",
        "unknown",
    ],
)
async def test_layer_disk_locality_uses_resolved_context(
    monkeypatch, context, endpoint
):
    """Guards RD-CONF-3: only a resolved Unix transport admits host disk facts."""
    from benchflow.sandbox import _docker_subnets as subnets

    layers = env_layers.DockerLayers(env={"DOCKER_CONTEXT": context})

    async def docker(*args, **kwargs):
        return env_layers._Run(
            "dummy-daemon" if args[-1] == "{{.ID}}" else "/tmp/dummy-root\tsame-host\n",
            "",
            0,
        )

    layers._docker = AsyncMock(side_effect=docker)
    monkeypatch.setattr(env_layers.socket, "gethostname", lambda: "same-host")
    monkeypatch.setattr(env_layers.os.path, "isdir", lambda path: True)
    disk = Mock(return_value=SimpleNamespace(free=123))
    monkeypatch.setattr(env_layers.shutil, "disk_usage", disk)
    monkeypatch.setattr(
        subnets.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(endpoint)),
    )
    expected = 123 if endpoint.startswith("unix:///") else None
    assert await layers._free_bytes() == expected
    assert disk.call_count == (1 if expected else 0)
