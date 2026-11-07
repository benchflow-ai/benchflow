"""Guards RD-CONF-2/5 against transport inference at ccba96d7."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _docker_subnets as subnets
from benchflow.sandbox import env_layers
from benchflow.sandbox.remote_docker import RemoteDockerHost


@pytest.fixture(autouse=True)
def reset_before_tip_caches():
    # The reviewed archive has a global ID cache; isolate each behavioral proof.
    getattr(env_layers.DockerLayers, "_daemon_ids", {}).clear()
    yield
    getattr(env_layers.DockerLayers, "_daemon_ids", {}).clear()


@pytest.mark.parametrize("first_id,second_id", [("A", "B"), ("B", "A")])
async def test_same_alias_different_ssh_clients_query_their_own_daemon(
    first_id, second_id
):
    """Guards RD-CONF-2/ccba96d7: a transport tuple cannot stand in for server ID."""
    # Both configs legitimately resolve builder, but select different daemons.
    env = {"DOCKER_HOST": "ssh://builder", "HOME": "/tmp/dummy-home"}
    first = env_layers.DockerLayers(env=env)
    second = env_layers.DockerLayers(env=env)
    first._docker = AsyncMock(return_value=env_layers._Run(first_id, "", 0))
    second._docker = AsyncMock(return_value=env_layers._Run(second_id, "", 0))
    assert await first.daemon_id() == first_id
    assert await second.daemon_id() == second_id
    assert await first.daemon_id() == first_id
    first._docker.assert_awaited_once()
    second._docker.assert_awaited_once()


@pytest.mark.parametrize(
    "endpoint", ["ssh://builder", "ssh://dummy@builder:2222", "tcp://builder:2376"]
)
def test_ordinary_identity_probe_keeps_originating_transport(monkeypatch, endpoint):
    """Guards RD-CONF-5/ccba96d7: ordinary Docker never inherits RD's SSH config."""
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_SSH_CONFIG", "/tmp/dummy-rd-config")
    env = {
        "DOCKER_HOST": endpoint,
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": "/tmp/dummy-certs",
    }
    monkeypatch.setattr(
        RemoteDockerHost,
        "client_env",
        lambda self, base=None: {"DUMMY_RD_TRANSPORT": "yes"},
    )
    seen = []

    def run(args, **kwargs):
        seen.append(kwargs["env"])
        return SimpleNamespace(
            stdout="B" if kwargs["env"].get("DUMMY_RD_TRANSPORT") else "A", returncode=0
        )

    monkeypatch.setattr(subnets.subprocess, "run", run)
    assert subnets._read_daemon_id(subnets.daemon_key(env)) == "A"
    assert seen[-1]["DOCKER_HOST"] == endpoint


@pytest.mark.parametrize("scheme", ["ssh", "tcp"])
@pytest.mark.parametrize("phase", ["spawn", "communicate", "cleanup"])
@pytest.mark.parametrize("exception", [RuntimeError, OSError])
async def test_layer_exceptions_cross_the_remote_redaction_boundary(
    monkeypatch, scheme, phase, exception
):
    """Guards the RD-CONF-1 exception sibling at ccba96d7, before fallback slicing."""
    import asyncio

    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = RemoteDockerHost(f"{scheme}://builder.internal.test")
    diagnostic = (
        'Get "https://builder.internal.test:2376/info": tls: '
        if scheme == "tcp"
        else "command [ssh dummy] has exited: "
    ) + "private.internal.test " * 80
    error = exception(diagnostic)
    if phase == "spawn":
        spawn = AsyncMock(side_effect=error)
    else:
        proc = SimpleNamespace(pid=99999999, communicate=AsyncMock(side_effect=error))
        if phase == "cleanup":
            proc.communicate.side_effect = [TimeoutError(), error]
            monkeypatch.setattr(env_layers.os, "killpg", lambda *args: None)
        spawn = AsyncMock(return_value=proc)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(Exception) as caught:
        await sandbox._docker_layers()._docker("pull", "dummy/image")
    assert "private.internal.test" not in str(caught.value)
    assert "builder.internal.test" not in str(caught.value)
    assert str(caught.value) == sandbox.docker_host.redact(diagnostic)


CASES = [
    ("docker", "unix:///tmp/dummy.sock", "default"),
    ("docker", "ssh://builder", "default"),
    ("remote-docker", "ssh://builder", "config-a"),
    ("remote-docker", "ssh://builder", "config-b"),
    ("remote-docker", "ssh://other-host", "config-a"),
    ("docker", "tcp://builder:2376", "tls-a"),
    ("remote-docker", "tcp://builder:2376", "tls-b"),
]


@pytest.mark.parametrize("first", CASES)
@pytest.mark.parametrize("second", CASES)
@pytest.mark.parametrize("same_server", [False, True])
async def test_all_consumers_share_queried_identity_not_transport_settings(
    tmp_path, monkeypatch, first, second, same_server
):
    """Guards RD-CONF-2/5 at ccba96d7: two configs, two hosts, local/SSH/TLS aliases."""
    import ipaddress
    import json
    from unittest.mock import Mock

    from benchflow.sandbox import _layer_lock, _pull_provenance
    from benchflow.sandbox._base import ExecResult
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    subnets.reset_allocators()
    env_layers.DockerLayers._claims.clear()
    monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(tmp_path / "locks"))
    monkeypatch.setenv(subnets.ENV_VAR, "172.20.0.0/16")
    monkeypatch.setattr(subnets, "host_reserved", lambda: [])
    monkeypatch.setattr(subnets, "daemon_on_this_host", lambda name: True)
    # These stand-in client environments represent the actual provider boundary,
    # including the selected SSH config; no daemon is inferred from a URL.
    monkeypatch.setattr(
        RemoteDockerHost,
        "client_env",
        lambda self: {
            "DUMMY_CONFIG": config,
            "DUMMY_SERVER": server,
        },
    )
    monkeypatch.setattr(
        subnets.subprocess,
        "run",
        lambda args, **kwargs: SimpleNamespace(
            stdout=json.dumps(kwargs["env"].get("DOCKER_HOST", "")),
            returncode=0,
        ),
    )
    clients = []
    probes = []
    for case, server in zip(
        (first, second),
        ("daemon-A", "daemon-A" if same_server else "daemon-B"),
        strict=True,
    ):
        backend, endpoint, config = case
        if backend == "docker":
            sandbox = object.__new__(DockerSandbox)
            sandbox._docker_client_env = (
                lambda endpoint=endpoint, config=config, server=server: {
                    "DOCKER_HOST": endpoint,
                    "DUMMY_CONFIG": config,
                    "DUMMY_SERVER": server,
                }
            )
        else:
            sandbox = object.__new__(RemoteDockerSandbox)
            sandbox.docker_host = RemoteDockerHost(endpoint)
        layers = sandbox._docker_layers()
        probe = AsyncMock(return_value=env_layers._Run(server, "", 0))
        layers._docker = probe
        assert await layers.daemon_id() == server
        again = sandbox._docker_layers()
        again._docker = AsyncMock(side_effect=AssertionError("identity queried twice"))
        assert await again.daemon_id() == server
        assert layers._daemon is sandbox._docker_subnet_key() is again._daemon
        assert layers._daemon.identity().server_id == server
        assert layers._daemon.identity().endpoint == endpoint
        # Successful machine data stays exact; only opaque identity is serialized.
        assert server not in subnets.daemon_identity(sandbox._docker_subnet_key())
        clients.append((sandbox, layers))
        probes.append(probe)

    async def networks(args):
        return ExecResult(
            stdout=json.dumps({"name": "same-host", "pools": []})
            if args[0] == "info"
            else "",
            stderr="",
            return_code=0,
        )

    pin = ipaddress.IPv4Network("172.20.0.0/28")
    subnets.pin_subnets([pin], key=clients[0][0]._docker_subnet_key())
    allocators = [
        await subnets.allocator_for(sandbox._docker_subnet_key(), networks)
        for sandbox, _ in clients
    ]
    assert (allocators[0] is allocators[1]) == same_server
    assert (
        bool(subnets.pinned_spans(key=clients[1][0]._docker_subnet_key()))
        == same_server
    )
    key = "a" * 64
    locks = []
    record = Mock()
    monkeypatch.setattr(_pull_provenance, "record_pull", record)
    for _sandbox, layers in clients:
        layers.tag_times_per_name = AsyncMock(return_value=False)
        await layers.note_pulled("dummy/image", {"Id": "sha256:dummy"})
        plock, hlock, _ = await layers._lock_key("agent", key, 1)
        locks.append(plock)
        hlock.release()
        plock.release()
    assert (locks[0] is locks[1]) == same_server
    assert [call.args[0] for call in record.call_args_list] == [
        "daemon-A",
        "daemon-A" if same_server else "daemon-B",
    ]
    for probe in probes:
        probe.assert_awaited_once()
    subnets.reset_allocators()


async def test_claim_release_cannot_release_another_daemons_layer():
    """Guards the RD-CONF-2 sibling at ccba96d7: claims belong to queried servers."""
    key = "a" * 64
    env_layers.DockerLayers._claims.clear()
    env_layers.DockerLayers._claims.update({("A", key): 1, ("B", key): 1})
    env_layers.DockerLayers.release(key, daemon="A")
    assert ("A", key) not in env_layers.DockerLayers._claims
    assert env_layers.DockerLayers._claims[("B", key)] == 1
    env_layers.DockerLayers.release(key)  # Unknown ownership cannot free B.
    assert env_layers.DockerLayers._claims[("B", key)] == 1
    env_layers.DockerLayers._claims.clear()


@pytest.mark.parametrize(
    "method",
    ["_docker_cli", "_run_docker_compose_command", "_run_docker_container_command"],
)
@pytest.mark.parametrize("scheme", ["ssh", "tcp"])
@pytest.mark.parametrize("exception", [RuntimeError, OSError, ValueError])
async def test_all_provider_error_paths_use_one_redactor(
    monkeypatch, method, scheme, exception
):
    """Guards RD-CONF-1's exception sibling at ccba96d7 on raw, Compose and direct paths."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    host = RemoteDockerHost(f"{scheme}://builder.internal.test")
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = host
    diagnostic = (
        'Get "https://builder.internal.test:2376/info": tls: '
        if scheme == "tcp"
        else "command [ssh dummy] has exited: "
    ) + "private.internal.test"
    monkeypatch.setattr(
        DockerSandbox, method, AsyncMock(side_effect=exception(diagnostic))
    )
    with pytest.raises(Exception) as caught:
        await getattr(sandbox, method)(["info"])
    assert str(caught.value) == host.redact(diagnostic)
    assert "private.internal.test" not in str(caught.value)


@pytest.mark.parametrize("first", ["layers", "subnets"])
async def test_concurrent_consumers_query_only_once(monkeypatch, first):
    """Guards RD-CONF-2/5 at ccba96d7: concurrent layer and subnet reads share the result."""
    import asyncio
    from unittest.mock import Mock

    daemon = subnets.daemon_key({"DOCKER_HOST": "ssh://dummy"})
    layer_a = env_layers.DockerLayers(daemon=daemon)
    layer_b = env_layers.DockerLayers(daemon=daemon)
    query = AsyncMock(return_value=env_layers._Run("server-A", "", 0))
    layer_a._docker = layer_b._docker = query
    sync_query = Mock(return_value="server-A")
    monkeypatch.setattr(subnets, "_read_daemon_id", sync_query)
    readers = [layer_a.daemon_id(), layer_b.daemon_id(), daemon.identify()]
    if first == "subnets":
        readers.reverse()
    await asyncio.gather(*readers)
    assert query.await_count + sync_query.call_count == 1
    assert daemon.identity().server_id == "server-A"


async def test_layer_claims_are_isolated_during_pruning(tmp_path, monkeypatch):
    """Guards RD-CONF-2's claim sibling at ccba96d7 with two real layer consumers."""
    import inspect

    from benchflow.sandbox import _layer_lock
    from tests.test_env_layers_agent import RECIPE, FakeDocker

    monkeypatch.setenv(_layer_lock.LOCK_DIR_ENV, str(tmp_path / "locks"))
    env_layers.DockerLayers._claims.clear()
    env_layers.DockerLayers._libc_cache.clear()
    layers = []
    results = []
    for name in ("A", "B"):
        fake = FakeDocker()
        fake.add_image("dummy/task")
        client = env_layers.DockerLayers()
        client._docker = fake.run
        client.daemon_id = AsyncMock(return_value=name)
        result = await client.ensure_agent_layer(RECIPE, "dummy/task", prune=False)
        layers.append(client)
        results.append(result)
    # The pre-fix API has no daemon argument; exercise its real release too.
    kwargs = (
        {"daemon": "B"}
        if "daemon" in inspect.signature(env_layers.DockerLayers.release).parameters
        else {}
    )
    env_layers.DockerLayers.release(results[1].key, **kwargs)
    report = await layers[1].prune(keep=0, min_free=None)
    assert report.removed == [results[1].key]
    assert (await layers[0].prune(keep=0, min_free=None)).in_use == [results[0].key]
    env_layers.DockerLayers._claims.clear()
