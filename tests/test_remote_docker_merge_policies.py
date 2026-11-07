"""RD transport must preserve U10/U14's integration policies."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from benchflow.sandbox import _docker_subnets as subnets
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.remote_docker import RemoteDockerSandbox
from tests import _fake_docker_cli
from tests.test_remote_docker_isolation import _certs, _sandbox
from tests.test_teardown_confirmation import InventoryDaemon


@pytest.mark.parametrize(
    "retained", ["operator/dummy:keep", "benchflow-recovery-lease:dummy"]
)
async def test_remote_cleanup_preserves_sole_foreign_reference(
    tmp_path, monkeypatch, retained
):
    """Guards 9a8e776e: a project label never owns a foreign tag."""
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy@remote.invalid")
    sandbox = _sandbox(tmp_path)
    daemon = InventoryDaemon(sandbox.compose_project_name)
    sandbox._docker_cli = daemon.cli
    daemon.built("sha256:dummy", retained, "main")
    await sandbox._remove_built_images()
    assert daemon._find(retained) == "sha256:dummy"
    assert daemon.parent_present


@pytest.mark.parametrize("recovery", [False, True])
async def test_remote_stop_uses_shared_cleanup_eligibility(
    tmp_path, monkeypatch, recovery
):
    """Guards 9a8e776e: recovery never releases its parent's build."""
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", "ssh://dummy@remote.invalid")
    sandbox = _sandbox(tmp_path)
    sandbox._host_checked = True
    sandbox._recovery_baseline = object() if recovery else None
    sandbox._project_leftovers = AsyncMock(return_value={})
    sandbox._remove_built_images = AsyncMock()
    monkeypatch.setattr(DockerSandbox, "stop", AsyncMock())
    await RemoteDockerSandbox.stop(sandbox, delete=True)
    assert sandbox._remove_built_images.await_count == (0 if recovery else 1)


@pytest.mark.parametrize(
    "endpoint", ["ssh://dummy@remote.invalid", "tcp://remote.invalid:2376"]
)
def test_remote_admission_key_selects_explicit_endpoint(
    tmp_path, monkeypatch, endpoint
):
    """Guards 9a8e776e: parent/worker probes select the trial's host."""
    monkeypatch.setenv("BENCHFLOW_REMOTE_DOCKER_HOST", endpoint)
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/dummy-local.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "dummy-local-context")
    if endpoint.startswith("tcp:"):
        monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
        monkeypatch.setenv("DOCKER_CERT_PATH", str(_certs(tmp_path / "dummy-certs")))
    fake = _fake_docker_cli.install(tmp_path, monkeypatch)
    key = subnets.backend_daemon_key("remote-docker")
    assert key.endpoint_hint == endpoint
    assert "DOCKER_HOST" not in key.env
    assert "DOCKER_CONTEXT" not in key.env
    sandbox = _sandbox(tmp_path)
    assert sandbox._docker_subnet_key().endpoint_hint == endpoint

    assert json.loads(subnets._read_daemon_id(key)) == fake.script["info"]
    assert fake.calls()[-1]["argv"] == ["info", "--format", "{{.ID}}"]
    assert fake.calls()[-1]["host"] == endpoint
    assert fake.calls()[-1]["DOCKER_HOST"] is None
    key.bind("dummy-daemon")
    assert not subnets._uses_local_socket(key)
