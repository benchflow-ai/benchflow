"""The ``remote-docker`` sandbox provider: tasks on a Docker host the user controls.

Every docker and compose call goes to the configured host (``DOCKER_HOST``
set explicitly, the caller's docker context ignored), no host path is
bind-mounted on the remote machine, an unreachable or undersized host is a
clear startup error, and teardown leaves no container, network or volume of
the rollout behind. The fake ``docker`` CLI (``tests/_fake_docker_cli.py``)
records what each call saw.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from benchflow.sandbox.protocol import SandboxStartupError
from benchflow.sandbox.remote_docker import (
    REMOTE_DOCKER_HOST_ENV,
    RemoteDockerConfigError,
    RemoteDockerSandbox,
    check_capacity,
    probe_remote_docker,
    resolve_remote_docker_host,
)
from benchflow.task.config import SandboxConfig
from benchflow.task.paths import RolloutPaths
from tests import _fake_docker_cli

SECRET_USER = "tok3nlikeuser9f8e7d6c5b4a"
SSH_URL = f"ssh://{SECRET_USER}@builder.example.test:2222"
_SSH_UNREACHABLE = (
    'error during connect: Get "http://docker.example.com/v1.55/info": command '
    f"[ssh -l {SECRET_USER} -p 2222 -o ConnectTimeout=30 -T -- builder.example.test "
    "docker system dial-stdio] has exited with exit status 255, make sure the URL "
    "is valid, and Docker 18.09 or later is installed on the remote host: "
    f"stderr={SECRET_USER}@builder.example.test: Permission denied (publickey)."
)


@pytest.fixture(autouse=True)
def _clean_docker_env(monkeypatch):
    for name in (
        REMOTE_DOCKER_HOST_ENV,
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return _fake_docker_cli.install(tmp_path, monkeypatch)


def _certs(root: Path, *names: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_text("pem\n")
    return root


# --- host selection ----------------------------------------------------------


def test_host_comes_from_benchflow_variable_before_docker_host(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://other@elsewhere.test")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    host = resolve_remote_docker_host()
    assert host.url == SSH_URL
    assert host.display == "ssh://***@builder.example.test:2222"


def test_host_falls_back_to_docker_host(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "ssh://builder.example.test")
    assert resolve_remote_docker_host().url == "ssh://builder.example.test"
    assert resolve_remote_docker_host().display == "ssh://builder.example.test"


def test_missing_host_is_refused_naming_both_variables():
    with pytest.raises(RemoteDockerConfigError) as info:
        resolve_remote_docker_host()
    assert REMOTE_DOCKER_HOST_ENV in str(info.value)
    assert "DOCKER_HOST" in str(info.value)


@pytest.mark.parametrize(
    "url", ["unix:///var/run/docker.sock", "npipe:////./pipe/docker_engine"]
)
def test_local_socket_points_to_the_local_provider(monkeypatch, url):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, url)
    with pytest.raises(RemoteDockerConfigError, match="--sandbox docker"):
        resolve_remote_docker_host()


def test_unknown_scheme_is_refused(monkeypatch):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "https://builder.example.test")
    with pytest.raises(RemoteDockerConfigError, match="ssh://"):
        resolve_remote_docker_host()


def test_plain_tcp_without_tls_is_refused(monkeypatch):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "tcp://builder.example.test:2375")
    with pytest.raises(RemoteDockerConfigError, match="DOCKER_TLS_VERIFY=1"):
        resolve_remote_docker_host()


def test_tcp_tls_names_the_missing_certificate_file(monkeypatch, tmp_path):
    certs = _certs(tmp_path / "certs", "ca.pem", "cert.pem")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "tcp://builder.example.test:2376")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", str(certs))
    with pytest.raises(RemoteDockerConfigError, match=r"key\.pem"):
        resolve_remote_docker_host()


def test_tcp_tls_host_passes_the_tls_settings_to_the_cli(monkeypatch, tmp_path):
    certs = _certs(tmp_path / "certs", "ca.pem", "cert.pem", "key.pem")
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, "tcp://builder.example.test:2376")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    monkeypatch.setenv("DOCKER_CERT_PATH", str(certs))
    monkeypatch.setenv("DOCKER_CONTEXT", "some-local-context")
    env = resolve_remote_docker_host().client_env()
    assert env["DOCKER_HOST"] == "tcp://builder.example.test:2376"
    assert env["DOCKER_TLS_VERIFY"] == "1"
    assert env["DOCKER_CERT_PATH"] == str(certs)
    assert "DOCKER_CONTEXT" not in env


# --- reachability and capacity ------------------------------------------------


def test_probe_reads_cpu_and_memory(monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    capacity = probe_remote_docker(resolve_remote_docker_host())
    assert (capacity.cpus, capacity.memory_mb) == (4, 8192)
    (call,) = fake.calls()
    assert call["argv"][0] == "info"
    assert call["DOCKER_HOST"] == SSH_URL


def test_unreachable_host_is_a_startup_error_without_the_ssh_user(monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(fail={"info": _SSH_UNREACHABLE})
    with pytest.raises(SandboxStartupError) as info:
        probe_remote_docker(resolve_remote_docker_host())
    message = str(info.value)
    assert message.startswith(
        "Remote Docker host unreachable: ssh://***@builder.example.test:2222"
    )
    assert "Permission denied (publickey)" in message
    assert SECRET_USER not in message
    assert SECRET_USER not in info.value.diagnostic.raw_message


def test_capacity_refuses_a_task_larger_than_the_host():
    from benchflow.sandbox.remote_docker import HostCapacity

    capacity = HostCapacity(cpus=4, memory_mb=8192, server_version="28.3.3")
    check_capacity(capacity, cpus=4, memory_mb=8192, host_display="ssh://h")
    with pytest.raises(SandboxStartupError, match="needs 8 CPUs; ssh://h has 4"):
        check_capacity(capacity, cpus=8, memory_mb=1024, host_display="ssh://h")
    with pytest.raises(
        SandboxStartupError, match="needs 16384 MB of memory; ssh://h has 8192 MB"
    ):
        check_capacity(capacity, cpus=1, memory_mb=16384, host_display="ssh://h")


def test_remote_startup_errors_are_not_retried():
    from benchflow.evaluation import RetryConfig

    for message in (
        "Sandbox startup failed: Remote Docker host unreachable: ssh://h: refused",
        "Sandbox startup failed: Remote Docker host lacks resources: task needs 8 CPUs",
    ):
        assert RetryConfig().should_retry(message) is False


# --- the sandbox --------------------------------------------------------------


def _sandbox(tmp_path: Path, **config) -> RemoteDockerSandbox:
    environment = tmp_path / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    return RemoteDockerSandbox(
        environment_dir=environment,
        environment_name="remote-task",
        session_id="remote-task__abc123",
        rollout_paths=paths,
        task_env_config=SandboxConfig(**config),
    )


async def test_every_call_targets_the_remote_host_and_mounts_nothing(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    monkeypatch.setenv("DOCKER_CONTEXT", "some-local-context")
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=False)
    await sandbox.exec("echo hi")
    await sandbox.stop(delete=True)

    calls = fake.calls()
    assert calls
    assert {c["DOCKER_HOST"] for c in calls} == {SSH_URL}
    assert {c["DOCKER_CONTEXT"] for c in calls} == {None}
    files = [a for argv in fake.argvs() for a in argv if a.endswith(".yaml")]
    assert any(f.endswith("docker-compose-remote-base.yaml") for f in files)
    assert not any(f.endswith("docker-compose-base.yaml") for f in files)
    assert sandbox.is_mounted is False
    execs = [" ".join(a) for a in fake.argvs() if "exec" in a]
    assert any(
        "mkdir -p /logs/agent /logs/verifier /logs/artifacts" in e for e in execs
    )
    # No host path was sent to the remote daemon as a mount probe.
    assert not any(".benchflow-mount-probe" in e for e in execs)


def test_remote_base_compose_file_has_no_bind_mounts_and_keeps_limits():
    import yaml

    from benchflow.sandbox._compose import COMPOSE_REMOTE_BASE_PATH

    doc = yaml.safe_load(COMPOSE_REMOTE_BASE_PATH.read_text())
    main = doc["services"]["main"]
    assert "volumes" not in main
    assert main["labels"] == {"benchflow.owned": "true"}
    assert main["deploy"]["resources"]["limits"] == {
        "cpus": "${CPUS}",
        "memory": "${MEMORY}",
    }
    assert doc["networks"]["default"]["labels"] == {"benchflow.owned": "true"}


async def test_no_network_task_keeps_the_network_none_overlay(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path, allow_internet=False)
    names = [p.name for p in sandbox._docker_compose_paths]
    assert names[0] == "docker-compose-remote-base.yaml"
    assert "docker-compose-no-network.yaml" in names


async def test_stop_without_delete_still_removes_volumes_and_orphans(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=False)
    await sandbox.stop(delete=False)
    downs = [a for a in fake.argvs() if "down" in a and "-t" in a]
    assert downs
    assert "--volumes" in downs[-1]
    assert "--remove-orphans" in downs[-1]


async def test_failed_down_force_removes_everything_with_the_project_label(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=False)
    fake.set(
        fail={"down": "Error response from daemon: removal already in progress"},
        leftovers={
            "containers": ["c1", "c2"],
            "networks": ["n1"],
            "volumes": ["v1"],
        },
    )
    await sandbox.stop(delete=True)
    assert fake.script["leftovers"] == {"containers": [], "networks": [], "volumes": []}
    label = "label=com.docker.compose.project=remote-task__abc123"
    listed = [a for a in fake.argvs() if label in a]
    assert any(a[:2] == ["volume", "ls"] for a in listed)
    assert {c["DOCKER_HOST"] for c in fake.calls()} == {SSH_URL}


async def test_unreachable_host_at_teardown_names_the_cleanup_command(
    tmp_path, monkeypatch, fake, caplog
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path)
    await sandbox.start(force_build=False)
    fake.set(
        fail={
            "down": _SSH_UNREACHABLE,
            "ps": _SSH_UNREACHABLE,
            "network": _SSH_UNREACHABLE,
            "volume": _SSH_UNREACHABLE,
            "exec": _SSH_UNREACHABLE,
        }
    )
    with caplog.at_level(logging.WARNING, logger="benchflow"):
        await sandbox.stop(delete=True)
    text = caplog.text
    assert "remote-task__abc123" in text
    assert "docker compose -p remote-task__abc123 down --volumes" in text
    assert SECRET_USER not in text


async def test_unreachable_host_fails_start_before_any_compose_call(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(fail={"info": _SSH_UNREACHABLE})
    sandbox = _sandbox(tmp_path)
    with pytest.raises(SandboxStartupError, match="Remote Docker host unreachable"):
        await sandbox.start(force_build=False)
    assert [a[0] for a in fake.argvs()] == ["info"]
    # Nothing was started, so teardown makes no compose call either.
    await sandbox.stop(delete=True)
    assert not any(a[0] == "compose" and "down" in a for a in fake.argvs())


async def test_oversized_task_fails_start_before_any_compose_call(
    tmp_path, monkeypatch, fake
):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    sandbox = _sandbox(tmp_path, cpus=16)
    with pytest.raises(SandboxStartupError, match="lacks resources"):
        await sandbox.start(force_build=False)
    assert [a[0] for a in fake.argvs()] == ["info"]


async def test_compose_errors_never_show_the_ssh_user(tmp_path, monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(fail={"up": _SSH_UNREACHABLE})
    sandbox = _sandbox(tmp_path)
    with pytest.raises(RuntimeError) as info:
        await sandbox.start(force_build=False)
    assert SECRET_USER not in str(info.value)


async def test_full_remote_disk_is_named(tmp_path, monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    fake.set(
        fail={
            "build": "failed to solve: write /var/lib/docker/tmp/x: "
            "no space left on device"
        }
    )
    sandbox = _sandbox(tmp_path)
    with pytest.raises(RuntimeError, match="remote Docker host is out of disk space"):
        await sandbox.start(force_build=True)


def test_host_mounts_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    with pytest.raises(RemoteDockerConfigError, match="host mounts"):
        RemoteDockerSandbox(
            environment_dir=environment,
            environment_name="t",
            session_id="t__1",
            rollout_paths=None,
            task_env_config=SandboxConfig(),
            mounts_json=[{"type": "bind", "source": "/data", "target": "/data"}],
        )


async def test_live_process_uses_the_remote_host(tmp_path, monkeypatch, fake):
    monkeypatch.setenv(REMOTE_DOCKER_HOST_ENV, SSH_URL)
    monkeypatch.setenv("DOCKER_CONTEXT", "some-local-context")
    sandbox = _sandbox(tmp_path)
    process = await sandbox.live_process()
    env = process._host_env()
    assert env["DOCKER_HOST"] == SSH_URL
    assert "DOCKER_CONTEXT" not in env
    # The local docker context is never consulted for a remote sandbox.
    assert not any(a[:1] == ["context"] for a in fake.argvs())
