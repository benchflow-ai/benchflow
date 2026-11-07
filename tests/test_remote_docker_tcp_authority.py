"""Docker must dial the authority we redact."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.diagnostics import TransportClosedError
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.process import DockerProcess
from benchflow.sandbox.process._base import _STDERR_TAIL_LIMIT
from benchflow.sandbox.remote_docker import (
    RemoteDockerSandbox,
    resolve_remote_docker_host,
)
from tests.test_remote_docker_tls_output import _drain

_HOSTS = [
    "Docker-Host",
    "Docker.Example.test",
    "docker.example.test.",
    "192.0.2.44",
    "[2001:DB8::44]",
    "[::ffff:192.0.2.44]",
    "xn--dcker-jua.example.test",
    "XN--DCKER-JUA.Example.test.",
]
# Docker keeps explicit port digits, including leading zeros. A supplied
# TCP hostname defaults to 2375 even when TLS verification is enabled.
_PORTS = [
    ("", "2375"),
    (":", "2375"),
    (":2375", "2375"),
    (":2376", "2376"),
    (":02376", "02376"),
]


@pytest.fixture(
    params=[(host, written, port) for host in _HOSTS for written, port in _PORTS]
)
def endpoint(request, tmp_path):
    hostname, written_port, effective_port = request.param
    for name in ("ca.pem", "cert.pem", "key.pem"):
        (tmp_path / name).write_text("dummy TLS file; no connection is made\n")
    env = {
        "BENCHFLOW_REMOTE_DOCKER_HOST": f"tcp://{hostname}{written_port}",
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": str(tmp_path),
    }
    return env, f"{hostname}:{effective_port}"


@pytest.mark.parametrize("user", ["", "dummy-user@"])
@pytest.mark.parametrize("path", ["", "/proxy"])
def test_tcp_launch_and_masking_share_explicit_authority(endpoint, user, path):
    """Guards the fix at 4f6ea75f for all accepted host/port forms."""
    env, authority = endpoint
    env["BENCHFLOW_REMOTE_DOCKER_HOST"] = (
        env["BENCHFLOW_REMOTE_DOCKER_HOST"].replace("tcp://", f"tcp://{user}") + path
    )
    host = resolve_remote_docker_host(env)
    expected = f"tcp://{authority}{path}"
    assert host.docker_args()[1] == expected
    assert host.compose_args({})[1] == expected
    said = f'Get "https://{authority}/v1.47/info": tls: failed to verify certificate: private.internal.test\n'
    assert "private.internal.test" not in host.redact(said, command_output=True)


@pytest.mark.parametrize("consumer", ["compose", "raw"])
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize(
    "method", ["Get", "Head", "Post", "Put", "Delete", "Patch", "Options"]
)
async def test_normalized_tcp_exec_failure_is_masked(
    endpoint, monkeypatch, consumer, checked, method
):
    """Guards the fix at 4f6ea75f on results and checked exceptions."""
    env, authority = endpoint
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = resolve_remote_docker_host(env)
    said = f'{method} "https://{authority}/v1.47/exec": tls: failed to verify certificate: private.internal.test\n'
    call = "_run_docker_compose_command" if consumer == "compose" else "_docker_cli"
    result = ExecResult(stdout=said, stderr=said, return_code=1)
    mock = (
        AsyncMock(side_effect=RuntimeError(said))
        if checked
        else AsyncMock(return_value=result)
    )
    monkeypatch.setattr(DockerSandbox, call, mock)
    if checked:
        with pytest.raises(RuntimeError) as caught:
            await getattr(sandbox, call)(["exec", "main", "false"])
        outputs = [str(caught.value)]
    else:
        result = await getattr(sandbox, call)(["exec", "main", "false"], check=False)
        outputs = [result.stdout, result.stderr]
    for output in outputs:
        assert "private.internal.test" not in output
        assert "failed to verify certificate" in output


@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
@pytest.mark.parametrize("consumer", ["live", "attach"])
async def test_normalized_tcp_stream_masks_before_retention(
    endpoint, monkeypatch, capsys, ending, consumer
):
    """Guards the fix at 4f6ea75f across read limits, tails and EOF."""
    env, authority = endpoint
    host = resolve_remote_docker_host(env)
    said = (
        f'Get "https://{authority}/v1.47/exec": tls: failed to verify certificate: '
        + "private.internal.test " * 4000
        + ending
    )
    assert len(said) > _STDERR_TAIL_LIMIT
    if consumer == "live":
        process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
        await _drain(process, said, 7)
        with pytest.raises(TransportClosedError) as caught:
            await process.readline()
        outputs = [
            process.stderr_tail,
            process._stderr_tail.decode(),
            str(caught.value),
            caught.value.diagnostic.raw_message,
            caught.value.diagnostic.stderr_snippet or "",
        ]
    else:
        sandbox = object.__new__(RemoteDockerSandbox)
        sandbox.docker_host = host
        monkeypatch.setattr(sandbox, "_compose_client", lambda: ([], {}))
        monkeypatch.setattr(sandbox, "_compose_base", lambda: ["docker", "compose"])
        monkeypatch.setattr(
            sandbox,
            "_run_docker_compose_command",
            AsyncMock(return_value=ExecResult(return_code=0)),
        )
        stderr = asyncio.StreamReader(limit=31)
        stderr.feed_data(said.encode())
        stderr.feed_eof()
        child = Mock(stderr=stderr, wait=AsyncMock(return_value=1))
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", AsyncMock(return_value=child)
        )
        await sandbox.attach()
        outputs = [capsys.readouterr().err]
    for output in outputs:
        assert "private.internal.test" not in output
        assert "failed to verify certificate" in output


@pytest.mark.parametrize(
    "control", ["success", "other-host", "other-port", "bare-tls", "show-address"]
)
def test_normalized_tcp_masking_preserves_task_details(endpoint, control):
    """Guards the fix at 4f6ea75f without reopening R5-1."""
    env, authority = endpoint
    if control == "show-address":
        env["BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS"] = "1"
    host = resolve_remote_docker_host(env)
    if control == "other-host":
        authority = "task.example.test:2375"
    elif control == "other-port":
        authority = authority.rsplit(":", 1)[0] + ":443"
    prefix = "" if control == "bare-tls" else f'Get "https://{authority}/info": '
    detail = "tls: failed to verify certificate: task-certificate.example.test"
    output = host.redact(
        prefix + detail, command_output=True, transport_errors=control != "success"
    )
    assert detail in output
