"""Attach TLS errors have no HTTP URL."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.diagnostics import TransportClosedError
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.process import DockerProcess
from benchflow.sandbox.process._base import _STDERR_TAIL_LIMIT
from benchflow.sandbox.remote_docker import RemoteDockerHost, RemoteDockerSandbox

_HIJACK = (
    "cannot connect to the Docker daemon. Is 'docker daemon' running on this host?: "
)
_TLS = "tls: failed to verify certificate: "
_DETAILS = [
    "x509: certificate is valid for private.internal.test, not docker.example.test",
    "x509: certificate is not valid for any names, but wanted to match private.internal.test",
    "x509: cannot validate certificate for 192.0.2.71 because it doesn't contain any IP SANs",
    'x509: certificate signed by unknown authority (candidate authority certificate "private.internal.test")',
    "x509: certificate relies on legacy Common Name field: private.internal.test",
    "future verifier diagnostic with private.internal.test",
]
_WRAPPERS = [_HIJACK, "error during connect: " + _HIJACK, _HIJACK.upper()]


def _sandbox():
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = RemoteDockerHost("tcp://docker.example.test:2376")
    return sandbox


def _assert_masked(text):
    assert "private.internal.test" not in text
    assert "192.0.2.71" not in text
    assert "docker.example.test" not in text
    assert _TLS in text
    assert "<remote certificate error>" in text


@pytest.mark.parametrize("consumer", ["compose", "raw"])
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("detail", _DETAILS)
@pytest.mark.parametrize("wrapper", _WRAPPERS)
async def test_failed_exec_masks_hijack_certificate_details(
    monkeypatch, consumer, checked, detail, wrapper
):
    """Guards the fix at 45fc4b95 in results and checked exceptions."""
    sandbox = _sandbox()
    said = wrapper + _TLS + detail + "\nordinary task output\n"
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
            await getattr(sandbox, call)(["exec", "main", "true"])
        outputs = [str(caught.value)]
    else:
        result = await getattr(sandbox, call)(["exec", "main", "true"], check=False)
        outputs = [result.stdout, result.stderr]
    for output in outputs:
        _assert_masked(output)
        assert output.endswith("\nordinary task output\n")


@pytest.mark.parametrize("consumer", ["compose", "raw"])
@pytest.mark.parametrize("detail", _DETAILS)
@pytest.mark.parametrize("return_code", [0, 1, 125])
@pytest.mark.parametrize(
    "control", ["success", "bare-tls", "task-wrapper", "other-host", "show-address"]
)
async def test_exec_keeps_successful_and_unrelated_tls_output(
    monkeypatch, consumer, detail, return_code, control
):
    """Guards the fix at 45fc4b95 without corrupting task output."""
    sandbox = _sandbox()
    if control == "show-address":
        sandbox.docker_host = RemoteDockerHost(
            "tcp://docker.example.test:2376", show_address=True
        )
    wrapper = {
        "success": _HIJACK,
        "bare-tls": "",
        "task-wrapper": "cannot connect to the task daemon: ",
        "other-host": 'Get "https://task.example.test:443/": ',
        "show-address": _HIJACK,
    }[control]
    if control == "success":
        return_code = 0
    said = wrapper + _TLS + detail + "\n"
    result = ExecResult(stdout=said, stderr=said, return_code=return_code)
    call = "_run_docker_compose_command" if consumer == "compose" else "_docker_cli"
    monkeypatch.setattr(DockerSandbox, call, AsyncMock(return_value=result))
    assert await getattr(sandbox, call)(["exec", "main", "true"], check=False) == result


@pytest.mark.parametrize(
    "endpoint",
    [
        "Docker-Host",
        "docker.example.test.",
        "192.0.2.44",
        "[2001:db8::44]",
        "[::ffff:192.0.2.44]",
        "xn--dcker-jua.example.test",
    ],
)
@pytest.mark.parametrize("port", ["", ":", ":2375", ":2376", ":02376"])
@pytest.mark.parametrize("command_output", [False, True])
@pytest.mark.parametrize("detail", _DETAILS)
def test_hijack_transport_context_does_not_require_a_printed_endpoint(
    endpoint, port, command_output, detail
):
    """Guards the fix at 45fc4b95 for every supported authority form."""
    host = RemoteDockerHost(f"tcp://{endpoint}{port}")
    text = _HIJACK + _TLS + detail
    output = host.redact(text, command_output=command_output)
    _assert_masked(output)
    assert host.redact(output, command_output=command_output) == output


async def _feed(process, text, chunk_size):
    stderr = asyncio.StreamReader(limit=31)
    stdout = asyncio.StreamReader()
    stdout.feed_eof()
    process._set_process(Mock(stderr=stderr, stdout=stdout, pid=42, returncode=1))
    data = text.encode()
    for start in range(0, len(data), chunk_size):
        stderr.feed_data(data[start : start + chunk_size])
        await asyncio.sleep(0)
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)


@pytest.mark.parametrize("consumer", ["live", "attach"])
@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
@pytest.mark.parametrize("chunk_size", [7, 8192, 65536])
@pytest.mark.parametrize("show_address", [False, True])
async def test_hijack_stream_masks_before_retention(
    monkeypatch, capsys, consumer, ending, chunk_size, show_address
):
    """Guards the fix at 45fc4b95 across read limits, tails and EOF."""
    host = RemoteDockerHost("tcp://docker.example.test:2376", show_address=show_address)
    said = _HIJACK + _TLS + _DETAILS[0] * 2000 + ending
    assert len(said) > _STDERR_TAIL_LIMIT
    if consumer == "live":
        process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
        await _feed(process, said, chunk_size)
        with pytest.raises(TransportClosedError) as caught:
            await process.readline()
        outputs = [
            process.stderr_tail,
            process._stderr_tail.decode(),
            str(caught.value),
            caught.value.diagnostic.raw_message,
            caught.value.diagnostic.stderr_snippet or "",
        ]
        if show_address:
            assert process.stderr_tail == said.encode()[-_STDERR_TAIL_LIMIT:].decode()
    else:
        sandbox = _sandbox()
        sandbox.docker_host = host
        monkeypatch.setattr(sandbox, "_compose_client", lambda: ([], {}))
        monkeypatch.setattr(sandbox, "_compose_base", lambda: ["docker", "compose"])
        monkeypatch.setattr(
            sandbox,
            "_run_docker_compose_command",
            AsyncMock(return_value=ExecResult(return_code=0)),
        )
        stderr = asyncio.StreamReader(limit=31)
        child = Mock(stderr=stderr, wait=AsyncMock(return_value=1))
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", AsyncMock(return_value=child)
        )
        task = asyncio.create_task(sandbox.attach())
        data = said.encode()
        for start in range(0, len(data), chunk_size):
            stderr.feed_data(data[start : start + chunk_size])
            await asyncio.sleep(0)
        stderr.feed_eof()
        await task
        outputs = [capsys.readouterr().err]
        if show_address:
            assert outputs == [said]
    if not show_address:
        for output in outputs:
            _assert_masked(output)


@pytest.mark.parametrize("split", range(len(_HIJACK + _TLS) + 1))
async def test_live_hijack_context_survives_every_prefix_split(split):
    """Guards the fix at 45fc4b95 before any raw tail is retained."""
    host = _sandbox().docker_host
    process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
    stderr = asyncio.StreamReader(limit=31)
    process._set_process(Mock(stderr=stderr))
    prefix = _HIJACK + _TLS
    stderr.feed_data(prefix[:split].encode())
    await asyncio.sleep(0)
    assert process.stderr_tail == ""
    stderr.feed_data(
        (prefix[split:] + _DETAILS[0] + "\nordinary task output\n").encode()
    )
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)
    _assert_masked(process.stderr_tail)
    assert process.stderr_tail.endswith("\nordinary task output\n")


@pytest.mark.parametrize("scheme", ["ssh", "tcp"])
@pytest.mark.parametrize("detail", _DETAILS)
async def test_live_preserves_unrelated_task_tls_and_ssh_output(scheme, detail):
    """Guards the fix at aa8ec69f for unrelated task TLS on both transports."""
    host = RemoteDockerHost(f"{scheme}://docker.example.test:2376")
    process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
    prefix = "cannot connect to the task daemon: " if scheme == "ssh" else "task TLS: "
    text = prefix + _TLS + detail + "\n"
    await _feed(process, text, 7)
    assert process.stderr_tail == text


@pytest.mark.parametrize("detail", _DETAILS)
async def test_live_setup_masks_hijack_tls_before_error_truncation(monkeypatch, detail):
    """Guards the fix at 45fc4b95 on live setup and rollback execs."""
    host = _sandbox().docker_host
    process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
    said = (_HIJACK + _TLS + detail * 2000).encode()
    stderr = asyncio.StreamReader()
    process._set_process(Mock(stderr=stderr))
    stderr.feed_data(said)
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)

    _assert_masked(process.stderr_tail)
