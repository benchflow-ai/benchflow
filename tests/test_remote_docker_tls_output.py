"""Guards b0699347's task-output corruption and truncated TLS leaks (RD R5)."""

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

_TASK_ERRORS = [
    "tls: failed to verify certificate: x509: certificate is valid for "
    "api.example.test, not expected.example.test",
    "tls: failed to verify certificate: x509: certificate is not valid for "
    "any names, but wanted to match expected.example.test",
    "tls: failed to verify certificate: x509: cannot validate certificate for "
    "192.0.2.71 because it doesn't contain any IP SANs",
]


@pytest.mark.parametrize("consumer", ["compose", "raw"])
@pytest.mark.parametrize("return_code", [0, 1, 125])
@pytest.mark.parametrize("wrapper", ["", 'Get "https://task.example.test:443/": '])
@pytest.mark.parametrize("said", _TASK_ERRORS)
async def test_task_tls_output_is_preserved(
    monkeypatch, consumer, return_code, wrapper, said
):
    """Guards the fix at b0699347 on both exec result streams."""
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = RemoteDockerHost("tcp://Docker-Host:2376")
    text = wrapper + said + "\n"
    result = ExecResult(stdout=text, stderr=text, return_code=return_code)
    if consumer == "compose":
        monkeypatch.setattr(
            DockerSandbox, "_run_docker_compose_command", AsyncMock(return_value=result)
        )
        actual = await sandbox._run_docker_compose_command(
            ["exec", "main", "false"], check=False
        )
    else:
        monkeypatch.setattr(
            DockerSandbox, "_docker_cli", AsyncMock(return_value=result)
        )
        actual = await sandbox._docker_cli(["exec", "container", "false"], check=False)
    assert actual == result


@pytest.mark.parametrize("consumer", ["compose", "raw"])
@pytest.mark.parametrize("said", _TASK_ERRORS)
async def test_checked_exec_keeps_task_tls_errors(monkeypatch, consumer, said):
    """Guards the fix at b0699347 on the exception path too."""
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = RemoteDockerHost("tcp://Docker-Host:2376")
    method = "_run_docker_compose_command" if consumer == "compose" else "_docker_cli"
    monkeypatch.setattr(
        DockerSandbox, method, AsyncMock(side_effect=RuntimeError(said))
    )
    with pytest.raises(RuntimeError) as caught:
        await getattr(sandbox, method)(["exec", "main", "false"])
    assert str(caught.value) == said


@pytest.mark.parametrize(
    "endpoint",
    [
        "Docker-Host:2376",
        "docker.example.test:2376",
        "192.0.2.44:2376",
        "[2001:db8::44]:2376",
    ],
)
@pytest.mark.parametrize(
    "method", ["Get", "Head", "Post", "Put", "Delete", "Patch", "Options"]
)
def test_daemon_tls_diagnostics_require_failed_exec(endpoint, method):
    """Success cannot be a transport failure."""
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = RemoteDockerHost(f"tcp://{endpoint}")
    text = f'error during connect: {method} "https://{endpoint}/v1.47/info": {_TASK_ERRORS[0]}\n'
    success = sandbox._redact_result(
        ExecResult(stdout=text, stderr=text, return_code=0), command_output=True
    )
    failure = sandbox._redact_result(
        ExecResult(stdout=text, stderr=text, return_code=1), command_output=True
    )
    for output in (success.stdout, success.stderr):
        assert _TASK_ERRORS[0] in output
    for output in (failure.stdout, failure.stderr):
        assert "api.example.test" not in output
        assert "expected.example.test" not in output


async def _drain(process, text, chunk_size):
    # A tiny reader limit forces readuntil's overrun path as well as ordinary
    # chunk splits, without starting a daemon or running a task command.
    stderr = asyncio.StreamReader(limit=31)
    stdout = asyncio.StreamReader()
    stdout.feed_eof()
    child = Mock(stderr=stderr, stdout=stdout, pid=42, returncode=1)
    process._set_process(child)
    data = text.encode()
    for start in range(0, len(data), chunk_size):
        stderr.feed_data(data[start : start + chunk_size])
        await asyncio.sleep(0)
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)


@pytest.mark.parametrize("chunk_size", [7, 8192, 65536])
@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
@pytest.mark.parametrize("show_address", [False, True])
async def test_live_tls_redacts_before_tail_truncation(
    chunk_size, ending, show_address
):
    """Guards the fix at b0699347 for a 96 KB daemon SAN mismatch."""
    host = RemoteDockerHost("tcp://Docker-Host:2376", show_address=show_address)
    process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
    names = ", ".join(f"node{n:04}.internal.test" for n in range(4000))
    text = (
        'error during connect: Get "https://Docker-Host:2376/v1.47/exec": '
        f"tls: failed to verify certificate: x509: certificate is valid for {names}, not Docker-Host{ending}"
    )
    assert len(text) > _STDERR_TAIL_LIMIT
    await _drain(process, text, chunk_size)
    with pytest.raises(TransportClosedError) as caught:
        await process.readline()
    shown = "\n".join(
        [
            process.stderr_tail,
            str(caught.value),
            caught.value.diagnostic.raw_message,
            caught.value.diagnostic.stderr_snippet or "",
        ]
    )
    if show_address:
        assert process.stderr_tail == text.encode()[-_STDERR_TAIL_LIMIT:].decode()
    else:
        assert "internal.test" not in shown
        assert "Docker-Host" not in shown
        assert "failed to verify certificate" in shown
        assert b"internal.test" not in process._stderr_tail
    assert len(process._stderr_tail) <= _STDERR_TAIL_LIMIT


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("said", _TASK_ERRORS)
async def test_live_keeps_long_task_tls_output(remote, said):
    """Guards the fixes at b0699347 for local and remote task stderr."""
    host = RemoteDockerHost("tcp://Docker-Host:2376")
    process = DockerProcess(
        "rd-test", "/tmp/rd-test", [], redact=host.redact if remote else None
    )
    text = "task-data " * 10000 + said + "\n"
    await _drain(process, text, 8192)
    assert process.stderr_tail == text.encode()[-_STDERR_TAIL_LIMIT:].decode()


_REQUEST = (
    'Get "https://Docker-Host:2376/v1.47/info": tls: failed to verify certificate: '
)


@pytest.mark.parametrize("split", range(len(_REQUEST) + 1))
async def test_live_keeps_tls_context_at_every_prefix_split(split):
    """Guards the fix at b0699347 without depending on x509 spelling."""
    host = RemoteDockerHost("tcp://Docker-Host:2376")
    process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
    stderr = asyncio.StreamReader(limit=31)
    process._set_process(Mock(stderr=stderr))
    stderr.feed_data(_REQUEST[:split].encode())
    await asyncio.sleep(0)
    # An incomplete prefix is never exposed as a context-free tail.
    assert process.stderr_tail == ""
    stderr.feed_data(
        (
            _REQUEST[split:] + "future x509 diagnostic private.test\nnormal task line\n"
        ).encode()
    )
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)
    assert "private.test" not in process.stderr_tail
    assert "failed to verify certificate" in process.stderr_tail
    assert process.stderr_tail.endswith("normal task line\n")


async def test_oversized_stderr_line_is_omitted_as_a_unit(monkeypatch):
    """Guards the fix at b0699347 above the bounded line buffer."""
    from benchflow.sandbox.process import docker

    monkeypatch.setattr(docker, "_BUFFER_LIMIT", 256)
    host = RemoteDockerHost("tcp://Docker-Host:2376")
    process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
    await _drain(process, _REQUEST + "private.test " * 1000 + "\nnext line\n", 7)
    assert (
        process.stderr_tail
        == "<Docker stderr line exceeded diagnostic limit>\nnext line\n"
    )


@pytest.mark.parametrize("said", [_TASK_ERRORS[0], _REQUEST + "private.test " * 8000])
async def test_attach_preserves_task_text_and_masks_large_tls_errors(
    monkeypatch, capsys, said
):
    """Guards the fixes at b0699347 at the attach output boundary."""
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = RemoteDockerHost("tcp://Docker-Host:2376")
    monkeypatch.setattr(sandbox, "_compose_client", lambda: ([], {}))
    monkeypatch.setattr(sandbox, "_compose_base", lambda: ["docker", "compose"])
    monkeypatch.setattr(
        sandbox,
        "_run_docker_compose_command",
        AsyncMock(return_value=ExecResult(return_code=0)),
    )
    stderr = asyncio.StreamReader(limit=31)
    stderr.feed_data((said + "\n").encode())
    stderr.feed_eof()
    child = Mock(stderr=stderr, wait=AsyncMock(return_value=1))
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=child)
    )
    await sandbox.attach()
    shown = capsys.readouterr().err
    if said == _TASK_ERRORS[0]:
        assert shown == said + "\n"
    else:
        assert "private.test" not in shown
        assert "Docker-Host" not in shown
        assert "failed to verify certificate" in shown
