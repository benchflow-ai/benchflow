"""Transport details never become output."""

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

_HTTP = 'Head "https://docker.example.test:2376/_ping": '
_HIJACK = (
    "cannot connect to the Docker daemon. Is 'docker daemon' running on this host?: "
)
_CERTIFICATE = "tls: failed to verify certificate: "
_FIXED_CERTIFICATE = (
    "remote-docker[tcp]: tls: failed to verify certificate: "
    "<remote certificate error>; check the remote Docker TLS and proxy configuration"
)
_FIXED_UNKNOWN = (
    "remote-docker[tcp]: cannot connect to the Docker daemon; "
    "check the remote Docker host, TLS and proxy configuration"
)
_WRAPPERS = [
    "",
    "proxyconnect tcp: ",
    "proxyconnect tcp: proxy.example.test:8443: ",
    "proxyconnect tcp: proxyconnect tcp: ",
    "future transport wrapper with proxy.example.test: ",
    "PROXYCONNECT TCP: ",
    "\x1b[31mproxyconnect tcp: ",
]
_DETAILS = [
    _CERTIFICATE
    + "x509: certificate is valid for private.internal.test, not proxy.example.test",
    _CERTIFICATE
    + "x509: certificate is not valid for any names, but wanted to match private.internal.test",
    _CERTIFICATE
    + "x509: cannot validate certificate for 192.0.2.71 because it doesn't contain any IP SANs",
    _CERTIFICATE
    + 'x509: certificate signed by unknown authority (candidate "private.internal.test")',
    "remote error: tls: private.internal.test",
    "future diagnostic containing private.internal.test and 192.0.2.71",
]


def _sandbox(host=None):
    sandbox = object.__new__(RemoteDockerSandbox)
    sandbox.docker_host = host or RemoteDockerHost("tcp://docker.example.test:2376")
    return sandbox


def _assert_fixed(text):
    assert text.rstrip("\r\n") in {_FIXED_CERTIFICATE, _FIXED_UNKNOWN}
    for marker in (
        "private.internal.test",
        "proxy.example.test",
        "192.0.2.71",
        "_ping",
    ):
        assert marker not in text


@pytest.mark.parametrize("consumer", ["raw", "compose"])
@pytest.mark.parametrize("checked", [False, True])
async def test_proxy_certificate_rotation_hides_dummy_names(
    monkeypatch, consumer, checked
):
    """Reproduces the issue at 498c23e1 using only dummy names."""
    sandbox = _sandbox()
    said = _HTTP + "proxyconnect tcp: " + _DETAILS[0]
    call = "_docker_cli" if consumer == "raw" else "_run_docker_compose_command"
    mock = (
        AsyncMock(side_effect=RuntimeError(said))
        if checked
        else AsyncMock(return_value=ExecResult(stdout=said, stderr=said, return_code=1))
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
        assert "private.internal.test" not in output
        assert "proxy.example.test" not in output


@pytest.mark.parametrize("context", [_HTTP, _HIJACK])
@pytest.mark.parametrize("wrapper", _WRAPPERS)
@pytest.mark.parametrize("detail", _DETAILS)
@pytest.mark.parametrize("command_output", [False, True])
@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_daemon_diagnostic_is_a_fixed_message(
    context, wrapper, detail, command_output, ending
):
    """Guards the fix at 498c23e1 for opaque wrapper/detail classes."""
    host = _sandbox().docker_host
    # Prefix and request path can also contain certificate or proxy names.
    text = "private.internal.test: error during connect: " + context + wrapper + detail
    output = host.redact(text + ending, command_output=command_output)
    _assert_fixed(output)
    assert output.endswith(ending)
    assert host.redact(output, command_output=command_output) == output


@pytest.mark.parametrize("consumer", ["raw", "compose"])
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("wrapper", _WRAPPERS)
@pytest.mark.parametrize("detail", _DETAILS)
async def test_exec_result_and_exception_never_copy_transport_details(
    monkeypatch, consumer, checked, wrapper, detail
):
    """Guards the fix at 498c23e1 on every exec error boundary."""
    sandbox = _sandbox()
    said = _HTTP + wrapper + detail + "\nordinary task output\n"
    call = "_docker_cli" if consumer == "raw" else "_run_docker_compose_command"
    mock = (
        AsyncMock(side_effect=RuntimeError(said))
        if checked
        else AsyncMock(return_value=ExecResult(stdout=said, stderr=said, return_code=1))
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
        first, rest = output.split("\n", 1)
        _assert_fixed(first)
        assert rest == "ordinary task output\n"


@pytest.mark.parametrize("consumer", ["raw", "compose"])
@pytest.mark.parametrize("detail", _DETAILS)
async def test_data_stdout_stays_exact_and_diagnostic_stderr_is_fixed(
    monkeypatch, consumer, detail
):
    """Guards the fix at 498c23e1 without corrupting parsed stdout."""
    sandbox = _sandbox()
    said = _HTTP + "future proxy wrapper: " + detail
    call = "_docker_cli" if consumer == "raw" else "_run_docker_compose_command"
    monkeypatch.setattr(
        DockerSandbox,
        call,
        AsyncMock(
            return_value=ExecResult(
                stdout='{"data":"exact"}', stderr=said, return_code=1
            )
        ),
    )
    command = ["info"] if consumer == "raw" else ["config"]
    kwargs = {} if consumer == "raw" else {"merge_stderr": False}
    result = await getattr(sandbox, call)(command, check=False, **kwargs)
    assert result.stdout == '{"data":"exact"}'
    assert result.stderr == _FIXED_UNKNOWN


@pytest.mark.parametrize(
    "endpoint",
    [
        "Docker-Host",
        "docker.example.test.",
        "xn--dcker-jua.example.test",
        "192.0.2.44",
        "[2001:db8::44]",
        "[::ffff:192.0.2.44]",
    ],
)
@pytest.mark.parametrize("port", ["", ":", ":2376", ":02376"])
@pytest.mark.parametrize(
    "method",
    [
        "Get",
        "Head",
        "Post",
        "Put",
        "Delete",
        "Patch",
        "Options",
        "Connect",
        "Trace",
        "future-method",
        "",
    ],
)
def test_supported_authorities_share_the_fail_closed_message(endpoint, port, method):
    """Guards the fix at 498c23e1 for the entire accepted URL grammar."""
    host = RemoteDockerHost(f"tcp://{endpoint}{port}")
    effective_port = port[1:] or "2375"
    said = f'{method} "https://{endpoint}:{effective_port}/private.internal.test?proxy=proxy.example.test": future error: private.internal.test'
    assert host.redact(said, command_output=True) == _FIXED_UNKNOWN


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "/unterminated",
        '?query=private.internal.test": ',
        '#private.internal.test": ',
        '" unexpected separator ',
    ],
)
@pytest.mark.parametrize("command_output", [False, True])
def test_unknown_request_forms_cannot_pass_details_through(suffix, command_output):
    """Guards the fix at 498c23e1 with conservative URL recognition."""
    text = (
        'future operation "https://docker.example.test:2376'
        + suffix
        + "private.internal.test"
    )
    # A URL must have an authority boundary; a different hostname is task output.
    if not suffix:
        text = (
            'future operation "https://docker.example.test:2376/private.internal.test'
        )
    assert (
        _sandbox().docker_host.redact(text, command_output=command_output)
        == _FIXED_UNKNOWN
    )


@pytest.mark.parametrize("detail", _DETAILS[:5])
def test_context_free_docker_tls_diagnostic_is_opaque(detail):
    """Guards the fix at 498c23e1 outside mixed task output."""
    host = _sandbox().docker_host
    assert host.redact("future wrapper: " + detail) == _FIXED_UNKNOWN
    assert host.redact(detail, command_output=True) == detail


def test_daemon_disk_failure_uses_a_fixed_actionable_message():
    """Guards the fix at 498c23e1 while retaining disk-full handling."""
    said = _HTTP + "write /private.internal.test: no space left on device"
    output = _sandbox().docker_host.redact(said, command_output=True)
    assert output == (
        "remote-docker[tcp]: remote Docker host is out of disk space: "
        "no space left on device; free space on the remote Docker host and retry"
    )


@pytest.mark.parametrize("consumer", ["live", "attach"])
@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
@pytest.mark.parametrize("chunk_size", [7, 8192, 65536])
@pytest.mark.parametrize("wrapper", ["proxyconnect tcp: ", "future proxy wrapper: "])
async def test_streaming_masks_before_retention(
    monkeypatch, capsys, consumer, ending, chunk_size, wrapper
):
    """Guards the fix at 498c23e1 across limits, chunks and EOF."""
    host = _sandbox().docker_host
    said = _HTTP + wrapper + _DETAILS[0] * 2000 + ending
    assert len(said) > _STDERR_TAIL_LIMIT
    stderr = asyncio.StreamReader(limit=31)
    stdout = asyncio.StreamReader()
    stdout.feed_eof()
    child = Mock(
        stderr=stderr,
        stdout=stdout,
        pid=42,
        returncode=1,
        wait=AsyncMock(return_value=1),
    )
    if consumer == "live":
        process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
        process._set_process(child)
    else:
        sandbox = _sandbox(host)
        monkeypatch.setattr(sandbox, "_compose_client", lambda: ([], {}))
        monkeypatch.setattr(sandbox, "_compose_base", lambda: ["docker", "compose"])
        monkeypatch.setattr(
            sandbox,
            "_run_docker_compose_command",
            AsyncMock(return_value=ExecResult(return_code=0)),
        )
        monkeypatch.setattr(
            asyncio, "create_subprocess_exec", AsyncMock(return_value=child)
        )
        task = asyncio.create_task(sandbox.attach())
    data = said.encode()
    for start in range(0, len(data), chunk_size):
        stderr.feed_data(data[start : start + chunk_size])
        await asyncio.sleep(0)
    stderr.feed_eof()
    if consumer == "live":
        await process._finish_stderr_drain(cancel_on_timeout=False)
        with pytest.raises(TransportClosedError) as caught:
            await process.readline()
        for text in (
            process.stderr_tail,
            process._stderr_tail.decode(),
            caught.value.diagnostic.stderr_snippet,
        ):
            _assert_fixed(text)
        assert "private.internal.test" not in str(caught.value)
        assert "private.internal.test" not in caught.value.diagnostic.raw_message
    else:
        await task
        _assert_fixed(capsys.readouterr().err)


@pytest.mark.parametrize(
    "split", range(len(_HTTP + "proxyconnect tcp: " + _CERTIFICATE) + 1)
)
async def test_streaming_waits_for_the_whole_diagnostic(split):
    """Guards the fix at 498c23e1 before any prefix is retained."""
    host = _sandbox().docker_host
    process = DockerProcess("rd-test", "/tmp/rd-test", [], redact=host.redact)
    stderr = asyncio.StreamReader(limit=31)
    process._set_process(Mock(stderr=stderr))
    prefix = _HTTP + "proxyconnect tcp: " + _CERTIFICATE
    stderr.feed_data(prefix[:split].encode())
    await asyncio.sleep(0)
    assert process.stderr_tail == ""
    stderr.feed_data((prefix[split:] + "private.internal.test\n").encode())
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)
    assert process.stderr_tail == _FIXED_UNKNOWN + "\n"


async def test_live_setup_diagnostic_is_fixed_before_truncation(monkeypatch):
    """Guards the fix at 498c23e1 during setup and rollback."""
    process = DockerProcess(
        "rd-test", "/tmp/rd-test", [], redact=_sandbox().docker_host.redact
    )
    said = (_HTTP + "future proxy wrapper: " + _DETAILS[0] * 2000).encode()
    stderr = asyncio.StreamReader()
    process._set_process(Mock(stderr=stderr))
    stderr.feed_data(said)
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)

    assert process.stderr_tail.endswith(_FIXED_UNKNOWN)


@pytest.mark.parametrize(
    "control",
    ["success", "bare-task-tls", "other-host", "other-port", "show-address", "ssh"],
)
@pytest.mark.parametrize("detail", _DETAILS)
def test_success_and_unrelated_task_output_keep_details(control, detail):
    """Guards the fix at 498c23e1 without changing task diagnostics."""
    host = RemoteDockerHost(
        ("ssh" if control == "ssh" else "tcp") + "://docker.example.test:2376",
        show_address=control == "show-address",
    )
    context = _HTTP
    if control == "bare-task-tls":
        context = "task output: "
    elif control == "other-host":
        context = 'Get "https://task.example.test:2376/": '
    elif control == "other-port":
        context = 'Get "https://docker.example.test:443/": '
    text = context + "proxyconnect tcp: " + detail + "\n"
    output = host.redact(
        text, command_output=True, transport_errors=control != "success"
    )
    assert "proxyconnect tcp: " + detail in output
