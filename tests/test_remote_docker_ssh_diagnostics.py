"""SSH helper stderr stays opaque."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from benchflow.diagnostics import TransportClosedError
from benchflow.sandbox import remote_docker as remote
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.process import DockerProcess
from benchflow.sandbox.process import docker as process_module

_FIXED = (
    "remote-docker[ssh]: cannot connect to the Docker daemon; "
    "check the remote Docker host, TLS and proxy configuration"
)
_HELPER = (
    "command [ssh -l ops -- gateway.example.test docker system dial-stdio] "
    "has exited with exit status 1, make sure the URL is valid, and Docker "
    "18.09 or later is installed on the remote host: stderr="
)
_RAW = "failed to open the raw stream connection: "
_HTTP = 'Get "http://docker.example.com/v1.51/exec/dummy/start": '
_HIJACK = (
    "cannot connect to the Docker daemon. Is 'docker daemon' running on this host?: "
)
_CONTEXTS = [
    _HTTP + _HELPER + _RAW,
    _HIJACK + _HELPER + _RAW,
    _HELPER,
    _RAW,
    _HTTP + "future helper envelope: ",
    "command [future-helper] unknown diagnostic: ",
    _HELPER.replace("has exited with exit status 1", "did not exit after EOF"),
    (_HTTP + _HELPER).upper(),
]
_DETAILS = [
    "tls: failed to verify certificate: x509: certificate is valid for "
    "private.internal.test, not engine.internal.test",
    "proxyconnect tcp: tls: failed to verify certificate: x509: certificate "
    "is not valid for any names, but wanted to match private.internal.test",
    "x509: cannot validate certificate for 192.0.2.71 because it doesn't "
    "contain any IP SANs",
    'x509: certificate signed by unknown authority (candidate "private.internal.test")',
    "remote error: tls: private.internal.test",
    "future diagnostic naming private.internal.test and 192.0.2.71",
    "unknown helper failure\nprivate.internal.test\r\nengine.internal.test",
]


@pytest.fixture(autouse=True)
def no_ssh_config(monkeypatch):
    # No connection, local SSH configuration or credential is needed.
    # The assigned comparison base predates SSH configuration inspection.
    if hasattr(remote.RemoteDockerHost, "ssh_view"):
        monkeypatch.setattr(
            remote.RemoteDockerHost,
            "ssh_view",
            lambda self: remote._SshView((), (), True, False),
        )


def _host(**kwargs):
    return remote.RemoteDockerHost(
        "ssh://ops@gateway.example.test", user="ops", **kwargs
    )


def _sandbox(host=None):
    sandbox = object.__new__(remote.RemoteDockerSandbox)
    sandbox.docker_host = host or _host()
    return sandbox


def _process(host=None):
    # Exercise production construction: every live process gets a fresh
    # redactor, even when it shares its immutable host with another process.
    return DockerProcess.from_sandbox_env(
        SimpleNamespace(
            docker_host=host or _host(),
            compose_project_name="rd-test",
            environment_dir=Path("/tmp/rd-task"),
            _docker_compose_paths=[],
        )
    )


def _assert_fixed(text):
    assert text
    assert all(line == _FIXED for line in text.splitlines() if line)
    for marker in ("private.internal.test", "engine.internal.test", "192.0.2.71"):
        assert marker not in text


@pytest.mark.parametrize("consumer", ["raw", "compose"])
@pytest.mark.parametrize("checked", [False, True])
async def test_ssh_certificate_rotation_hides_dummy_names(
    monkeypatch, consumer, checked
):
    """Reproduces the issue at aa8ec69f with an ordinary TLS failure."""
    sandbox = _sandbox()
    said = _HTTP + _HELPER + _RAW + _DETAILS[0]
    call = "_docker_cli" if consumer == "raw" else "_run_docker_compose_command"
    monkeypatch.setattr(
        DockerSandbox,
        call,
        AsyncMock(side_effect=RuntimeError(said))
        if checked
        else AsyncMock(
            return_value=ExecResult(stdout=said, stderr=said, return_code=1)
        ),
    )
    if checked:
        with pytest.raises(RuntimeError) as caught:
            await getattr(sandbox, call)(["exec", "main", "true"])
        outputs = [str(caught.value)]
    else:
        result = await getattr(sandbox, call)(["exec", "main", "true"], check=False)
        outputs = [result.stdout, result.stderr]
    for output in outputs:
        assert "private.internal.test" not in output
        assert "engine.internal.test" not in output
        _assert_fixed(output)


@pytest.mark.parametrize("consumer", ["raw", "compose"])
@pytest.mark.parametrize("checked", [False, True])
@pytest.mark.parametrize("context", [_HTTP, _HIJACK, _HELPER, _RAW])
@pytest.mark.parametrize("detail", _DETAILS)
async def test_exec_error_boundaries_discard_unknown_helper_contents(
    monkeypatch, consumer, checked, context, detail
):
    """Guards the fix at aa8ec69f across result and exception paths."""
    sandbox = _sandbox()
    said = context + detail
    call = "_docker_cli" if consumer == "raw" else "_run_docker_compose_command"
    monkeypatch.setattr(
        DockerSandbox,
        call,
        AsyncMock(side_effect=RuntimeError(said))
        if checked
        else AsyncMock(
            return_value=ExecResult(stdout=said, stderr=said, return_code=125)
        ),
    )
    if checked:
        with pytest.raises(RuntimeError) as caught:
            await getattr(sandbox, call)(["exec", "main", "true"])
        _assert_fixed(str(caught.value))
    else:
        result = await getattr(sandbox, call)(["exec", "main", "true"], check=False)
        _assert_fixed(result.stdout)
        _assert_fixed(result.stderr)


@pytest.mark.parametrize("context", _CONTEXTS)
@pytest.mark.parametrize("detail", _DETAILS)
@pytest.mark.parametrize("command_output", [False, True])
@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_ssh_envelopes_discard_every_detail_and_continuation(
    context, detail, command_output, ending
):
    """Guards the fix at aa8ec69f across helper envelope forms."""
    host = _host()
    said = "private.internal.test: " + context + detail + ending
    output = host.redact(said, command_output=command_output)
    _assert_fixed(output)
    assert output.endswith(ending)
    assert host.redact(output, command_output=command_output) == output


@pytest.mark.parametrize("consumer", ["raw", "compose"])
@pytest.mark.parametrize("merged", [False, True])
async def test_machine_stdout_and_helper_stderr_have_separate_boundaries(
    monkeypatch, consumer, merged
):
    """Guards the fix at aa8ec69f without corrupting data stdout."""
    sandbox = _sandbox()
    said = _HTTP + _HELPER + _DETAILS[-1]
    call = "_docker_cli" if consumer == "raw" else "_run_docker_compose_command"
    monkeypatch.setattr(
        DockerSandbox,
        call,
        AsyncMock(
            return_value=ExecResult(
                stdout=said if merged else '{"data":"exact"}',
                stderr=said,
                return_code=1,
            )
        ),
    )
    command = ["inspect", "dummy"] if consumer == "raw" else ["config"]
    kwargs = {} if consumer == "raw" else {"merge_stderr": merged}
    result = await getattr(sandbox, call)(
        command,
        check=False,
        raw_output=not (merged and consumer == "compose"),
        **kwargs,
    )
    # Data stays raw; the diagnostic copy is sanitized at its display boundary.
    _assert_fixed(sandbox._diagnostic_text(result.stderr))
    if merged and consumer == "compose":
        _assert_fixed(result.stdout)
    else:
        assert result.stdout == (said if merged else '{"data":"exact"}')


@pytest.mark.parametrize("consumer", ["live", "attach"])
@pytest.mark.parametrize("context", _CONTEXTS)
@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
@pytest.mark.parametrize("chunk_size", [7, 8192])
@pytest.mark.parametrize("line_cap", [None, 128])
async def test_helper_stream_is_opaque_through_chunks_lines_and_eof(
    monkeypatch, capsys, consumer, context, ending, chunk_size, line_cap
):
    """Guards the fix at aa8ec69f before retention and truncation."""
    if line_cap is not None:
        monkeypatch.setattr(process_module, "_BUFFER_LIMIT", line_cap)
    host = _host()
    said = context + _DETAILS[0] * 2000 + "\n" + _DETAILS[-1] + ending
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
        process = _process(host)
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
        assert "private.internal.test" not in caught.value.diagnostic.raw_message
    else:
        await task
        _assert_fixed(capsys.readouterr().err)


@pytest.mark.parametrize("split", range(len(_HTTP + _HELPER) + 1))
async def test_helper_prefix_is_never_retained_before_it_is_complete(split):
    """Guards the fix at aa8ec69f for every helper-envelope split."""
    process = _process()
    stderr = asyncio.StreamReader(limit=31)
    process._set_process(Mock(stderr=stderr))
    prefix = _HTTP + _HELPER
    stderr.feed_data(prefix[:split].encode())
    await asyncio.sleep(0)
    assert process.stderr_tail == ""
    stderr.feed_data(
        (prefix[split:] + _DETAILS[0] + "\nprivate.internal.test\n").encode()
    )
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)
    _assert_fixed(process._stderr_tail.decode())


async def test_setup_and_rollback_discard_helper_details_before_truncation(monkeypatch):
    """Guards the fix at aa8ec69f during env setup and rollback."""
    process = _process()
    said = (_HELPER + _DETAILS[0] * 2000 + "\nprivate.internal.test").encode()
    stderr = asyncio.StreamReader()
    process._set_process(Mock(stderr=stderr))
    stderr.feed_data(said)
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)

    assert "private.internal.test" not in process.stderr_tail
    assert _FIXED in process.stderr_tail


@pytest.mark.parametrize("prefix", [_HELPER, "future undecidable helper prefix: "])
async def test_oversized_unknown_lines_taint_following_stream_lines(
    monkeypatch, prefix
):
    """Guards the fix at aa8ec69f when the line cap loses context."""
    monkeypatch.setattr(process_module, "_BUFFER_LIMIT", 128)
    process = _process()
    stderr = asyncio.StreamReader(limit=31)
    process._set_process(Mock(stderr=stderr))
    stderr.feed_data((prefix + "x" * 1000 + "\nprivate.internal.test\n").encode())
    stderr.feed_eof()
    await process._finish_stderr_drain(cancel_on_timeout=False)
    _assert_fixed(process._stderr_tail.decode())


async def test_stream_taint_is_per_process_and_preserves_unrelated_output():
    """Guards the fix at aa8ec69f without tainting another process."""
    host = _host()
    first, second = _process(host), _process(host)
    ordinary = "task TLS: " + _DETAILS[0] + "\n"
    for process, text in (
        (first, _HELPER + _DETAILS[0] + "\nprivate.internal.test\n"),
        (second, ordinary),
    ):
        stderr = asyncio.StreamReader()
        stderr.feed_data(text.encode())
        stderr.feed_eof()
        process._set_process(Mock(stderr=stderr))
        await process._finish_stderr_drain(cancel_on_timeout=False)
    _assert_fixed(first._stderr_tail.decode())
    assert second._stderr_tail.decode() == ordinary
    # A new subprocess on the same object must have a clean stream state.
    stderr = asyncio.StreamReader()
    stderr.feed_data(ordinary.encode())
    stderr.feed_eof()
    first._set_process(Mock(stderr=stderr))
    await first._finish_stderr_drain(cancel_on_timeout=False)
    assert first.stderr_tail == ordinary
    assert host.redact(ordinary, command_output=True) == ordinary


@pytest.mark.parametrize("detail", _DETAILS)
@pytest.mark.parametrize("control", ["success", "task-tls", "task-url", "show-address"])
def test_success_and_unrelated_task_output_preserve_details(detail, control):
    """Guards the fix at aa8ec69f for supported task-output controls."""
    host = _host(show_address=control == "show-address")
    prefix = _HELPER
    if control == "task-tls":
        prefix = "task TLS: "
    elif control == "task-url":
        prefix = 'Get "https://task.example.test:443/": '
    output = host.redact(
        prefix + detail, command_output=True, transport_errors=control != "success"
    )
    assert detail in output


@pytest.mark.parametrize(
    "known, fixed",
    [
        (
            "Permission denied (publickey).",
            "Permission denied (publickey); check SSH authentication and remote Docker access",
        ),
        (
            "Host key verification failed.",
            "Host key verification failed; verify the remote SSH host key and known_hosts",
        ),
        (
            "Connection refused",
            "Connection refused; check the remote SSH host and port",
        ),
        (
            "no space left on device",
            "remote Docker host is out of disk space: no space left on device; free space on the remote Docker host and retry",
        ),
    ],
)
def test_known_ssh_failures_select_only_fixed_messages(known, fixed):
    """Guards the fix at aa8ec69f while preserving actionable errors."""
    host = _host()
    text = _HELPER + "private.internal.test: " + known + " engine.internal.test"
    output = host.redact(text, command_output=True)
    assert output == f"remote-docker[ssh]: {fixed}"
    redact = host.stream_redactor()
    assert redact(text, command_output=True) == output
    assert redact(output, command_output=True) == output
    _assert_fixed(redact("private.internal.test\n", command_output=True))
