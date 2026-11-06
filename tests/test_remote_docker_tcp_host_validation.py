"""TCP hosts must survive Go URL encoding."""

from __future__ import annotations

import pytest

from benchflow.sandbox.remote_docker import (
    RemoteDockerConfigError,
    RemoteDockerHost,
    resolve_remote_docker_host,
)

# Literal, decomposed and percent-encoded Unicode all describe the same
# normalization class. Percent escapes in DNS names are refused altogether;
# Scoped IPv6 is refused, including its URL zone separator and ASCII forms.
_UNSTABLE_HOSTS = [
    "döcker.example.test",
    "DÖCKER.Example.test.",
    "do\u0308cker.example.test",
    "例子.example.test",
    "🐳.example.test",
    "d%C3%B6cker.example.test",
    "d%c3%b6cker.example.test",
    "%64ocker.example.test",
    "docker%25.example.test",
    "[fe80::44%25éth0]",
    "[fe80::44%eth0]",
]


@pytest.mark.parametrize("hostname", ["[::1]döcker", "[::1]suffix"])
@pytest.mark.parametrize("port", ["", ":", ":2376"])
@pytest.mark.parametrize("source", ["BENCHFLOW_REMOTE_DOCKER_HOST", "DOCKER_HOST"])
def test_malformed_bracket_hosts_remain_refused(hostname, port, source):
    """Guards the fix at 3caef02c without relaxing parser refusals."""
    with pytest.raises(RemoteDockerConfigError):
        resolve_remote_docker_host({source: f"tcp://{hostname}{port}"})


@pytest.mark.parametrize("port", ["", ":", ":2376"])
def test_unicode_daemon_certificate_cannot_be_published(tmp_path, port):
    """Guards the fix at 3caef02c with Go's actual URL spelling."""
    for name in ("ca.pem", "cert.pem", "key.pem"):
        (tmp_path / name).write_text("dummy TLS file; no connection is made\n")
    try:
        host = resolve_remote_docker_host(
            {
                "BENCHFLOW_REMOTE_DOCKER_HOST": f"tcp://döcker.example.test{port}",
                "DOCKER_TLS_VERIFY": "1",
                "DOCKER_CERT_PATH": str(tmp_path),
            }
        )
    except RemoteDockerConfigError as exc:
        assert "ASCII" in str(exc)
        return
    effective_port = "2376" if port == ":2376" else "2375"
    said = (
        f'Get "https://d%C3%B6cker.example.test:{effective_port}/v1.47/exec": '
        "tls: failed to verify certificate: x509: certificate is valid for "
        "private.internal.test, not xn--dcker-jua.example.test"
    )
    assert "private.internal.test" not in host.redact(said, command_output=True)


@pytest.mark.parametrize("hostname", _UNSTABLE_HOSTS)
@pytest.mark.parametrize("port", ["", ":", ":2376", ":02376"])
@pytest.mark.parametrize("source", ["BENCHFLOW_REMOTE_DOCKER_HOST", "DOCKER_HOST"])
@pytest.mark.parametrize("suffix", ["", "/proxy?test=1#fragment"])
@pytest.mark.parametrize("show_address", ["0", "1"])
def test_tcp_host_is_refused_before_launch(
    tmp_path, hostname, port, source, suffix, show_address
):
    """Guards the fix at 3caef02c through both configuration sources."""
    for name in ("ca.pem", "cert.pem", "key.pem"):
        (tmp_path / name).write_text("dummy TLS file; no connection is made\n")
    env = {
        source: f"tcp://dummy-user@{hostname}{port}{suffix}",
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": str(tmp_path),
        "BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS": show_address,
    }
    with pytest.raises(RemoteDockerConfigError) as caught:
        resolve_remote_docker_host(env)
    assert "ASCII" in str(caught.value)
    assert "IDNA" in str(caught.value)
    assert hostname not in str(caught.value)
    assert "dummy-user" not in str(caught.value)


@pytest.mark.parametrize("hostname", _UNSTABLE_HOSTS)
@pytest.mark.parametrize("port", ["", ":", ":2376"])
@pytest.mark.parametrize("consumer", ["raw", "compose", "diagnostic"])
def test_direct_tcp_consumers_require_the_same_host_alphabet(hostname, port, consumer):
    """Guards the fix at 3caef02c even without the env resolver."""
    host = RemoteDockerHost(f"tcp://{hostname}{port}")
    with pytest.raises(RemoteDockerConfigError, match="ASCII"):
        if consumer == "raw":
            host.docker_args()
        elif consumer == "compose":
            host.compose_args({})
        else:
            host.redact("transport failed", command_output=True)
