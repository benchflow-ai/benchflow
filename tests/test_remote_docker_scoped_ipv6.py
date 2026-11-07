"""Refuse TCP zones before Docker parses them."""

from __future__ import annotations

import pytest

from benchflow.sandbox import remote_docker
from benchflow.sandbox.remote_docker import (
    RemoteDockerConfigError,
    RemoteDockerHost,
    resolve_remote_docker_host,
)

# Docker parses the endpoint twice. A canonical-looking %25 followed by a
# numeric zone prefix can therefore be another escape on the second pass.
# Refuse the whole zone class, including raw, encoded and nested spellings.
_ZONES = [
    "%eth0",
    "%25eth0",
    "%2525eth0",
    "%252525eth0",
    "%251",
    "%25ETH0",
    "%25en-0._~",
    "%25éth0",
    "%25",
    "%2525",
]
_HOSTS = [
    f"[{address}{zone}]" for address in ("fe80::44", "FE80::44") for zone in _ZONES
]


@pytest.fixture
def tls_env(tmp_path):
    for name in ("ca.pem", "cert.pem", "key.pem"):
        (tmp_path / name).write_text("dummy TLS file; no connection is made\n")
    return {"DOCKER_TLS_VERIFY": "1", "DOCKER_CERT_PATH": str(tmp_path)}


@pytest.mark.parametrize("zone", ["%", "%25%65th0", "%25eth%30"])
def test_malformed_zones_remain_refused(tls_env, zone):
    """Guards the fix at 10242cd7 without weakening parser refusals."""
    with pytest.raises(RemoteDockerConfigError):
        resolve_remote_docker_host(
            {**tls_env, "BENCHFLOW_REMOTE_DOCKER_HOST": f"tcp://[fe80::44{zone}]:2376"}
        )


@pytest.mark.parametrize("port", ["", ":", ":2376", ":02376"])
def test_double_parsed_zone_cannot_publish_certificate_names(tls_env, port):
    """Guards the fix at 10242cd7 with Docker's resulting authority."""
    try:
        host = resolve_remote_docker_host(
            {
                **tls_env,
                "BENCHFLOW_REMOTE_DOCKER_HOST": f"tcp://[fe80::44%2525eth0]{port}",
            }
        )
    except RemoteDockerConfigError as exc:
        assert "unscoped" in str(exc)
        return
    effective_port = port[1:] or "2375"
    said = (
        f'Get "https://[fe80::44%25eth0]:{effective_port}/v1.47/exec": '
        "tls: failed to verify certificate: x509: certificate is valid for "
        "private.internal.test, not fe80::44%eth0"
    )
    assert "private.internal.test" not in host.redact(said, command_output=True)


@pytest.mark.parametrize("hostname", _HOSTS)
@pytest.mark.parametrize("port", ["", ":", ":2376", ":02376"])
@pytest.mark.parametrize("source", ["BENCHFLOW_REMOTE_DOCKER_HOST", "DOCKER_HOST"])
@pytest.mark.parametrize(
    "user_suffix", [("", ""), ("dummy-user@", "/proxy?test=1#fragment")]
)
@pytest.mark.parametrize("show_address", ["0", "1"])
def test_every_scoped_tcp_host_is_refused_before_launch(
    tls_env, hostname, port, source, user_suffix, show_address
):
    """Guards the fix at 10242cd7 through both configuration sources."""
    user, suffix = user_suffix
    with pytest.raises(RemoteDockerConfigError) as caught:
        resolve_remote_docker_host(
            {
                **tls_env,
                source: f"tcp://{user}{hostname}{port}{suffix}",
                "BENCHFLOW_REMOTE_DOCKER_SHOW_ADDRESS": show_address,
            }
        )
    assert "ASCII" in str(caught.value)
    assert "use" in str(caught.value)
    assert hostname not in str(caught.value)
    assert "dummy-user" not in str(caught.value)


@pytest.mark.parametrize("hostname", _HOSTS)
@pytest.mark.parametrize("port", ["", ":", ":2376", ":02376"])
@pytest.mark.parametrize("consumer", ["raw", "compose", "diagnostic", "client-env"])
def test_direct_tcp_consumers_cannot_accept_zones(
    monkeypatch, hostname, port, consumer
):
    """Guards the fix at 10242cd7 without relying on the env resolver."""
    host = RemoteDockerHost(f"tcp://{hostname}{port}")
    with pytest.raises(RemoteDockerConfigError, match="ASCII"):
        if consumer == "raw":
            host.docker_args()
        elif consumer == "compose":
            host.compose_args({})
        elif consumer == "client-env":
            monkeypatch.setattr(
                remote_docker,
                "_real_program",
                lambda name, path=None: (
                    "/tmp/rd-dummy-docker" if name == "docker" else None
                ),
            )
            host.client_env({})
        else:
            host.redact("transport failed", command_output=True)
