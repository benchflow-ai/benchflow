"""A temporary DNS failure is not a refusal (: a 502 on GitHub's runner).

In allow mode every unmarked DNS query in the sandbox, the egress proxy's own
included, goes to the proxy's DNS filter, which forwards a listed name to the
sandbox's resolver over UDP. On a busy runner one lost or late datagram made
the filter answer SERVFAIL, the proxy's ``getaddrinfo`` fail with EAI_AGAIN,
and an allowed host's CONNECT end in ``502 Bad Gateway`` ("cannot resolve
destination"): ``test_the_allowlist_still_admits_its_host_through_the_proxy``
saw ``Tunnel connection failed: 502``. The filter now sends a query again as
a stub resolver does, and the proxy asks again after a temporary failure; a
name that does not exist is refused at once, as before.
"""

from __future__ import annotations

import socket
import threading

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy_mod
from benchflow.sandbox.egress_denylist import certificate_material

HOST = "example.test"
ADDRESS = "93.184.216.34"


@pytest.fixture
def allow_proxy(tmp_path):
    material = certificate_material((HOST,))
    certs = tmp_path / "certs"
    certs.mkdir()
    (certs / f"{HOST}.pem").write_bytes(material[f"{HOST}.pem"])
    proxy = proxy_mod.serve(
        0,
        proxy_mod.Policy([], [], allowed_hosts=[HOST]),
        proxy_mod.CertStore(str(certs)),
        proxy_mod.Log(str(tmp_path / "denied.jsonl")),
    )
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        yield proxy.server_address
    finally:
        proxy.shutdown()
        proxy.server_close()


def _resolver(monkeypatch, failures: list[int]) -> list[str]:
    """getaddrinfo that fails for HOST with the errors in *failures*, in
    turn, then answers; every other name as the real one does."""
    real = socket.getaddrinfo
    asked: list[str] = []

    def getaddrinfo(host, port, *args, **kwargs):
        if host != HOST:
            return real(host, port, *args, **kwargs)
        asked.append(host)
        if failures:
            code = failures.pop(0)
            raise socket.gaierror(code, "fake resolver failure")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ADDRESS, port))]

    monkeypatch.setattr(proxy_mod.socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(proxy_mod, "_RESOLVE_RETRY_WAITS", (0.01, 0.01), raising=False)
    return asked


def _connect_status(address) -> bytes:
    with socket.create_connection(address, timeout=20) as sock:
        sock.sendall(
            f"CONNECT {HOST}:443 HTTP/1.1\r\nHost: {HOST}:443\r\n\r\n".encode()
        )
        return sock.recv(4096).split(b"\r\n", 1)[0]


def test_a_temporary_resolver_failure_is_asked_again_not_refused(
    allow_proxy, monkeypatch
):
    asked = _resolver(monkeypatch, [socket.EAI_AGAIN])
    assert _connect_status(allow_proxy) == b"HTTP/1.1 200 Connection Established"
    assert asked == [HOST, HOST]


def test_a_temporary_failure_that_lasts_is_still_a_502(allow_proxy, monkeypatch):
    asked = _resolver(monkeypatch, [socket.EAI_AGAIN] * 10)
    assert b" 502 " in _connect_status(allow_proxy)
    assert len(asked) == 1 + len(proxy_mod._RESOLVE_RETRY_WAITS)


def test_a_name_that_does_not_exist_is_refused_without_asking_again(
    allow_proxy, monkeypatch
):
    asked = _resolver(monkeypatch, [socket.EAI_NONAME])
    assert b" 502 " in _connect_status(allow_proxy)
    assert asked == [HOST]


def _lossy_upstream(drop: int):
    """A UDP DNS stand-in on loopback that ignores the first *drop* queries
    and echoes the rest back (an answer with the query's id)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    seen: list[bytes] = []

    def run():
        while True:
            try:
                data, peer = server.recvfrom(65535)
            except OSError:
                return
            seen.append(data)
            if len(seen) > drop:
                server.sendto(data[:2] + b"\x81\x80" + data[4:], peer)

    threading.Thread(target=run, daemon=True).start()
    return server, seen


def test_the_dns_filter_sends_a_lost_query_again():
    server, seen = _lossy_upstream(drop=1)
    try:
        query = (
            b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
            + b"\x07example\x04test\x00\x00\x01\x00\x01"
        )
        reply = proxy_mod._forward_udp(query, [f"127.0.0.1:{server.getsockname()[1]}"])
        assert reply[:2] == query[:2]
        assert reply[3] & 0x0F == 0, "the filter answered SERVFAIL"
        assert len(seen) == 2
    finally:
        server.close()
