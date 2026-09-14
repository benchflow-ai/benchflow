"""Guard the CONNECT authority bypass present at BenchFlow commit b3b8afaf.

The origin deliberately shares one socket address across unrelated TLS names and
routes by HTTP Host, as a shared CDN can. No public network is needed.
"""

from __future__ import annotations

import contextlib
import json
import socket
import socketserver
import ssl
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy_mod
from benchflow.sandbox.egress_denylist import certificate_material


class _VirtualHostOrigin(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            self._serve_requests()
        finally:
            self.server.connection_closed.set()

    def _serve_requests(self):
        # Keep accepting requests even when a client asks for Connection: close.
        # Enforcement must not depend on a remote server honoring that header.
        self.connection.settimeout(2)
        while True:
            try:
                line = self.rfile.readline()
                if not line:
                    self.server.client_eof.set()
                    return
                method, path, _ = line.decode("ascii").strip().split(" ")
                headers = {}
                while line := self.rfile.readline():
                    if line == b"\r\n":
                        break
                    name, _, value = line.decode("latin-1").partition(":")
                    headers[name.lower()] = value.strip()
                if headers.get("transfer-encoding", "").lower() == "chunked":
                    chunks = []
                    while True:
                        length = int(self.rfile.readline().split(b";", 1)[0], 16)
                        if not length:
                            while self.rfile.readline() != b"\r\n":
                                pass
                            break
                        chunks.append(self.rfile.read(length))
                        assert self.rfile.read(2) == b"\r\n"
                    body = b"".join(chunks)
                else:
                    length = int(headers.get("content-length", "0"))
                    body = self.rfile.read(min(length, 3))
                    if body:
                        self.server.body_prefix_seen.set()
                    body += self.rfile.read(length - len(body))
                record = {
                    "method": method,
                    "path": path,
                    "host": headers["host"],
                    "body": body.decode("ascii"),
                }
                self.server.requests.append(record)
                if record["host"] == "paper.test" and path == "/protected":
                    self.server.protected_seen.set()
                payload = json.dumps(record).encode("ascii")
                self.wfile.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: "
                    + str(len(payload)).encode("ascii")
                    + b"\r\nContent-Type: application/json\r\n\r\n"
                    + payload
                )
                self.wfile.flush()
            except (OSError, ValueError):
                return


class _Origin(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture
def authority_stack(tmp_path: Path, monkeypatch):
    """Guard commit b3b8afaf with a local origin that supports CDN host fronting."""
    origin_material = certificate_material(("paper.test", "carrier.test"))
    proxy_material = certificate_material(("paper.test", "carrier.test"))
    origin_ca = tmp_path / "origin-ca.crt"
    origin_ca.write_bytes(origin_material["ca.crt"])
    proxy_ca = tmp_path / "proxy-ca.crt"
    proxy_ca.write_bytes(proxy_material["ca.crt"])
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    for host in ("paper.test", "carrier.test"):
        (cert_dir / f"{host}.pem").write_bytes(proxy_material[f"{host}.pem"])
    origin_contexts = {}
    for host in ("paper.test", "carrier.test"):
        pem = tmp_path / f"origin-{host}.pem"
        pem.write_bytes(origin_material[f"{host}.pem"])
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(pem))
        origin_contexts[host] = ctx

    def select_origin_certificate(sock, hostname, context):
        sock.context = origin_contexts.get(hostname, context)

    origin_ctx = origin_contexts["carrier.test"]
    origin_ctx.set_servername_callback(select_origin_certificate)
    origin = _Origin(("127.0.0.1", 0), _VirtualHostOrigin)
    origin.requests = []
    origin.protected_seen = threading.Event()
    origin.body_prefix_seen = threading.Event()
    origin.client_eof = threading.Event()
    origin.connection_closed = threading.Event()
    origin.socket = origin_ctx.wrap_socket(origin.socket, server_side=True)

    def connect_shared_origin(host, port, *, model_gateway_port=None):
        assert host in {"paper.test", "carrier.test"}
        assert port == 443
        return socket.create_connection(origin.server_address, timeout=3)

    monkeypatch.setattr(proxy_mod, "_connect_upstream", connect_shared_origin)
    monkeypatch.setattr(proxy_mod, "_resolve", lambda host, port: ["93.184.216.34"])
    log = tmp_path / "denied.jsonl"
    proxy = proxy_mod.serve(
        0,
        proxy_mod.Policy(["https://paper.test/protected"], []),
        proxy_mod.CertStore(str(cert_dir)),
        proxy_mod.Log(str(log)),
        upstream_ca=str(origin_ca),
    )
    handler_finished = threading.Event()
    original_process_request = proxy.process_request_thread

    def observe_handler_release(request, client_address):
        try:
            original_process_request(request, client_address)
        finally:
            handler_finished.set()

    proxy.process_request_thread = observe_handler_release
    for server in (origin, proxy):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(
        address=proxy.server_address,
        # Trust only the proxy CA, proving that allowed hosts are intercepted.
        client_ctx=ssl.create_default_context(cafile=str(proxy_ca)),
        requests=origin.requests,
        protected_seen=origin.protected_seen,
        body_prefix_seen=origin.body_prefix_seen,
        client_eof=origin.client_eof,
        origin_closed=origin.connection_closed,
        handler_finished=handler_finished,
        log=log,
    )
    for server in (proxy, origin):
        server.shutdown()
        server.server_close()


def _read_response(sock):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("response ended before its headers")
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    fields = head.decode("latin-1").split("\r\n")
    status = int(fields[0].split(" ", 2)[1])
    headers = dict(line.lower().split(": ", 1) for line in fields[1:])
    length = int(headers.get("content-length", "0"))
    while len(body) < length:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("response ended before its body")
        body += chunk
    return status, headers, body[:length]


@contextlib.contextmanager
def _tunnel(stack, *, connect="carrier.test", sni="carrier.test", verify=True):
    raw = socket.create_connection(stack.address, timeout=3)
    raw.settimeout(3)
    try:
        raw.sendall(
            f"CONNECT {connect}:443 HTTP/1.1\r\nHost: {connect}:443\r\n\r\n".encode()
        )
        assert _read_response(raw)[0] == 200
        context = stack.client_ctx if verify else ssl._create_unverified_context()
        with context.wrap_socket(raw, server_hostname=sni) as tls:
            tls.settimeout(3)
            yield tls
    finally:
        raw.close()


def _get(host="carrier.test", path="/ordinary"):
    return f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode()


def test_connect_sni_cannot_select_protected_virtual_host(authority_stack):
    """Guards the unlisted CONNECT plus protected SNI bypass in commit b3b8afaf."""
    with (
        pytest.raises((ssl.SSLError, ConnectionError, OSError)),
        _tunnel(authority_stack, sni="paper.test", verify=False) as tls,
    ):
        tls.sendall(_get("paper.test", "/protected"))
        _read_response(tls)
    assert authority_stack.requests == []
    events = [json.loads(line) for line in authority_stack.log.read_text().splitlines()]
    assert any(event["rule"] == "tls-sni-mismatch" for event in events)


def test_same_sni_cannot_front_a_different_http_host(authority_stack):
    """Guards HTTP Host fronting with certificate verification off at commit b3b8afaf."""
    with _tunnel(authority_stack, verify=False) as tls:
        tls.sendall(_get("paper.test", "/protected"))
        code, headers, _ = _read_response(tls)
    assert code == 403
    assert headers["x-benchflow-blocked"] == "1"
    assert authority_stack.requests == []


def test_ordinary_unlisted_https_remains_readable(authority_stack):
    """Guards allowed HTTPS access while closing the opaque tunnel in commit b3b8afaf."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(_get())
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body) == {
        "method": "GET",
        "host": "carrier.test",
        "path": "/ordinary",
        "body": "",
    }


def test_allowed_sibling_of_blocked_paper_remains_readable(authority_stack):
    """Guards path-specific blocking from commit b3b8afaf when all TLS is inspected."""
    with _tunnel(authority_stack, connect="paper.test", sni="paper.test") as tls:
        tls.sendall(_get("paper.test", "/allowed-paper"))
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["path"] == "/allowed-paper"


def test_absolute_form_target_must_match_connect_authority(authority_stack):
    """Guards absolute-URI authority ambiguity at commit b3b8afaf."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(_get(path="https://paper.test/protected"))
        assert _read_response(tls)[0] == 403
    assert authority_stack.requests == []


@pytest.mark.parametrize("host,expected", [("carrier.test", 200), ("paper.test", 403)])
def test_no_sni_cannot_change_connect_authority(authority_stack, host, expected):
    """Guards the absent-SNI variant of the authority bypass in commit b3b8afaf."""
    with _tunnel(authority_stack, sni=None, verify=False) as tls:
        tls.sendall(_get(host, "/protected"))
        assert _read_response(tls)[0] == expected
    assert not authority_stack.protected_seen.is_set()


@pytest.mark.parametrize(
    "extra_headers",
    [
        "Host: paper.test\r\n",
        "Content-Length: 0\r\nTransfer-Encoding: chunked\r\n",
        "Content-Length: 0\r\nContent-Length: 1\r\n",
        "Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked\r\n",
        "Transfer-Encoding: gzip\r\n",
    ],
)
def test_ambiguous_request_headers_fail_closed(authority_stack, extra_headers):
    """Guards request-smuggling authority changes through commit b3b8afaf's tunnel."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            (
                "POST /ordinary HTTP/1.1\r\nHost: carrier.test\r\n"
                + extra_headers
                + "\r\n"
            ).encode()
        )
        assert _read_response(tls)[0] == 400
    assert authority_stack.requests == []


@pytest.mark.parametrize("chunked", [False, True])
def test_post_request_body_reaches_allowed_origin(authority_stack, chunked):
    """Guards streaming HTTP bodies after replacing commit b3b8afaf's raw relay."""
    framing = "Transfer-Encoding: chunked" if chunked else "Content-Length: 6"
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            f"POST /upload HTTP/1.1\r\nHost: carrier.test\r\n{framing}\r\n\r\n".encode()
        )
        tls.sendall(b"3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n" if chunked else b"abcdef")
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["body"] == "abcdef"


@pytest.mark.parametrize("connect", ["carrier.test", "paper.test"])
def test_late_pipelined_authority_switch_never_reaches_origin(authority_stack, connect):
    """Guards a second request bypassing the checked headers at commit b3b8afaf."""
    with _tunnel(authority_stack, connect=connect, sni=connect, verify=False) as tls:
        tls.sendall(_get(connect))
        assert _read_response(tls)[0] == 200
        with contextlib.suppress(OSError):
            tls.sendall(_get("paper.test", "/protected"))
        assert not authority_stack.protected_seen.wait(0.2)
    assert len(authority_stack.requests) == 1


def test_chunked_body_cannot_carry_a_pipelined_authority_switch(authority_stack):
    """Guards chunked body over-read in the opaque relay at commit b3b8afaf."""
    with _tunnel(authority_stack, verify=False) as tls:
        tls.sendall(
            b"POST /upload HTTP/1.1\r\nHost: carrier.test\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"3\r\nabc\r\n0\r\n\r\n" + _get("paper.test", "/protected")
        )
        code, _, body = _read_response(tls)
        assert code == 200
        assert json.loads(body)["body"] == "abc"
        assert not authority_stack.protected_seen.wait(0.2)
    assert len(authority_stack.requests) == 1


@pytest.mark.parametrize("expectation", ["100-continue", "100-Continue"])
def test_expect_continue_allows_a_fragmented_post_body(authority_stack, expectation):
    """Guards normal waiting POST clients when fixing commit b3b8afaf's raw relay."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(
            (
                "POST /upload HTTP/1.1\r\nHost: carrier.test\r\n"
                "Content-Length: 6\r\nExpect: " + expectation + "\r\n\r\n"
            ).encode()
        )
        # A client that waits for 100 must not deadlock with the body copier.
        assert _read_response(tls)[0] == 100
        tls.sendall(b"abc")
        # Ensure the first fragment actually reaches the origin before the
        # second is sent, rather than relying on TCP packet boundaries.
        assert authority_stack.body_prefix_seen.wait(1)
        tls.sendall(b"def")
        code, _, body = _read_response(tls)
    assert code == 200
    assert json.loads(body)["body"] == "abcdef"


def test_client_close_releases_checked_upstream_without_forwarding_pipeline(
    authority_stack,
):
    """Guards PR #1122 against the connection leak introduced by commit 10090f48."""
    with _tunnel(authority_stack) as tls:
        tls.sendall(_get())
        code, _, body = _read_response(tls)
        assert code == 200 and json.loads(body)["path"] == "/ordinary"
        # The origin remains ready for another request despite Connection: close.
        # Extra client bytes must be discarded, while its eventual EOF must be
        # propagated so neither the origin socket nor proxy handler is leaked.
        tls.sendall(_get("paper.test", "/protected"))
    assert authority_stack.client_eof.wait(0.75), (
        "origin still waiting after client closed"
    )
    assert authority_stack.origin_closed.wait(0.75), (
        "origin connection was not released"
    )
    assert authority_stack.handler_finished.wait(0.75), "proxy handler was not released"
    assert not authority_stack.protected_seen.is_set()
    assert len(authority_stack.requests) == 1


def test_http_request_half_close_preserves_complete_response(
    authority_stack, monkeypatch
):
    """Guards PR #1122 cleanup after 10090f48 without breaking request half-close."""
    origin = _Origin(("127.0.0.1", 0), _VirtualHostOrigin)
    origin.requests = authority_stack.requests
    origin.protected_seen = authority_stack.protected_seen
    origin.body_prefix_seen = authority_stack.body_prefix_seen
    origin.client_eof = authority_stack.client_eof
    origin.connection_closed = authority_stack.origin_closed

    def connect_plain_origin(host, port, *, model_gateway_port=None):
        assert (host, port) == ("carrier.test", 80)
        return socket.create_connection(origin.server_address, timeout=3)

    monkeypatch.setattr(proxy_mod, "_connect_upstream", connect_plain_origin)
    threading.Thread(target=origin.serve_forever, daemon=True).start()
    try:
        with socket.create_connection(authority_stack.address, timeout=3) as client:
            client.sendall(_get(path="http://carrier.test/ordinary"))
            # HTTP permits request-side EOF while the client is still reading
            # its response. Cleanup must not close the upstream read side.
            client.shutdown(socket.SHUT_WR)
            code, _, body = _read_response(client)
        assert code == 200
        assert json.loads(body) == {
            "method": "GET",
            "host": "carrier.test",
            "path": "/ordinary",
            "body": "",
        }
        assert authority_stack.client_eof.wait(0.75)
        assert authority_stack.handler_finished.wait(0.75)
    finally:
        origin.shutdown()
        origin.server_close()
