"""Loopback egress filter: the sandbox agent's only road to the internet.

Runs inside the sandbox next to the LiteLLM proxy, using nothing but the
standard library, so any backend that can run Python can run it. The agent
process receives ``HTTP_PROXY``/``HTTPS_PROXY`` pointing here and a uid
firewall that rejects every other destination, so each request the agent or
one of its tools makes passes through :meth:`Policy.decide`.

Plain HTTP requests are decided on the full URL. A ``CONNECT`` tunnel is
decided on the name the client's TLS handshake asks for (the SNI), never on
the address it asked to connect to, and the upstream connection goes to
that name: an alias, a CNAME, or a numeric address cannot stand in for a
blocked host. When the policy carries path rules for that name the tunnel
is terminated with a per-run certificate so the path becomes visible;
otherwise the bytes are relayed untouched. Tunnels that do not start with a
TLS handshake are closed. Every decision is appended to a JSON-lines log
that the host collects into the rollout.

Inside the sandbox::

    python egress_filter.py launch launch.json   # detach a server
    python egress_filter.py serve  launch.json   # run in the foreground
"""

from __future__ import annotations

import http.client
import json
import os
import re
import selectors
import socket
import ssl
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from posixpath import normpath
from typing import Any
from urllib.parse import unquote, urlsplit

UPSTREAM_TIMEOUT_SEC = 60
#: A connection with no traffic in either direction for this long is closed.
IDLE_TIMEOUT_SEC = 60
BLOCKED_STATUS = 403
BLOCKED_HEADER = "X-BenchFlow-Network-Policy"
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_CHUNK = 65536
_TLS_HANDSHAKE = 0x16
_TLS_CLIENT_HELLO = 0x01
_TLS_EXTENSION_SERVER_NAME = 0x0000


def canonical_path(path: str) -> str:
    """The path a server would resolve: decoded, dot segments collapsed.

    ``/abs/%32401.01234`` and ``/pdf/../abs/2401.01234`` name the same
    resource as ``/abs/2401.01234``; rules must see them the same way. A
    trailing slash is kept, so ``/simple/requests/`` stays a directory prefix.
    """
    path, query_mark, query = path.partition("?")
    decoded = unquote(path)
    collapsed = "/" + normpath(decoded or "/").lstrip("/")
    if decoded.endswith("/") and collapsed != "/":
        collapsed += "/"
    return f"{collapsed}{query_mark}{unquote(query)}"


def host_within(host: str, domain: str) -> bool:
    """Whether ``host`` is ``domain`` or one of its subdomains."""
    return host == domain or host.endswith(f".{domain}")


def is_numeric_host(host: str) -> bool:
    """Whether ``host`` is an address in any spelling libc accepts, not a name."""
    try:
        socket.getaddrinfo(host.strip("[]"), None, flags=socket.AI_NUMERICHOST)
    except (OSError, UnicodeError):
        return False
    return True


@dataclass(frozen=True)
class Rule:
    """One policy entry: a host, or a host plus a path prefix.

    ``arxiv.org`` covers the host and its subdomains; ``arxiv.org/abs/2401``
    covers every path starting with ``/abs/2401`` on them. The query string
    is part of the path, so ``openreview.net/forum?id=X`` is one entry.
    """

    host: str
    path: str = ""

    @classmethod
    def parse(cls, entry: str) -> Rule:
        """Read ``host``, ``host/path``, or an http(s) URL; anything else raises."""
        text = entry.strip()
        url = urlsplit(text if _SCHEME.match(text) else f"//{text}")
        if url.scheme and url.scheme.lower() not in {"http", "https"}:
            raise ValueError(f"{entry!r}: only http and https URLs can be rules")
        if url.port not in {None, 80, 443}:
            raise ValueError(f"{entry!r}: rules cannot name a port")
        host = (url.hostname or "").rstrip(".")
        if not host:
            raise ValueError(f"{entry!r}: a rule starts with a hostname")
        path = ""
        if url.path or url.query:
            path = canonical_path(url.path + (f"?{url.query}" if url.query else ""))
        return cls(host, "" if path == "/" else path)

    def matches(self, host: str, path: str) -> bool:
        """``path`` must already be canonical."""
        return host_within(host, self.host) and path.startswith(self.path)

    def __str__(self) -> str:
        return f"{self.host}{self.path}"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    rule: Rule | None


class Policy:
    """The resolved network policy: a mode and its rules."""

    def __init__(self, mode: str, rules: Iterable[Rule]) -> None:
        if mode not in {"blocklist", "allowlist"}:
            raise ValueError(f"Unsupported network policy mode: {mode!r}")
        self.mode = mode
        self.rules = tuple(rules)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Policy:
        return cls(payload["mode"], (Rule.parse(entry) for entry in payload["rules"]))

    def decide(self, host: str, path: str) -> Decision:
        host = host.lower().rstrip(".")
        path = canonical_path(path)
        match = next((rule for rule in self.rules if rule.matches(host, path)), None)
        allowed = match is None if self.mode == "blocklist" else match is not None
        return Decision(allowed, match)

    @property
    def inspected_hosts(self) -> tuple[str, ...]:
        """Hosts whose TLS the filter terminates, so the certificate can name them."""
        if self.mode != "blocklist":
            return ()
        return tuple(sorted({rule.host for rule in self.rules if rule.path}))

    def inspects(self, host: str) -> bool:
        """Whether tunnels to ``host`` must be opened to see the request path."""
        host = host.lower().rstrip(".")
        return any(host_within(host, domain) for domain in self.inspected_hosts)


class EgressFilter(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        policy: Policy,
        log_path: str,
        certificate: tuple[str, str] | None = None,
    ) -> None:
        super().__init__(address, _ProxyHandler)
        self.policy = policy
        # The log names the rules; only the filter's own user may read it.
        self._log = open(  # noqa: SIM115
            os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600),
            "a",
            encoding="utf-8",
        )
        self._log_lock = threading.Lock()
        self.server_context: ssl.SSLContext | None = None
        if certificate is not None:
            self.server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self.server_context.load_cert_chain(*certificate)

    def record(
        self,
        *,
        method: str,
        host: str,
        path: str | None,
        decision: str,
        rule: Rule | None = None,
    ) -> None:
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": method,
            "host": host,
            "path": path,
            "decision": decision,
            "rule": None if rule is None else str(rule),
        }
        with self._log_lock:
            self._log.write(json.dumps(entry) + "\n")
            self._log.flush()

    def server_close(self) -> None:
        super().server_close()
        self._log.close()


class _BoundedBody:
    """Exactly ``length`` bytes of a request body, as a file-like object."""

    def __init__(self, source: Any, length: int) -> None:
        self._source = source
        self._remaining = length

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        size = self._remaining if size < 0 else min(size, self._remaining)
        chunk = self._source.read(size)
        self._remaining -= len(chunk)
        return chunk


class _ProxyHandler(BaseHTTPRequestHandler):
    """One client connection: forward proxy requests, or open a tunnel."""

    protocol_version = "HTTP/1.1"
    timeout = IDLE_TIMEOUT_SEC
    server: EgressFilter
    scheme = "http"
    _headers_sent = False

    def log_message(self, format: str, *args: Any) -> None:
        return

    # Ordinary requests: absolute URLs from a proxy-aware client, or plain
    # paths on a connection whose TLS we terminated.
    def _proxy(self) -> None:
        target = self._target()
        if target is None:
            self._reply(400, "benchflow egress filter: request has no host")
            return
        scheme, host, port, path = target
        if is_numeric_host(host):
            self.server.record(
                method=self.command, host=host, path=path, decision="block"
            )
            self._refuse()
            return
        decision = self.server.policy.decide(host, path)
        # Allowed requests are logged without their query string: a signed
        # URL or an API token in it is the agent's business, not the log's.
        self.server.record(
            method=self.command,
            host=host,
            path=_without_query(path) if decision.allowed else path,
            decision="allow" if decision.allowed else "block",
            rule=decision.rule,
        )
        if not decision.allowed:
            self._refuse()
            return
        self._forward(scheme, host, port, path)

    do_GET = do_HEAD = do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _proxy

    def do_CONNECT(self) -> None:
        authority, port = _authority(self.path, 443)
        if is_numeric_host(authority):
            self.server.record(
                method="CONNECT", host=authority, path=None, decision="block"
            )
            self._refuse()
            return
        early = self.server.policy.decide(authority, "/")
        if not early.allowed and not self.server.policy.inspects(authority):
            # A name the policy refuses outright gets a clear 403 up front.
            self.server.record(
                method="CONNECT",
                host=authority,
                path=None,
                decision="block",
                rule=early.rule,
            )
            self._refuse()
            return
        self.send_response(200, "Connection established")
        self.end_headers()
        self.close_connection = True
        host = _peek_server_name(self.connection)
        if host is None or is_numeric_host(host):
            # No TLS handshake, or one that names no host: nothing to decide on.
            self.server.record(
                method="CONNECT", host=authority, path=None, decision="block"
            )
            return
        if self.server.policy.inspects(host):
            self.server.record(
                method="CONNECT", host=host, path=None, decision="inspect"
            )
            if self.server.server_context is None:
                return  # never relay blind what the policy says must be inspected
            self._inspect()
            return
        decision = self.server.policy.decide(host, "/")
        self.server.record(
            method="CONNECT",
            host=host,
            path=None,
            decision="allow" if decision.allowed else "block",
            rule=decision.rule,
        )
        if decision.allowed:
            self._tunnel(host, port)

    def _target(self) -> tuple[str, str, int, str] | None:
        if _SCHEME.match(self.path):
            url = urlsplit(self.path)
            if not url.hostname:
                return None
            default_port = 443 if url.scheme == "https" else 80
            path = url.path or "/"
            if url.query:
                path = f"{path}?{url.query}"
            return url.scheme, url.hostname.lower(), url.port or default_port, path
        authority = self.headers.get("Host")
        if not authority:
            return None
        host, port = _authority(authority, 443 if self.scheme == "https" else 80)
        return self.scheme, host, port, self.path

    def _forward(self, scheme: str, host: str, port: int, path: str) -> None:
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self._reply(411, "benchflow egress filter: send a Content-Length")
            return
        declared = self.headers.get("Content-Length", "0").strip()
        if not declared.isdigit():
            self._reply(400, "benchflow egress filter: bad Content-Length")
            return
        length = int(declared)
        headers = _end_to_end(self.headers)
        headers.pop("Content-Length", None)
        if length:
            headers["Content-Length"] = str(length)
        # The decided host is the one we talk to; the client's Host header
        # must not point the upstream elsewhere.
        headers["Host"] = host if port in (80, 443) else f"{host}:{port}"
        headers["Connection"] = "close"
        if scheme == "https":
            upstream: http.client.HTTPConnection = http.client.HTTPSConnection(
                host,
                port,
                context=ssl.create_default_context(),
                timeout=UPSTREAM_TIMEOUT_SEC,
            )
        else:
            upstream = http.client.HTTPConnection(
                host, port, timeout=UPSTREAM_TIMEOUT_SEC
            )
        try:
            body = _BoundedBody(self.rfile, length) if length else None
            upstream.request(self.command, path, body=body, headers=headers)
            self._stream(upstream.getresponse())
        except (OSError, http.client.HTTPException) as exc:
            if not self._headers_sent:
                self._reply(502, f"benchflow egress filter: upstream failure: {exc}")
        finally:
            upstream.close()
            self.close_connection = True

    def _stream(self, response: http.client.HTTPResponse) -> None:
        """Relay status, headers, and body as they arrive; never buffer a file."""
        has_body = self.command != "HEAD" and response.status not in (204, 304)
        chunked = has_body and (
            bool(getattr(response, "chunked", False))
            or response.getheader("Content-Length") is None
        )
        self.send_response(response.status, response.reason)
        for key, value in _end_to_end(response.msg).items():
            if not (chunked and key.lower() == "content-length"):
                self.send_header(key, value)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self._headers_sent = True
        if not has_body:
            return
        read = getattr(response, "read1", response.read)
        while chunk := read(_CHUNK):
            if chunked:
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            else:
                self.wfile.write(chunk)
            self.wfile.flush()
        if chunked:
            self.wfile.write(b"0\r\n\r\n")

    def _inspect(self) -> None:
        """Terminate the client's TLS and serve its requests on this thread."""
        context: ssl.SSLContext = self.server.server_context  # type: ignore[assignment]
        try:
            with context.wrap_socket(self.connection, server_side=True) as secured:
                _TunnelHandler(secured, self.client_address, self.server)
        except (ssl.SSLError, OSError):
            return

    def _tunnel(self, host: str, port: int) -> None:
        try:
            upstream = socket.create_connection(
                (host, port), timeout=UPSTREAM_TIMEOUT_SEC
            )
        except OSError:
            return
        with upstream:
            _relay(self.connection, upstream)

    def _refuse(self) -> None:
        # The rule that matched stays in the log, not in the agent's face.
        self.send_response(BLOCKED_STATUS)
        self.send_header(BLOCKED_HEADER, "blocked")
        self._finish_text("benchflow: blocked by network policy\n")

    def _reply(self, status: int, text: str) -> None:
        self.send_response(status)
        self._finish_text(text + "\n")

    def _finish_text(self, text: str) -> None:
        payload = text.encode()
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True


class _TunnelHandler(_ProxyHandler):
    scheme = "https"


def _without_query(path: str) -> str:
    return path.split("?", 1)[0]


def _end_to_end(headers: Any) -> dict[str, str]:
    """Headers minus hop-by-hop ones, including those ``Connection`` names."""
    named = {name.strip().lower() for name in headers.get("Connection", "").split(",")}
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP and key.lower() not in named
    }


def _authority(value: str, default_port: int) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    if host and port.isdigit():
        return host.strip("[]").lower().rstrip("."), int(port)
    return value.strip("[]").lower().rstrip("."), default_port


def _peek_server_name(sock: socket.socket) -> str | None:
    """The SNI of the ClientHello waiting on ``sock``, without consuming it."""
    deadline = time.monotonic() + IDLE_TIMEOUT_SEC
    while time.monotonic() < deadline:
        try:
            data = sock.recv(_CHUNK, socket.MSG_PEEK)
        except OSError:
            return None
        if not data:
            return None
        if len(data) >= 5 and len(data) >= 5 + int.from_bytes(data[3:5], "big"):
            return server_name_from_client_hello(data)
        if len(data) >= 5 and data[0] != _TLS_HANDSHAKE:
            return None
        time.sleep(0.01)
    return None


def server_name_from_client_hello(data: bytes) -> str | None:
    """Parse the ``server_name`` extension out of a TLS ClientHello record."""
    try:
        if data[0] != _TLS_HANDSHAKE or data[5] != _TLS_CLIENT_HELLO:
            return None
        cursor = 5 + 4 + 2 + 32  # handshake header, version, random
        cursor += 1 + data[cursor]  # session id
        cursor += 2 + int.from_bytes(data[cursor : cursor + 2], "big")  # cipher suites
        cursor += 1 + data[cursor]  # compression methods
        end = cursor + 2 + int.from_bytes(data[cursor : cursor + 2], "big")
        cursor += 2
        while cursor + 4 <= end:
            kind = int.from_bytes(data[cursor : cursor + 2], "big")
            size = int.from_bytes(data[cursor + 2 : cursor + 4], "big")
            cursor += 4
            if kind == _TLS_EXTENSION_SERVER_NAME:
                names = cursor + 2
                while names + 3 <= cursor + size:
                    name_type = data[names]
                    name_size = int.from_bytes(data[names + 1 : names + 3], "big")
                    names += 3
                    if name_type == 0:
                        return (
                            data[names : names + name_size]
                            .decode("ascii")
                            .lower()
                            .rstrip(".")
                        )
                    names += name_size
                return None
            cursor += size
    except (IndexError, UnicodeDecodeError):
        return None
    return None


def _relay(client: socket.socket, upstream: socket.socket) -> None:
    """Copy bytes both ways until either side closes or the tunnel idles."""
    pairs = {client: upstream, upstream: client}
    with selectors.DefaultSelector() as selector:
        for sock in pairs:
            sock.settimeout(IDLE_TIMEOUT_SEC)
            selector.register(sock, selectors.EVENT_READ)
        while True:
            ready = selector.select(timeout=IDLE_TIMEOUT_SEC)
            if not ready:
                return
            for key, _ in ready:
                source: socket.socket = key.fileobj  # type: ignore[assignment]
                try:
                    chunk = source.recv(_CHUNK)
                    if not chunk:
                        return
                    pairs[source].sendall(chunk)
                except OSError:
                    return


# Sandbox entry points.


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_ca_bundle(ca_path: str, bundle_path: str) -> None:
    """The system trust store plus the run's CA, for tools that take one file."""
    candidates = [
        ssl.get_default_verify_paths().cafile,
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/cert.pem",
    ]
    system = next((p for p in candidates if p and os.path.exists(p)), None)
    if system is None:
        raise RuntimeError(
            "no system CA bundle found; the filter would break every HTTPS request"
        )
    with open(bundle_path, "wb") as bundle:
        with open(system, "rb") as source:
            bundle.write(source.read().rstrip(b"\n") + b"\n")
        with open(ca_path, "rb") as source:
            bundle.write(source.read())
    os.chmod(bundle_path, 0o644)


def serve(config: dict[str, Any]) -> None:
    os.umask(0o077)
    policy = Policy.from_json(_load(config["policy"]))
    certificate = None
    if config.get("cert") and config.get("key"):
        certificate = (config["cert"], config["key"])
    server = EgressFilter(
        ("127.0.0.1", 0), policy=policy, log_path=config["log"], certificate=certificate
    )
    with open(config["state"], "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "port": server.server_port}, handle)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def launch(config_path: str) -> None:
    os.umask(0o077)
    config = _load(config_path)
    if config.get("ca") and config.get("ca_bundle"):
        _write_ca_bundle(config["ca"], config["ca_bundle"])
    process = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "serve", config_path],
        stdin=subprocess.DEVNULL,
        stdout=open(config["stdout"], "ab"),  # noqa: SIM115
        stderr=open(config["stderr"], "ab"),  # noqa: SIM115
        start_new_session=True,
    )
    with open(config["pid"], "w", encoding="utf-8") as handle:
        handle.write(str(process.pid))
    print(json.dumps({"pid": process.pid}))


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {"launch", "serve"}:
        print("usage: egress_filter.py {launch|serve} launch.json", file=sys.stderr)
        return 2
    if argv[1] == "launch":
        launch(argv[2])
    else:
        serve(_load(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
