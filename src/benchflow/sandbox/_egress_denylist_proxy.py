"""Loopback egress proxy for network_mode='denylist'. Stdlib only; runs inside the sandbox as root.

Hosts named in ``blocked_urls`` are TLS-intercepted with pre-generated
certificates so the full path is visible; every other host passes through as
an opaque CONNECT tunnel.
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import re
import socket
import socketserver
import ssl
import sys
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

HEAD_LIMIT = 64 * 1024
HEAD_TIMEOUT = 30
IDLE_TIMEOUT = 900
BLOCK_BODY = "Blocked by the task network policy: {url}\n"


def host_key(host: str) -> str:
    """Comparison form of a hostname: lowercase, no trailing dot, no leading www."""
    host = host.strip().rstrip(".").lower()
    return host[4:] if host.startswith("www.") else host


def _path_key(path: str) -> str:
    """Comparison form of a path: decoded, dot segments resolved, one slash between segments."""
    path = path.split("?", 1)[0].split("#", 1)[0]
    for _ in range(3):
        decoded = urllib.parse.unquote(path)
        if decoded == path:
            break
        path = decoded
    path = path.replace("\\", "/")
    segments: list[str] = []
    for segment in path.split("/"):
        segment = segment.split(";", 1)[0]
        if segment in ("", "."):
            continue
        if segment == "..":
            if segments:
                segments.pop()
            continue
        segments.append(segment)
    key = "/" + "/".join(segments)
    if path.endswith(("/", "/.", "/..")) and key != "/":
        key += "/"
    return key.lower()


_EMBEDDED_IPV4 = re.compile(r"(?:^|[.-])(?:\d{1,3}[.-]){3}\d{1,3}(?:[.-]|$)")
_HEX_IPV4_LABEL = re.compile(r"^[0-9a-f]{8}$")
_WILDCARD_DNS = (
    "nip.io",
    "sslip.io",
    "xip.io",
    "traefik.me",
    "localtest.me",
    "lvh.me",
    "vcap.me",
)


def _looks_like_address(host: str) -> bool:
    """True for anything that names an address rather than a site.

    glibc accepts decimal, octal, hex and short dotted forms (``3232235777``,
    ``0xc0a80101``, ``0300.0250.1.1``, ``127.1``), and wildcard DNS services
    resolve an address embedded in the name (``1-2-3-4.sslip.io``).
    """
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    labels = host.split(".")
    if not any(c.isalpha() for c in host):
        return True
    if labels[-1].isdigit():
        return True
    if any(label.startswith("0x") or _HEX_IPV4_LABEL.match(label) for label in labels):
        return True
    if _EMBEDDED_IPV4.search(host):
        return True
    return any(host == d or host.endswith("." + d) for d in _WILDCARD_DNS)


class Policy:
    """Match hosts and URLs against the denylist; scheme, port and query are ignored."""

    def __init__(self, blocked_urls: list[str], blocked_hosts: list[str]):
        self.prefixes: list[tuple[str, str]] = []
        for raw in blocked_urls:
            url = raw if "://" in raw else "https://" + raw
            parts = urllib.parse.urlsplit(url)
            if not parts.hostname:
                raise ValueError(f"blocked_urls entry has no host: {raw!r}")
            prefix = _path_key(parts.path).rstrip("/") or "/"
            self.prefixes.append((host_key(parts.hostname), prefix))
        self.hosts = {h.strip().rstrip(".").lower() for h in blocked_hosts if h.strip()}
        self.inspect_hosts = {host for host, _ in self.prefixes}

    @classmethod
    def load(cls, path: str) -> Policy:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            list(data.get("blocked_urls") or []),
            list(data.get("blocked_hosts") or []),
        )

    def host_rule(self, host: str) -> str | None:
        name = host.strip().rstrip(".").lower()
        if _looks_like_address(name):
            return "ip-literal"
        for blocked in self.hosts:
            if name == blocked or name.endswith("." + blocked):
                return f"host:{blocked}"
        return None

    def url_rule(self, host: str, path: str) -> str | None:
        rule = self.host_rule(host)
        if rule:
            return rule
        key, pkey = host_key(host), _path_key(path)
        for bhost, bpath in self.prefixes:
            if key == bhost and pkey.startswith(bpath):
                return f"url:{bhost}{bpath}"
        return None

    def inspect(self, host: str) -> bool:
        return host_key(host) in self.inspect_hosts


class CertStore:
    """Server TLS contexts for intercepted hosts, from ``<cert_dir>/<host>.pem`` (cert + key)."""

    def __init__(self, cert_dir: str):
        self.cert_dir = Path(cert_dir)
        self._lock = threading.Lock()
        self._contexts: dict[str, ssl.SSLContext] = {}

    def context_for(self, host: str) -> ssl.SSLContext:
        key = host_key(host)
        with self._lock:
            ctx = self._contexts.get(key)
            if ctx is None:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(str(self.cert_dir / f"{key}.pem"))
                ctx.set_alpn_protocols(["http/1.1"])
                self._contexts[key] = ctx
            return ctx


class Log:
    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()

    def write(self, **fields: object) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017
        line = json.dumps({"ts": stamp, **fields}, sort_keys=True)
        with self._lock:
            if self.path:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            else:
                print(line, file=sys.stderr, flush=True)


def _read_head(sock: socket.socket) -> bytes:
    buf = b""
    sock.settimeout(HEAD_TIMEOUT)
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("client closed before request head")
        buf += chunk
        if len(buf) > HEAD_LIMIT:
            raise ConnectionError("request head too large")
    return buf


def _parse_head(head: bytes) -> tuple[str, str, str, list[tuple[str, str]], bytes]:
    raw, _, rest = head.partition(b"\r\n\r\n")
    lines = raw.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3:
        raise ConnectionError(f"bad request line {lines[0]!r}")
    headers = []
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers.append((name.strip(), value.strip()))
    return parts[0], parts[1], parts[2], headers, rest


_HOP_HEADERS = {"proxy-connection", "proxy-authorization", "connection", "keep-alive"}
_ABSOLUTE_FORM = re.compile(r"^https?://", re.IGNORECASE)


class _PrivateDestination(Exception):
    """The destination resolves to a loopback, private or otherwise non-global address."""


def _resolve(host: str, port: int) -> list[str]:
    addresses: dict[str, None] = {}
    for _family, _type, _proto, _name, sockaddr in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        addresses.setdefault(str(sockaddr[0]))
    return list(addresses)


def _upstream_allowed(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_global
    except ValueError:
        return False


def _connect_upstream(host: str, port: int) -> socket.socket:
    """Connect to a vetted address of ``host``; the root proxy must not reach sandbox-internal services."""
    addresses = _resolve(host, port)
    if not addresses or not all(_upstream_allowed(a) for a in addresses):
        raise _PrivateDestination(host)
    error: OSError | None = None
    for address in addresses:
        try:
            return socket.create_connection((address, port), timeout=HEAD_TIMEOUT)
        except OSError as exc:
            error = exc
    raise error or OSError(f"cannot connect to {host}:{port}")


def _body_prefix(headers: list[tuple[str, str]], rest: bytes) -> bytes:
    """Bytes after the head that belong to this request's body; a pipelined request is dropped."""
    names = {n.lower(): v for n, v in headers}
    if "transfer-encoding" in names:
        return rest
    try:
        return rest[: int(names.get("content-length", "0"))]
    except ValueError:
        return b""


def _build_head(
    method: str, target: str, version: str, headers: list[tuple[str, str]]
) -> bytes:
    kept = [(n, v) for n, v in headers if n.lower() not in _HOP_HEADERS]
    kept.append(("Connection", "close"))
    lines = [f"{method} {target} {version}"] + [f"{n}: {v}" for n, v in kept]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def _response(status: str, body: str, extra: str = "") -> bytes:
    data = body.encode("utf-8")
    head = (
        f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(data)}\r\nConnection: close\r\n{extra}\r\n"
    )
    return head.encode("latin-1") + data


def _pump(src: socket.socket, dst: socket.socket) -> None:
    """Copy until EOF, then half-close the destination so the other direction can finish."""
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_RDWR)
        return
    with contextlib.suppress(OSError):
        dst.shutdown(socket.SHUT_WR)


def _relay(client: socket.socket, upstream: socket.socket) -> None:
    client.settimeout(IDLE_TIMEOUT)
    upstream.settimeout(IDLE_TIMEOUT)
    t = threading.Thread(target=_pump, args=(upstream, client), daemon=True)
    t.start()
    _pump(client, upstream)
    t.join()
    with contextlib.suppress(OSError):
        upstream.close()


class Proxy(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        addr: tuple[str, int],
        policy: Policy,
        certs: CertStore,
        log: Log,
        upstream_ca: str | None = None,
    ):
        super().__init__(addr, Handler)
        self.policy = policy
        self.certs = certs
        self.log = log
        self.upstream_ctx = ssl.create_default_context(cafile=upstream_ca)


class Handler(socketserver.BaseRequestHandler):
    @property
    def proxy(self) -> Proxy:
        return cast(Proxy, self.server)

    def handle(self) -> None:
        try:
            self._handle()
        except (OSError, ConnectionError, ssl.SSLError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError):
                self.request.close()

    def _handle(self) -> None:
        method, target, version, headers, rest = _parse_head(_read_head(self.request))
        if method == "CONNECT":
            self._connect(target, rest)
        elif target.startswith("/"):
            if target == "/healthz":
                self.request.sendall(_response("200 OK", "ok\n"))
            else:
                self.request.sendall(
                    _response("400 Bad Request", "absolute URL required\n")
                )
        else:
            self._forward(
                self.request, method, target, version, headers, rest, secure=False
            )

    def _deny(self, sock: socket.socket, method: str, url: str, rule: str) -> None:
        self.proxy.log.write(action="blocked", method=method, url=url, rule=rule)
        sock.sendall(
            _response(
                "403 Forbidden",
                BLOCK_BODY.format(url=url),
                "X-BenchFlow-Blocked: 1\r\n",
            )
        )

    def _connect(self, target: str, early: bytes) -> None:
        host, _, port_s = target.rpartition(":")
        host = host.strip("[]").rstrip(".").lower()
        port = int(port_s) if port_s.isdigit() else 443
        rule = self.proxy.policy.host_rule(host)
        if rule:
            self._deny(self.request, "CONNECT", f"{host}:{port}", rule)
            return
        if self.proxy.policy.inspect(host):
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            tls = self.proxy.certs.context_for(host).wrap_socket(
                self.request, server_side=True
            )
            method, path, ver, headers, rest = _parse_head(_read_head(tls))
            self._forward(
                tls, method, path, ver, headers, rest, secure=True, host=host, port=port
            )
            return
        try:
            upstream = _connect_upstream(host, port)
        except _PrivateDestination:
            self._deny(self.request, "CONNECT", f"{host}:{port}", "private-address")
            return
        except OSError as exc:
            self.request.sendall(
                _response(
                    "502 Bad Gateway", f"cannot connect to {host}:{port}: {exc}\n"
                )
            )
            return
        self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        if early:
            upstream.sendall(early)
        _relay(self.request, upstream)

    def _forward(
        self,
        sock: socket.socket,
        method: str,
        target: str,
        version: str,
        headers: list[tuple[str, str]],
        rest: bytes,
        *,
        secure: bool,
        host: str = "",
        port: int = 0,
    ) -> None:
        if secure:
            path = target
            if _ABSOLUTE_FORM.match(target):
                path = "/" + target.split("://", 1)[1].partition("/")[2]
            url = f"https://{host}{path}"
            authority = host if port == 443 else f"{host}:{port}"
        else:
            parts = urllib.parse.urlsplit(target)
            host, port = (parts.hostname or "").rstrip("."), parts.port or 80
            path = (parts.path or "/") + (("?" + parts.query) if parts.query else "")
            url = f"http://{host}{path}"
            authority = host if port == 80 else f"{host}:{port}"
            if not host:
                sock.sendall(_response("400 Bad Request", "absolute URL required\n"))
                return
        rule = self.proxy.policy.url_rule(host, path)
        if rule:
            self._deny(sock, method, url, rule)
            return
        headers = [(n, v) for n, v in headers if n.lower() != "host"]
        headers.insert(0, ("Host", authority))
        rest = _body_prefix(headers, rest)
        try:
            upstream = _connect_upstream(host, port)
            if secure:
                upstream = self.proxy.upstream_ctx.wrap_socket(
                    upstream, server_hostname=host
                )
        except _PrivateDestination:
            self._deny(sock, method, url, "private-address")
            return
        except (OSError, ssl.SSLError) as exc:
            sock.sendall(
                _response(
                    "502 Bad Gateway", f"cannot connect to {host}:{port}: {exc}\n"
                )
            )
            return
        upstream.sendall(_build_head(method, path, version, headers) + rest)
        _relay(sock, upstream)


def serve(
    port: int,
    policy: Policy,
    certs: CertStore,
    log: Log,
    *,
    upstream_ca: str | None = None,
) -> Proxy:
    """Bind the proxy on loopback and return it (callers run ``serve_forever``)."""
    return Proxy(("127.0.0.1", port), policy, certs, log, upstream_ca)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--policy", required=True, help="JSON with blocked_urls, blocked_hosts"
    )
    parser.add_argument(
        "--cert-dir", required=True, help="directory of <host>.pem leaf certificates"
    )
    parser.add_argument(
        "--log", help="JSONL file for blocked attempts (default stderr)"
    )
    parser.add_argument("--upstream-ca", help="CA bundle for upstream TLS (tests)")
    args = parser.parse_args(argv)
    server = serve(
        args.port,
        Policy.load(args.policy),
        CertStore(args.cert_dir),
        Log(args.log),
        upstream_ca=args.upstream_ca,
    )
    print(
        f"egress proxy listening on 127.0.0.1:{args.port}", file=sys.stderr, flush=True
    )
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever(poll_interval=0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
