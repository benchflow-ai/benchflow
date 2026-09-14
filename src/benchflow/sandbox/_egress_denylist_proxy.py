"""Loopback egress proxy for network_mode='denylist'. Stdlib only; runs inside the sandbox as root.

HTTPS is intercepted so the CONNECT authority, TLS name and decrypted HTTP
authority stay bound to the same destination. Leaf certificates are generated
on demand with the per-rollout CA kept in the root-only runtime directory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import ipaddress
import json
import os
import re
import secrets
import socket
import socketserver
import ssl
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

HEAD_LIMIT = 64 * 1024
HEAD_TIMEOUT = 30
IDLE_TIMEOUT = 900
BLOCK_BODY = "Blocked by the task network policy: {url}\n"
CERT_MINT_TIMEOUT = 15


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

    def __init__(
        self,
        blocked_urls: list[str],
        blocked_hosts: list[str],
        model_gateway_port: int | None = None,
    ):
        # Controller-supplied runtime state, never a task-authored allowlist.
        if model_gateway_port is not None and (
            type(model_gateway_port) is not int
            or not 1024 <= model_gateway_port <= 65535
        ):
            raise ValueError("invalid model gateway port")
        self.model_gateway_port = model_gateway_port
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
            data.get("model_gateway_port"),
        )

    def host_rule(self, host: str, port: int = 0) -> str | None:
        if (host, port) == ("127.0.0.1", self.model_gateway_port):
            return None
        name = host.strip().rstrip(".").lower()
        if _looks_like_address(name):
            return "ip-literal"
        for blocked in self.hosts:
            if name == blocked or name.endswith("." + blocked):
                return f"host:{blocked}"
        return None

    def url_rule(self, host: str, path: str, port: int = 0) -> str | None:
        rule = self.host_rule(host, port)
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
    """Cached server TLS contexts, with root-only on-demand leaf signing."""

    def __init__(
        self,
        cert_dir: str,
        *,
        ca_cert: str | None = None,
        ca_key: str | None = None,
        openssl_bin: str = "openssl",
    ):
        if (ca_cert is None) != (ca_key is None):
            raise ValueError("ca_cert and ca_key must be provided together")
        self.cert_dir = Path(cert_dir)
        self.ca_cert = Path(ca_cert) if ca_cert is not None else None
        self.ca_key = Path(ca_key) if ca_key is not None else None
        self.openssl_bin = openssl_bin
        self._lock = threading.Lock()
        self._contexts: dict[str, ssl.SSLContext] = {}

    def context_for(self, host: str) -> ssl.SSLContext:
        key = self._certificate_host(host)
        with self._lock:
            ctx = self._contexts.get(key)
            if ctx is None:
                pem = self.cert_dir / f"{key}.pem"
                if not pem.is_file():
                    pem = self._mint(key)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(str(pem))
                ctx.set_alpn_protocols(["http/1.1"])
                ctx.sni_callback = _check_sni
                self._contexts[key] = ctx
            return ctx

    @staticmethod
    def _certificate_host(host: str) -> str:
        """Return a safe ASCII DNS name for certificate and cache use."""
        try:
            name = host_key(host).encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise ValueError("invalid certificate hostname") from exc
        if len(name) > 253 or _looks_like_address(name):
            raise ValueError("invalid certificate hostname")
        labels = name.split(".")
        if any(
            not label
            or len(label) > 63
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
            for label in labels
        ):
            raise ValueError("invalid certificate hostname")
        return name

    def _mint(self, host: str) -> Path:
        """Mint one leaf with argv-only OpenSSL calls and atomically cache it."""
        if self.ca_cert is None or self.ca_key is None:
            raise RuntimeError(f"no signer configured for TLS destination {host!r}")
        digest = hashlib.sha256(host.encode("ascii")).hexdigest()
        destination = self.cert_dir / f"dynamic-{digest}.pem"
        if destination.is_file():
            return destination
        alt_names = [host]
        if not host.startswith("www.") and len(f"www.{host}") <= 253:
            alt_names.append(f"www.{host}")
        alt_config = "\n".join(
            f"DNS.{index} = {name}" for index, name in enumerate(alt_names, 1)
        )
        config = (
            "[req]\n"
            "prompt = no\n"
            "distinguished_name = dn\n"
            "req_extensions = leaf\n"
            "[dn]\n"
            "CN = BenchFlow egress proxy\n"
            "[leaf]\n"
            "basicConstraints = critical,CA:FALSE\n"
            "keyUsage = critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage = serverAuth\n"
            "subjectAltName = @alt_names\n"
            "[alt_names]\n"
            f"{alt_config}\n"
        )
        self.cert_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=".mint-", dir=self.cert_dir) as tmp:
                work = Path(tmp)
                config_path = work / "leaf.cnf"
                key_path, request_path = work / "leaf.key", work / "leaf.csr"
                cert_path = work / "leaf.crt"
                config_path.write_text(config, encoding="ascii")
                commands = (
                    [
                        self.openssl_bin,
                        "req",
                        "-new",
                        "-newkey",
                        "ec",
                        "-pkeyopt",
                        "ec_paramgen_curve:P-256",
                        "-nodes",
                        "-config",
                        str(config_path),
                        "-keyout",
                        str(key_path),
                        "-out",
                        str(request_path),
                    ],
                    [
                        self.openssl_bin,
                        "x509",
                        "-req",
                        "-in",
                        str(request_path),
                        "-CA",
                        str(self.ca_cert),
                        "-CAkey",
                        str(self.ca_key),
                        "-set_serial",
                        f"0x{secrets.randbits(159) or 1:x}",
                        "-days",
                        "30",
                        "-sha256",
                        "-extfile",
                        str(config_path),
                        "-extensions",
                        "leaf",
                        "-out",
                        str(cert_path),
                    ],
                )
                for command in commands:
                    subprocess.run(
                        command,
                        check=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=CERT_MINT_TIMEOUT,
                    )
                temporary = self.cert_dir / f".{digest}.{secrets.token_hex(8)}.tmp"
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(descriptor, "wb") as output:
                    output.write(cert_path.read_bytes())
                    output.write(key_path.read_bytes())
                os.replace(temporary, destination)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"could not mint TLS certificate for {host!r}") from exc
        return destination


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
        if not sep or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
            raise ConnectionError("invalid request header")
        if any(ord(c) < 32 and c != "\t" for c in value):
            raise ConnectionError("invalid request header value")
        headers.append((name, value.strip()))
    return parts[0], parts[1], parts[2], headers, rest


_HOP_HEADERS = {"proxy-connection", "proxy-authorization", "connection", "keep-alive"}
_ABSOLUTE_FORM = re.compile(r"^https?://", re.IGNORECASE)


def _authority(value: str, default_port: int) -> tuple[str, int]:
    """Parse an HTTP authority without accepting userinfo, paths or whitespace."""
    if (
        not value
        or any(c.isspace() for c in value)
        or any(c in value for c in "/?#@\\")
    ):
        raise ValueError("invalid authority")
    parts = urllib.parse.urlsplit("//" + value)
    if not parts.hostname:
        raise ValueError("missing hostname")
    return parts.hostname.rstrip(".").lower(), parts.port or default_port


def _check_sni(
    sock: ssl.SSLObject | ssl.SSLSocket, server_name: str | None, _ctx: ssl.SSLContext
) -> int | None:
    """Bind TLS routing to CONNECT even for clients that disable certificate checks."""
    host = getattr(sock, "_benchflow_connect_host", "")
    if server_name is not None and server_name.rstrip(".").lower() != host:
        log = getattr(sock, "_benchflow_log", None)
        if log is not None:
            log.write(
                action="blocked",
                method="CONNECT",
                url=host,
                rule="tls-sni-mismatch",
                server_name=server_name,
            )
        return ssl.ALERT_DESCRIPTION_UNRECOGNIZED_NAME
    return None


def _request_framing(headers: list[tuple[str, str]]) -> tuple[int, bool]:
    """Accept one unambiguous HTTP request body, never a following request."""
    lengths = [v for n, v in headers if n.lower() == "content-length"]
    encodings = [v for n, v in headers if n.lower() == "transfer-encoding"]
    if len(lengths) > 1 or len(encodings) > 1 or (lengths and encodings):
        raise ValueError("ambiguous request framing")
    if encodings:
        if encodings[0].lower() != "chunked":
            raise ValueError("unsupported transfer encoding")
        return 0, True
    if lengths and not re.fullmatch(r"[0-9]+", lengths[0]):
        raise ValueError("invalid content length")
    return int(lengths[0]) if lengths else 0, False


class _BodyReader:
    def __init__(self, sock: socket.socket, initial: bytes):
        self.sock, self.buffer = sock, initial

    def take(self, size: int) -> bytes:
        if not self.buffer:
            self.buffer = self.sock.recv(min(size, 65536))
            if not self.buffer:
                raise ConnectionError("incomplete request body")
        data, self.buffer = self.buffer[:size], self.buffer[size:]
        return data

    def line(self) -> bytes:
        while b"\r\n" not in self.buffer:
            if len(self.buffer) > HEAD_LIMIT:
                raise ConnectionError("body line too large")
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("incomplete chunked body")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\r\n", 1)
        if len(line) > HEAD_LIMIT:
            raise ConnectionError("body line too large")
        return line


def _copy_request_body(
    client: socket.socket,
    upstream: socket.socket,
    rest: bytes,
    length: int,
    chunked: bool,
) -> None:
    """Stream only this body. Canonicalize chunks and discard trailers/pipeline bytes."""
    reader = _BodyReader(client, rest)

    def copy(size: int) -> None:
        while size:
            data = reader.take(min(size, 65536))
            upstream.sendall(data)
            size -= len(data)

    if not chunked:
        copy(length)
        return
    while True:
        size_text = reader.line().split(b";", 1)[0]
        if not re.fullmatch(rb"[0-9a-fA-F]+", size_text):
            raise ConnectionError("invalid chunk size")
        size = int(size_text, 16)
        if size == 0:
            trailer_size = 0
            while line := reader.line():
                trailer_size += len(line) + 2
                if trailer_size > HEAD_LIMIT:
                    raise ConnectionError("trailers too large")
            upstream.sendall(b"0\r\n\r\n")
            return
        upstream.sendall(f"{size:x}\r\n".encode("ascii"))
        copy(size)
        if reader.line() != b"":
            raise ConnectionError("invalid chunk terminator")
        upstream.sendall(b"\r\n")


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


def _connect_upstream(
    host: str, port: int, *, model_gateway_port: int | None = None
) -> socket.socket:
    """Connect to a vetted address of ``host``; the root proxy must not reach sandbox-internal services."""
    # Gemini's Undici ProxyAgent ignores NO_PROXY, even for the local model
    # gateway. Permit only the endpoint BenchFlow created; do not resolve a
    # hostname or expose any other private address/loopback port.
    if (host, port) == ("127.0.0.1", model_gateway_port):
        return socket.create_connection((host, port), timeout=HEAD_TIMEOUT)
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

    def _deny(
        self, sock: socket.socket, method: str, url: str, rule: str, **evidence: object
    ) -> None:
        self.proxy.log.write(
            action="blocked", method=method, url=url, rule=rule, **evidence
        )
        sock.sendall(
            _response(
                "403 Forbidden",
                BLOCK_BODY.format(url=url),
                "X-BenchFlow-Blocked: 1\r\n",
            )
        )

    def _connect(self, target: str, early: bytes) -> None:
        host, port = _authority(target, 443)
        rule = self.proxy.policy.host_rule(host, port)
        if rule:
            self._deny(self.request, "CONNECT", f"{host}:{port}", rule)
            return
        # Only the controller-created HTTP model gateway may use an opaque
        # tunnel. Every external CONNECT must expose its HTTP authority/path:
        # a CDN can route a permitted SNI to a protected inner Host as well.
        if (host, port) != ("127.0.0.1", self.proxy.policy.model_gateway_port):
            # Reject internal destinations before issuing a certificate or
            # acknowledging CONNECT. Known URL-policy hosts remain inspectable
            # even when their origin is offline, so blocked paths still log.
            if not self.proxy.policy.inspect(host):
                try:
                    addresses = _resolve(host, port)
                except OSError:
                    self.request.sendall(
                        _response("502 Bad Gateway", "cannot resolve destination\n")
                    )
                    return
                if not addresses or not all(_upstream_allowed(a) for a in addresses):
                    self._deny(
                        self.request, "CONNECT", f"{host}:{port}", "private-address"
                    )
                    return
            if early:
                self.request.sendall(
                    _response("400 Bad Request", "wait for CONNECT response\n")
                )
                return
            context = self.proxy.certs.context_for(host)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            with context.wrap_socket(
                self.request, server_side=True, do_handshake_on_connect=False
            ) as tls:
                tls._benchflow_connect_host = host  # ty: ignore[invalid-assignment]
                tls._benchflow_log = self.proxy.log  # ty: ignore[invalid-assignment]
                tls.settimeout(HEAD_TIMEOUT)
                tls.do_handshake()
                method, path, ver, headers, rest = _parse_head(_read_head(tls))
                self._forward(
                    tls,
                    method,
                    path,
                    ver,
                    headers,
                    rest,
                    secure=True,
                    host=host,
                    port=port,
                )
            return
        try:
            upstream = _connect_upstream(
                host, port, model_gateway_port=self.proxy.policy.model_gateway_port
            )
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
        if (
            version not in ("HTTP/1.0", "HTTP/1.1")
            or not re.fullmatch(r"[A-Z]+", method)
            or method == "CONNECT"
        ):
            sock.sendall(_response("400 Bad Request", "HTTP/1.x request required\n"))
            return
        if secure:
            path = target
            if _ABSOLUTE_FORM.match(target):
                parts = urllib.parse.urlsplit(target)
                if parts.scheme.lower() != "https" or _authority(parts.netloc, 443) != (
                    host,
                    port,
                ):
                    self._deny(sock, method, target, "authority-mismatch")
                    return
                path = (parts.path or "/") + (
                    ("?" + parts.query) if parts.query else ""
                )
            url = f"https://{host}{path}"
            authority = host if port == 443 else f"{host}:{port}"
        else:
            parts = urllib.parse.urlsplit(target)
            if parts.scheme.lower() != "http":
                sock.sendall(_response("400 Bad Request", "HTTP URL required\n"))
                return
            host, port = _authority(parts.netloc, 80)
            path = (parts.path or "/") + (("?" + parts.query) if parts.query else "")
            url = f"http://{host}{path}"
            authority = host if port == 80 else f"{host}:{port}"
        if not path.startswith("/") or "#" in path:
            sock.sendall(_response("400 Bad Request", "origin path required\n"))
            return
        authorities = [v for n, v in headers if n.lower() == "host"]
        if len(authorities) > 1 or (version == "HTTP/1.1" and not authorities):
            sock.sendall(_response("400 Bad Request", "one Host header required\n"))
            return
        if authorities and _authority(authorities[0], 443 if secure else 80) != (
            host,
            port,
        ):
            self._deny(
                sock,
                method,
                url,
                "authority-mismatch",
                received_authority=authorities[0],
                expected_authority=authority,
            )
            return
        try:
            length, chunked = _request_framing(headers)
        except ValueError as exc:
            sock.sendall(_response("400 Bad Request", str(exc) + "\n"))
            return
        expectations = [v.lower() for n, v in headers if n.lower() == "expect"]
        if expectations and expectations != ["100-continue"]:
            sock.sendall(
                _response("417 Expectation Failed", "unsupported expectation\n")
            )
            return
        rule = self.proxy.policy.url_rule(host, path, port)
        if rule:
            self._deny(sock, method, url, rule)
            return
        headers = [
            (n, v)
            for n, v in headers
            if n.lower() not in {"host", "expect", "upgrade", "trailer"}
        ]
        headers.insert(0, ("Host", authority))
        try:
            upstream = _connect_upstream(
                host, port, model_gateway_port=self.proxy.policy.model_gateway_port
            )
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
        try:
            sock.settimeout(IDLE_TIMEOUT)
            upstream.settimeout(IDLE_TIMEOUT)
            upstream.sendall(_build_head(method, path, version, headers))
            if expectations:
                sock.sendall(b"HTTP/1.1 100 Continue\r\n\r\n")
            _copy_request_body(
                sock, upstream, _body_prefix(headers, rest), length, chunked
            )
            # Never relay additional client bytes: a later request could name
            # an unchecked virtual host or path on the same origin connection.
            _pump(upstream, sock)
        finally:
            upstream.close()


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
        "--ca-cert", required=True, help="per-rollout signer certificate"
    )
    parser.add_argument(
        "--ca-key", required=True, help="root-only per-rollout signer key"
    )
    parser.add_argument("--openssl", required=True, help="OpenSSL executable")
    parser.add_argument(
        "--log", help="JSONL file for blocked attempts (default stderr)"
    )
    parser.add_argument("--upstream-ca", help="CA bundle for upstream TLS (tests)")
    args = parser.parse_args(argv)
    server = serve(
        args.port,
        Policy.load(args.policy),
        CertStore(
            args.cert_dir,
            ca_cert=args.ca_cert,
            ca_key=args.ca_key,
            openssl_bin=args.openssl,
        ),
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
