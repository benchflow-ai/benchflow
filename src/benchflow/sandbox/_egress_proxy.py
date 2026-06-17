"""Allowlist egress proxy — stdlib-only, default-deny.

Runs inside the ``bf-egress`` sidecar (a minimal ``python`` image). The agent
container is attached to an ``internal: true`` Docker network with no route to
the outside world and is pointed at this proxy via ``HTTP(S)_PROXY``; the proxy
is the *only* path off-box, and it forwards a request only when the target host
matches the allowlist. Everything else — non-allowlisted hosts, raw-IP CONNECTs,
and any tool that ignores the proxy env and tries a direct socket — is refused
or has no route, so egress is confined to ``ALLOWED_HOSTS``.

Config via env: ``ALLOWED_HOSTS`` (comma-separated), ``PORT`` (default 8080).
Matching is exact or proper-subdomain (``api.example.com`` matches the entry
``example.com``); never substring. No request body, header, or host is logged
beyond the host being allowed/denied.
"""

from __future__ import annotations

import contextlib
import os
import select
import socket
import sys
import threading

_ALLOWED: tuple[str, ...] = tuple(
    h.strip().lower().rstrip(".")
    for h in os.environ.get("ALLOWED_HOSTS", "").split(",")
    if h.strip()
)
_PORT = int(os.environ.get("PORT", "8080"))
_BUF = 65536
#: Dedicated always-allow target (the benchflow model proxy) — permitted even
#: when the task allowlist is empty (no-network + model-only egress).
_LANE = os.environ.get("BENCHFLOW_EGRESS_LANE_HOST", "").strip().lower().rstrip(".")


def _host_allowed(host: str) -> bool:
    host = host.strip().lower().rstrip(".")
    if not host:
        return False
    if _LANE and host == _LANE:  # the model lane, allowed regardless of _ALLOWED
        return True
    if not _ALLOWED:
        return False
    for a in _ALLOWED:
        if a.startswith("*."):
            # leading-label wildcard: any subdomain at any depth, never the apex
            if host.endswith("." + a[2:]):
                return True
        elif host == a or host.endswith("." + a):
            return True
    return False


def _recv_headers(sock: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < _BUF:
        chunk = sock.recv(_BUF)
        if not chunk:
            break
        data += chunk
    return data


def _pipe(a: socket.socket, b: socket.socket) -> None:
    socks = [a, b]
    try:
        while True:
            r, _, x = select.select(socks, [], socks, 60)
            if x or not r:
                break
            for s in r:
                data = s.recv(_BUF)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        return


def _deny(client: socket.socket, host: str) -> None:
    client.sendall(
        b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n"
        b"X-Benchflow-Egress: denied\r\nConnection: close\r\n\r\n"
    )
    sys.stderr.write(f"egress-proxy: DENY {host or '?'}\n")
    sys.stderr.flush()


def _handle(client: socket.socket) -> None:
    upstream: socket.socket | None = None
    try:
        client.settimeout(30)
        header = _recv_headers(client)
        if not header:
            return
        line = header.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = line.split(" ")
        if len(parts) < 2:
            return
        method, target = parts[0], parts[1]

        if method.upper() == "CONNECT":
            host = target.rsplit(":", 1)[0].strip("[]")
            port = int(target.rsplit(":", 1)[1]) if ":" in target else 443
            if not _host_allowed(host):
                _deny(client, host)
                return
            try:
                upstream = socket.create_connection((host, port), timeout=30)
            except OSError:
                client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
                return
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            _pipe(client, upstream)
            return

        # Plain HTTP: an absolute-URI proxy target (http://host/path). Only treat
        # it as absolute when it actually starts with a scheme — a '://' inside a
        # query value of an origin-form target must NOT be read as the authority.
        host = _absolute_uri_host(target) or ""
        if not host:
            for hl in header.split(b"\r\n"):
                if hl.lower().startswith(b"host:"):
                    host = hl.split(b":", 1)[1].decode("latin-1").strip()
                    break
        hostname = host.rsplit(":", 1)[0].strip("[]")
        port = int(host.rsplit(":", 1)[1]) if ":" in host and "]" not in host else 80
        if not _host_allowed(hostname):
            _deny(client, hostname)
            return
        try:
            upstream = socket.create_connection((hostname, port), timeout=30)
        except OSError:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
            return
        upstream.sendall(_to_origin_form(header))
        _pipe(client, upstream)
    except Exception:
        pass
    finally:
        # Always release the upstream socket — a raise in sendall/_pipe would
        # otherwise skip its close() and leak the fd on non-refcounted runtimes.
        if upstream is not None:
            with contextlib.suppress(OSError):
                upstream.close()
        with contextlib.suppress(OSError):
            client.close()


def _to_origin_form(header: bytes) -> bytes:
    """Rewrite a proxy-style absolute-URI request line to origin-form.

    ``POST http://host:port/path?q HTTP/1.1`` -> ``POST /path?q HTTP/1.1`` so the
    upstream (e.g. a FastAPI/uvicorn server such as the host litellm proxy on the
    model lane) routes it instead of returning 404/400 for the absolute form.
    Already-origin-form request lines are returned unchanged.
    """
    line, sep, rest = header.partition(b"\r\n")
    parts = line.split(b" ")
    if len(parts) != 3 or not parts[1].lower().startswith((b"http://", b"https://")):
        return header
    method, target, version = parts
    after = target.split(b"://", 1)[1]
    # The authority ends at the first of '/', '?' or '#'; everything from there
    # is the origin-form path. A query/fragment with no path is prefixed with
    # '/' so the query survives (http://h?x=1 -> /?x=1, not / ).
    cuts = [
        i for i in (after.find(b"/"), after.find(b"?"), after.find(b"#")) if i != -1
    ]
    if not cuts:
        path = b"/"
    else:
        path = after[min(cuts) :]
        if not path.startswith(b"/"):
            path = b"/" + path
    return method + b" " + path + b" " + version + sep + rest


def _absolute_uri_host(target: str) -> str | None:
    """Authority of an absolute-form proxy target, or None for origin-form.

    Only an actual ``http://`` / ``https://`` prefix counts as absolute-form;
    a ``://`` appearing inside the query of an origin-form target (e.g.
    ``/cb?u=http://evil``) must not be mistaken for the authority. The
    authority ends at the first of ``/``, ``?`` or ``#``.
    """
    if not target.lower().startswith(("http://", "https://")):
        return None
    after = target.split("://", 1)[1]
    cuts = [i for i in (after.find("/"), after.find("?"), after.find("#")) if i != -1]
    return after[: min(cuts)] if cuts else after


def main() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", _PORT))
    srv.listen(128)
    sys.stderr.write(f"egress-proxy: listening on :{_PORT}; allow={list(_ALLOWED)}\n")
    sys.stderr.flush()
    while True:
        try:
            client, _ = srv.accept()
        except OSError:
            continue
        threading.Thread(target=_handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
