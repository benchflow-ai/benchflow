"""Root's relay to the egress gateway, inside a sandbox the gateway holds. Stdlib only.

Uploaded by :mod:`benchflow.sandbox.egress_gateway` and run as root on the task
image's ``python3`` (3.8 or newer). It listens on ``127.0.0.1:<port>``, the
egress proxy's port, so the proxy and CA variables BenchFlow gives the agent,
the oracle and test.sh are the ones it gives them on Docker. The sandbox may
reach no address but the gateway's (Modal's ``outbound_cidr_allowlist``).

Each connection:

- ``CONNECT 127.0.0.1:<p>`` (or ``localhost``) with ``p`` from 1024 to 65535,
  not this relay's own port, is served here: the relay answers 200 and splices
  the client to that loopback port. Gemini's Undici client CONNECTs through
  ``HTTPS_PROXY`` even to the in-sandbox model gateway and ignores
  ``NO_PROXY``; at the gateway that address would be the gateway host's own.
  It grants nothing new: with no uid firewall in a held sandbox, every user
  reaches every loopback port directly.
- Anything else goes to the gateway byte for byte. The gateway first sends
  ``BFEG/1 <nonce>``; the relay answers ``<rollout> <hmac>``, the HMAC-SHA256
  of ``BFEG/1|<nonce>|<rollout>`` under the rollout's token, so the token never
  crosses the connection. Then the bytes read so far, then a splice.

The configuration (JSON: ``gateway`` as ``host:port``, ``rollout``, ``token``,
``port``) is root's, mode 600, in a folder only root can enter: the token is
never on a command line or in the environment.

Usage: ``python3 -I relay.py <config.json>``
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import re
import socket
import sys
import threading
from typing import Any

PROTOCOL = "BFEG/1"
CONNECT_TIMEOUT = 15.0
#: Longest challenge line read from the gateway.
CHALLENGE_LIMIT = 256
#: Seconds a client has to send its first line, and the most read for it.
HEAD_TIMEOUT = 30.0
HEAD_LIMIT = 8192
_LOOPBACK_CONNECT = re.compile(
    rb"CONNECT (?:127\.0\.0\.1|localhost):([0-9]{1,5}) HTTP/1\.[01]\r?\n"
)


def answer(token: str, nonce: str, rollout: str) -> str:
    """The challenge answer: HMAC-SHA256 of ``PROTOCOL|nonce|rollout`` under the token."""
    message = f"{PROTOCOL}|{nonce}|{rollout}".encode()
    return hmac.new(token.encode(), message, hashlib.sha256).hexdigest()


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            dst.shutdown(socket.SHUT_WR)


def _close(*socks: socket.socket | None) -> None:
    for sock in socks:
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()


def splice(client: socket.socket, upstream: socket.socket) -> None:
    """Copy bytes both ways until both directions end; then close both."""
    for sock in (client, upstream):
        sock.settimeout(None)
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    back = threading.Thread(target=_pump, args=(upstream, client))
    back.daemon = True
    back.start()
    _pump(client, upstream)
    back.join()
    _close(client, upstream)


def _challenge(upstream: socket.socket) -> str | None:
    """The gateway's nonce, or None for anything but one ``BFEG/1 <nonce>`` line."""
    data = b""
    while b"\n" not in data:
        chunk = upstream.recv(CHALLENGE_LIMIT - len(data))
        if not chunk:
            return None
        data += chunk
        if len(data) >= CHALLENGE_LIMIT:
            return None
    head = data.partition(b"\n")[0].decode("ascii", "replace").split(" ")
    if len(head) != 2 or head[0] != PROTOCOL or not head[1]:
        return None
    return head[1]


def _first_line(client: socket.socket) -> bytes:
    """What the client sent up to its first newline (all it sent, when it
    closes or stops first, or over :data:`HEAD_LIMIT`)."""
    data = b""
    client.settimeout(HEAD_TIMEOUT)
    try:
        while b"\n" not in data and len(data) < HEAD_LIMIT:
            chunk = client.recv(HEAD_LIMIT - len(data))
            if not chunk:
                break
            data += chunk
    except OSError:
        pass
    return data


def loopback_port(head: bytes, own: int) -> int | None:
    """The loopback port a ``CONNECT`` in *head* names, when the relay serves it."""
    found = _LOOPBACK_CONNECT.match(head)
    if not found:
        return None
    port = int(found.group(1))
    return port if 1024 <= port <= 65535 and port != own else None


def _serve_loopback(client: socket.socket, head: bytes, port: int) -> None:
    """Answer the CONNECT and splice the client to ``127.0.0.1:port``."""
    data = head
    try:
        while b"\r\n\r\n" not in data and b"\n\n" not in data:
            if len(data) > HEAD_LIMIT:
                raise OSError("request head too large")
            chunk = client.recv(HEAD_LIMIT)
            if not chunk:
                raise OSError("client closed in the request head")
            data += chunk
        rest = data.split(b"\r\n\r\n", 1)[1] if b"\r\n\r\n" in data else b""
        target = socket.create_connection(("127.0.0.1", port), timeout=CONNECT_TIMEOUT)
    except OSError:
        with contextlib.suppress(OSError):
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        _close(client)
        return
    try:
        client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
        if rest:
            target.sendall(rest)
    except OSError:
        _close(client, target)
        return
    splice(client, target)


def serve_one(client: socket.socket, config: dict[str, Any]) -> None:
    """One client connection: a loopback CONNECT here, anything else at the gateway."""
    head = _first_line(client)
    if not head:
        _close(client)
        return
    port = loopback_port(head, int(config["port"]))
    if port is not None:
        _serve_loopback(client, head, port)
        return
    host, _, gateway_port = str(config["gateway"]).rpartition(":")
    upstream = None
    try:
        upstream = socket.create_connection(
            (host, int(gateway_port)), timeout=CONNECT_TIMEOUT
        )
        nonce = _challenge(upstream)
        if nonce is None:
            _close(client, upstream)
            return
        rollout = str(config["rollout"])
        mac = answer(str(config["token"]), nonce, rollout)
        upstream.sendall(f"{rollout} {mac}\n".encode() + head)
    except OSError:
        _close(client, upstream)
        return
    splice(client, upstream)


def main(argv: list[str]) -> None:
    """Serve until killed."""
    with open(argv[1], encoding="utf-8") as handle:
        config = json.load(handle)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(config["port"])))
    listener.listen(512)
    sys.stderr.write(f"egress gateway relay on 127.0.0.1:{config['port']}\n")
    sys.stderr.flush()
    while True:
        try:
            client, _ = listener.accept()
        except OSError:
            continue
        worker = threading.Thread(target=serve_one, args=(client, config))
        worker.daemon = True
        worker.start()


if __name__ == "__main__":
    main(sys.argv)
