"""The egress proxy relays a long model response instead of cutting it.

Both relays (the CONNECT tunnel and the response relay of an intercepted
request) gave each socket a 900 s timeout and treated the client side's
silence as the end: a streaming model response is client silence by nature,
so any response longer than 15 minutes was cut, and Claude Code retried the
same request from scratch. Idleness now counts bytes in either direction.
These tests scale the 900 s down to 1 s and stream for several seconds.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request

import pytest

from benchflow.sandbox import _egress_denylist_proxy as proxy_mod
from tests.test_egress_denylist import stack as _tls_stack

stack = _tls_stack
IDLE = 1.0


@pytest.fixture(autouse=True)
def _one_second_idle(monkeypatch):
    monkeypatch.setattr(proxy_mod, "IDLE_TIMEOUT", IDLE)
    monkeypatch.setattr(proxy_mod, "RELAY_POLL", 0.2)


def _stream_through(relay_name: str, seconds: float) -> tuple[int, int, float]:
    """Origin streams for ``seconds`` while the agent only reads."""
    agent, proxy_client = socket.socketpair()
    proxy_upstream, origin = socket.socketpair()
    relay = getattr(proxy_mod, relay_name)
    threading.Thread(
        target=relay, args=(proxy_client, proxy_upstream), daemon=True
    ).start()
    sent = 0

    def stream() -> None:
        nonlocal sent
        deadline = time.monotonic() + seconds
        with origin:
            while time.monotonic() < deadline:
                origin.sendall(b"x" * 100)
                sent += 100
                time.sleep(0.2)

    threading.Thread(target=stream, daemon=True).start()
    received = 0
    started = time.monotonic()
    agent.settimeout(seconds + 5)
    with agent:
        while chunk := agent.recv(65536):
            received += len(chunk)
    return sent, received, time.monotonic() - started


@pytest.mark.parametrize("relay_name", ["_relay_response", "_relay"])
def test_a_response_longer_than_the_idle_timeout_arrives_whole(relay_name):
    sent, received, elapsed = _stream_through(relay_name, seconds=4 * IDLE)
    assert elapsed >= 4 * IDLE - 0.5
    assert received == sent > 0


@pytest.mark.parametrize("relay_name", ["_relay_response", "_relay"])
def test_a_connection_silent_both_ways_still_closes(relay_name):
    agent, proxy_client = socket.socketpair()
    proxy_upstream, origin = socket.socketpair()
    relay = getattr(proxy_mod, relay_name)
    done = threading.Event()

    def run() -> None:
        relay(proxy_client, proxy_upstream)
        done.set()

    started = time.monotonic()
    threading.Thread(target=run, daemon=True).start()
    with agent, origin:
        assert done.wait(10 * IDLE)
    assert IDLE - 0.1 <= time.monotonic() - started <= 5 * IDLE


class _Sock:
    """A socket whose sends time out ``stalls`` times before they go through."""

    def __init__(self, stalls: int):
        self.stalls = stalls
        self.data = b""

    def send(self, view) -> int:
        if self.stalls:
            self.stalls -= 1
            raise TimeoutError("The write operation timed out")
        self.data += bytes(view[:3])
        return min(3, len(view))


def test_a_slow_reader_is_waited_for_while_the_connection_is_live():
    idle = proxy_mod._Idle()
    sock = _Sock(stalls=2)
    proxy_mod._send(sock, b"streamed", idle)
    assert sock.data == b"streamed"


def test_a_reader_that_never_reads_counts_as_idle():
    idle = proxy_mod._Idle()
    idle.last -= IDLE
    with pytest.raises(TimeoutError):
        proxy_mod._send(_Sock(stalls=1), b"x", idle)


@pytest.mark.parametrize(
    "url", ["https://paper.test/slow/15/0.25", "http://plain.test/slow/15/0.25"]
)
def test_proxy_relays_a_slow_stream_past_the_idle_timeout(stack, url):
    """Through the real proxy: TLS interception and plain HTTP forwarding."""
    started = time.monotonic()
    with stack.opener.open(urllib.request.Request(url), timeout=30) as response:
        body = response.read().decode()
    assert time.monotonic() - started >= 3 * IDLE
    assert body.splitlines() == [f"chunk {i}" for i in range(15)]
