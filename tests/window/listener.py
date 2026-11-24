"""The window check's host listener: a TCP server on every host address, for the sandbox's start-up connect.

It logs ``<time_ns> accept <ip>:<port>`` for each connection from a Docker bridge (172.16/12, 192.168/16) and
closes everything else unlogged. Run in-process by tests/test_egress_window_hostside.py, or alone:

    python3 -I tests/window/listener.py LOG [PORT]
"""

from __future__ import annotations

import ipaddress
import socket
import sys
import threading
import time
from pathlib import Path

BRIDGES = (
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)


class Listener:
    def __init__(self, log: Path, port: int = 0) -> None:
        self.log = log
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", port))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self._lock = threading.Lock()
        self._stop = False
        self._write(f"listening {self.port}")
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _write(self, text: str) -> None:
        with self._lock, open(self.log, "a") as fh:
            fh.write(f"{time.time_ns()} {text}\n")

    def _serve(self) -> None:
        while not self._stop:
            try:
                conn, (ip, port) = self.sock.accept()
            except OSError:
                return
            if any(ipaddress.ip_address(ip) in net for net in BRIDGES):
                self._write(f"accept {ip}:{port}")
            conn.close()

    def lines(self) -> list[str]:
        with self._lock:
            return self.log.read_text().splitlines()

    def close(self) -> None:
        self._stop = True
        self.sock.close()


if __name__ == "__main__":
    server = Listener(
        Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 47123
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.close()
