#!/bin/bash
# 1 when /app/hello.txt is right and port 53 on loopback answers itself.
python3 - <<'PY' > /logs/verifier/reward.txt
import socket

def tcp(family, addr):
    with socket.socket(family, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((addr, 53))
        srv.listen(1)
        with socket.create_connection((addr, 53), timeout=5):
            pass

def udp(family, addr):
    with socket.socket(family, socket.SOCK_DGRAM) as srv, socket.socket(family, socket.SOCK_DGRAM) as cli:
        srv.bind((addr, 53))
        srv.settimeout(5)
        cli.connect((addr, 53))
        cli.send(b"q")
        srv.recvfrom(16)
        # A refused UDP send shows up on the sender's next call.
        cli.send(b"q")
        srv.recvfrom(16)

ok = open("/app/hello.txt").read() == "Hello, world!" if __import__("os").path.exists("/app/hello.txt") else False
for family, addr in [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")]:
    try:
        socket.socket(family, socket.SOCK_DGRAM).bind((addr, 0))
    except OSError:
        continue  # no IPv6 loopback here
    for check in (tcp, udp):
        try:
            check(family, addr)
            print(f"{check.__name__} {addr}:53 answered", file=__import__("sys").stderr)
        except OSError as e:
            print(f"{check.__name__} {addr}:53 failed: {e}", file=__import__("sys").stderr)
            ok = False
print(1 if ok else 0)
PY
