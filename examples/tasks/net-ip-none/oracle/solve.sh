#!/bin/bash
# This runs as root. It connects by address, so pinned
# names cannot stop it; only the firewall can. No outside host is involved:
# 198.51.100.7 is a reserved test address (TEST-NET-2) that no network
# routes, so only the firewall refuses it at once (without the firewall the
# connection would hang until its timeout); 127.0.0.3 and 127.0.0.2 are
# loopback addresses with listeners of the probe's own.
python3 - <<'PY'
import socket, threading
def serve(addr):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((addr, 8080)); s.listen(8)
    threading.Thread(target=lambda: [s.accept() for _ in range(8)], daemon=True).start()
def probe(addr, port):
    try:
        socket.create_connection((addr, port), 5).close()
        return "answered"
    except ConnectionRefusedError:
        return "refused"
    except OSError as e:
        return "no-answer" if isinstance(e, TimeoutError) else type(e).__name__
for a in ("127.0.0.3", "127.0.0.2"):
    serve(a)
lines = ["forbidden=" + probe("198.51.100.7", 443), "local=" + probe("127.0.0.3", 8080)]

open("/app/net.txt", "w").write("\n".join(lines) + "\n")
PY
