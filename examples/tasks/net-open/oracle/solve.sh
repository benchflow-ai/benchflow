#!/bin/bash
# This runs as root. Everything it touches is in the sandbox or reserved:
# 198.51.100.7 (TEST-NET-2) and 198.18.0.1 (the benchmarking range) are
# addresses no network routes, so a connection to one hangs unless something
# refuses it at once (the firewall) or no route exists (no network at all);
# 127.0.0.2 and 127.0.0.3 are loopback addresses with listeners of its own.
python3 - <<'PY'
import errno, socket, subprocess, threading
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
    except TimeoutError:
        return "no-answer"
    except OSError as e:
        return "unreachable" if e.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH) else type(e).__name__
for a in ("127.0.0.2", "127.0.0.3"):
    serve(a)
lines = []
# The sandbox's own address on its network (no packet is sent to learn it):
# it exists only when the sandbox has a network, so a server there is one
# that only an open policy lets this probe reach; with no network at all
# (Docker's `none`) there is no such address.
try:
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    u.connect(("198.51.100.7", 9))
    own = u.getsockname()[0]
except OSError:
    own = None
if own and not own.startswith("127."):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((own, 8081)); s.listen(8)
    threading.Thread(target=lambda: [s.accept() for _ in range(8)], daemon=True).start()
    own_result = probe(own, 8081)
else:
    own_result = "no-address"
lines += ["reserved=" + probe("198.51.100.7", 443), "own=" + own_result, "local=" + probe("127.0.0.3", 8080)]
open("/app/net.txt", "w").write("\n".join(lines) + "\n")
PY
