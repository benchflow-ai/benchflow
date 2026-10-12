Probe, by address and with no outside host, what this sandbox's network lets through, and write one `key=result` line per probe to `/app/net.txt`. Each result is one of `answered` (the connection opened), `refused` (refused at once), `no-answer` (no reply within 5 seconds), `unreachable` (no route), or `no-address`. The probes: `reserved=`, a TCP connection to 198.51.100.7 port 443 (a reserved test address that no network routes); `own=`, the sandbox's own non-loopback address, found with a UDP connect to 198.51.100.7 (which sends no packet) and read back as the socket's local address, with a server of your own listening on it on port 8081 and a TCP connection to that server, or `own=no-address` when the sandbox has no such address; and `local=`, a TCP connection to a server of your own listening on 127.0.0.3 port 8080. The network is open: the sandbox has an address of its own on its network, and the server there answers (with no network at all there is no such address), and nothing in the sandbox refuses a connection, so the reserved address hangs, or has no route on a host that has no outside route, but is never refused at once; the local server answers. Expected: `reserved=no-answer` or `reserved=unreachable`, `own=answered`, and `local=answered`.

```toml task
name = "examples/net-open"
title = "Network policy: net-open"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
network = "open"

[verifier]
timeout = "1m"
network = "none"
```
