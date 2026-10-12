Probe, by address and with no outside host, what this sandbox's network lets through, and write one `key=result` line per probe to `/app/net.txt`. Each result is one of `answered` (the connection opened), `refused` (refused at once), `no-answer` (no reply within 5 seconds), `unreachable` (no route), or `no-address`. The probes: `reserved=`, a TCP connection to 198.51.100.7 port 443 (a reserved test address that no network routes); `own=`, the sandbox's own non-loopback address, found with a UDP connect to 198.51.100.7 (which sends no packet) and read back as the socket's local address, with a server of your own listening on it on port 8081 and a TCP connection to that server, or `own=no-address` when the sandbox has no such address; and `local=`, a TCP connection to a server of your own listening on 127.0.0.3 port 8080. There is no network: the reserved address is refused at once (a firewall), or has no route at all, in which case the sandbox has no address of its own either; loopback, and so the local server, still works. Expected: `reserved=refused`, or `reserved=unreachable` together with `own=no-address`; and `local=answered`.

```toml task
name = "examples/net-none"
title = "Network policy: net-none"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
network = "none"

[verifier]
timeout = "1m"
```
