Probe, by address and with no name lookup, whether a connection to 198.51.100.7 port 443 (a reserved test address no network routes) is refused at once, and whether a listener of your own on 127.0.0.3 port 8080 answers, and whether a listener on 127.0.0.2 port 8080 is refused. Write `forbidden=<refused|answered|no-answer>`, `local=<...>` and `blocked-local=<...>` on their own lines of `/app/net.txt`. The agent's network blocks 198.51.100.7 and the loopback address 127.0.0.2, and allows everything else. No outside host is involved.

```toml task
name = "examples/net-ip-blocklist"
title = "Network policy by address: net-ip-blocklist"
version = "1.0.0"

[agent]
timeout = "2m"
network = { block = ["198.51.100.7/32", "127.0.0.2/32"] }

[sandbox]
network = "open"

[verifier]
timeout = "1m"
network = "none"
```
