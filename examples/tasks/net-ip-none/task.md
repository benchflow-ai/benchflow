Probe, by address and with no name lookup, whether a connection to 198.51.100.7 port 443 (a reserved test address no network routes) is refused at once, and whether a listener of your own on 127.0.0.3 port 8080 answers. Write `forbidden=<refused|answered|no-answer>`, `local=<...>` on their own lines of `/app/net.txt`. The agent's network is none; the verifier's is open, so `none` holds for the agent's phase only. No outside host is involved.

```toml task
name = "examples/net-ip-none"
title = "Network policy by address: net-ip-none"
version = "1.0.0"

[agent]
timeout = "2m"
network = "none"

[sandbox]
network = "open"

[verifier]
timeout = "1m"
network = "open"
```
