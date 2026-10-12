Probe, by address and with no name lookup, whether a connection to 198.51.100.7 port 443 (a reserved test address no network routes) is refused at once, and whether a listener of your own on 127.0.0.3 port 8080 answers. Write `forbidden=<refused|answered|no-answer>`, `local=<...>` on their own lines of `/app/net.txt`. The agent's network allows only 198.18.0.1 (a reserved address that no network routes, so the self-test contacts no outside host). No outside host is involved.

```toml task
name = "examples/net-ip-allowlist"
title = "Network policy by address: net-ip-allowlist"
version = "1.0.0"

[agent]
timeout = "2m"
network = ["198.18.0.1/32"]

[sandbox]
network = "open"

[verifier]
timeout = "1m"
network = "none"
```
