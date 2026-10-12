Probe, by address and with no outside host, what this sandbox's network lets through, and write one `key=result` line per probe to `/app/net.txt` (results: answered, refused, no-answer, unreachable). The agent's network allows only 198.18.0.0/15: a listed address passes the firewall (and then hangs, since nothing routes it), an unlisted one is refused at once, and the local listener answers. Expected: `listed=` no-answer or unreachable, `unlisted=` refused, `local=` answered.

```toml task
name = "examples/net-allowlist"
title = "Network policy: net-allowlist"
version = "1.0.0"

[agent]
timeout = "2m"
network = ["198.18.0.0/15"]

[sandbox]
network = "open"

[verifier]
timeout = "1m"
network = "none"
```
