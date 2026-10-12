Probe, by address and with no outside host, what this sandbox's network lets through, and write one `key=result` line per probe to `/app/net.txt` (results: answered, refused, no-answer, unreachable). The agent's network blocks the reserved name blocked.bench.invalid (pinned to an address that routes nowhere) and the loopback address 127.0.0.2; everything else is open, so the listener on 127.0.0.3 answers. Expected: `name=` pinned, `blocked-local=` refused, `local=` answered.

```toml task
name = "examples/net-blocklist"
title = "Network policy: net-blocklist"
version = "1.0.0"

[agent]
timeout = "2m"
network = { block = ["blocked.bench.invalid", "127.0.0.2/32"] }

[sandbox]
network = "open"

[verifier]
timeout = "1m"
network = "none"
```
