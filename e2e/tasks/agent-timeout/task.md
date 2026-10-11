The oracle sleeps past the agent's timeout, so the file it would write never appears.

```toml task
name = "e2e/agent-timeout"
version = "1.0.0"

[agent]
timeout = "10s"

[verifier]
timeout = "1m"
```
