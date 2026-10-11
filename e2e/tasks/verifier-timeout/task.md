The verifier sleeps past its timeout.

```toml task
name = "e2e/verifier-timeout"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "10s"
```
