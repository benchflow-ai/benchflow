Do nothing. The verifier sleeps past its 5-second budget.

```toml task
name = "examples/verifier-timeout"
title = "The verifier runs past its timeout"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "5s"
```
