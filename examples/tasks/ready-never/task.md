Do nothing. The sandbox's readiness check never passes, so no agent starts.

```toml task
name = "examples/ready-never"
title = "A readiness check that never passes"
version = "1.0.0"

[agent]
timeout = "1m"

[sandbox]
ready = { run = "test -f /tmp/never", interval = "1s", timeout = "15s" }

[verifier]
timeout = "1m"
```
