Write `done` to `/app/done.txt`. (The reference solution is too slow on purpose: it outlives the agent's 10-second budget.)

```toml task
name = "examples/agent-timeout"
title = "The agent runs past its timeout"
version = "1.0.0"

[agent]
timeout = "10s"

[verifier]
timeout = "1m"
```
