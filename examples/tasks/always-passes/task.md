Write `42` to `/app/answer.txt`.

```toml task
name = "examples/always-passes"
title = "A broken verifier: it passes whatever the agent did"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"
```
