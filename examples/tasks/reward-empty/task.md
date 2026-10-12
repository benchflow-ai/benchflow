Do nothing: this task's verifier always writes the same reward file. An empty reward file.

```toml task
name = "examples/reward-empty"
title = "Reward file shape: reward-empty"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"
```
