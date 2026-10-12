Do nothing: this task's verifier always writes the same reward file. reward.json with a top-level number outside [0, 1] beside its reward.

```toml task
name = "examples/reward-json-bad-metric"
title = "Reward file shape: reward-json-bad-metric"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"
```
