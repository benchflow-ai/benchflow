Do nothing: this task's verifier always writes the same reward file. reward.json with metrics but no reward key.

```toml task
name = "examples/reward-json-metrics"
title = "Reward file shape: reward-json-metrics"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"
```
