Do nothing: this task's verifier always writes the same reward file. reward.json with a reward, and a count kept under details.

```toml task
name = "examples/reward-json"
title = "Reward file shape: reward-json"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"
```
