Do nothing: this task's verifier always writes the same reward file. A partial reward in reward.txt.

```toml task
name = "examples/reward-fraction"
title = "Reward file shape: reward-fraction"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"
```
