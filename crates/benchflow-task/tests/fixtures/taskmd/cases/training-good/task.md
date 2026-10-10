Fix the bug in `app.py`.

```toml task
name = "acme/training-good"
version = "1.0.0"
[roles.critic]
policy = true

[training]
fork = false
max_concurrent = 2
reward = "partial"

[agent]
timeout_basis = "wall"

[verifier]
feedback = "verdicts"
```
