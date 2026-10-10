Fix the bug in `app.py`.

```toml task
name = "acme/training-keys"
version = "1.0.0"
[roles.critic]
policy = "yes"

[training]
fork = "no"
max_concurrent = 0
learnable_band = [0.9, 0.1]
admit = 3
difficulty = 5
reward = "binary"
```
