Fix the bug in `app.py`.

```toml task
name = "acme/moved-keys"
version = "1.0.0"
[training]
deterministic = true
verifier_latency_p50 = "0.4s"

[trajectory]
tokens = "optional"
format = "atif"
span_attributes = []

[world]
kind = "virtual"

[world.desktop]
timezone = "America/Los_Angeles"
clock = "frozen:2026-03-14T09:00:00"
```
