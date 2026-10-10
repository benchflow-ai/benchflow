Fix the bug in `app.py`.

```toml task
name = "acme/clock-variant-clock"
version = "1.0.0"
[agent]
timeout = "10m"
network = "none"

[variants.frozen.sandbox]
clock = { start = "2026-03-14T09:00:00Z" }
```
