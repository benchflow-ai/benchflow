Fix the bug in `app.py`.

```toml task
name = "acme/clock-bad-6"
version = "1.0.0"
[agent]
timeout = "30m"
network = "none"

[sandbox]
timezone = "America/New_York"
clock = "frozen:2026-03-14T09:00:00"
```
