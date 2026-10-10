Fix the bug in `app.py`.

```toml task
name = "acme/clock-bad-2"
version = "1.0.0"
[agent]
timeout = "30m"
network = "none"

[sandbox]
timezone = "America/New_York"
clock = { start = "2026-03-14T09:00:00-04:00" }
```
