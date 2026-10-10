Fix the bug in `app.py`.

```toml task
name = "acme/clock-env-tz"
version = "1.0.0"
[agent]
timeout = "30m"
network = "none"

[sandbox]
timezone = "America/New_York"
env = { TZ = "UTC" }
clock = { start = "2026-03-14T09:00:00-04:00", enforce = "preload" }
```
