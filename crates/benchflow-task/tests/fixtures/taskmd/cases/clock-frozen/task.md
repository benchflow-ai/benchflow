Fix the bug in `app.py`.

```toml task
name = "acme/clock-frozen"
version = "1.0.0"
[agent]
timeout = "30m"
network = "none"

[sandbox]
timezone = "America/New_York"
clock = { start = "2026-03-14T23:40:00-04:00", advance = "none", enforce = "world" }
```
