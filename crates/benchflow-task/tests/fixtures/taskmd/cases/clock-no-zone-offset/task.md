Fix the bug in `app.py`.

```toml task
name = "acme/clock-no-zone-offset"
version = "1.0.0"
[agent]
timeout = "10m"
network = "none"

[sandbox]
clock = { start = "2026-03-14T09:00:00+05:30", enforce = "vm" }
```
