Fix the bug in `app.py`.

```toml task
name = "acme/clock-desktop-zone"
version = "1.0.0"
[agent]
timeout = "20m"
network = "none"

[sandbox]
clock = { start = "2026-03-14T09:00:00-07:00", advance = "none", enforce = "world" }

[world]
kind = "virtual"

[world.desktop]
timezone = "America/Los_Angeles"
```
