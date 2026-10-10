Fix the bug in `app.py`.

```toml task
name = "acme/clock-bad-17"
version = "1.0.0"
[agent]
network = "none"

[sandbox]
clock = { start = "2026-03-14T09:00:00-04:00", enforce = "vm" }
```
