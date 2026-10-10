Fix the bug in `app.py`.

```toml task
name = "acme/clock-bad-16"
version = "1.0.0"
[agent]
timeout = "30m"
network = "none"

[sandbox]
timezone = "America/New_York"
clock = { advance = "environment" }
```
