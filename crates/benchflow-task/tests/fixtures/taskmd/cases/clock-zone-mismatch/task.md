Fix the bug in `app.py`.

```toml task
name = "acme/clock-zone-mismatch"
version = "1.0.0"
[sandbox]
timezone = "UTC"

[world]
kind = "virtual"

[world.browser]
timezone = "Europe/Paris"
```
