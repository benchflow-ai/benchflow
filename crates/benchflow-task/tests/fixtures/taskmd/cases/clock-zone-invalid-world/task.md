Fix the bug in `app.py`.

```toml task
name = "acme/clock-zone-invalid-world"
version = "1.0.0"
[sandbox]
timezone = 5

[world]
kind = "virtual"

[world.desktop]
timezone = "Asia/Tokyo"
```
