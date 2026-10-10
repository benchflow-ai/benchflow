Fix the bug in `app.py`.

```toml task
name = "acme/durations-ok"
version = "1.0.0"
[agent]
timeout = "1h30m"

[sandbox]
build_timeout = "0.5s"

[verifier]
timeout = "90s"
```
