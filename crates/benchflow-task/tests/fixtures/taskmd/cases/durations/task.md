Fix the bug in `app.py`.

```toml task
name = "acme/durations"
version = "1.0.0"
[agent]
timeout = 600

[sandbox]
build_timeout = "1 hour"

[verifier]
timeout = "600"
```
