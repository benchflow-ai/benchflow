Fix the bug in `app.py`.

```toml task
name = "acme/ready"
version = "1.0.0"
[sandbox]
ready = { interval = 5, timeout = "30s", retries = -1, start_period = "1 min" }
```
