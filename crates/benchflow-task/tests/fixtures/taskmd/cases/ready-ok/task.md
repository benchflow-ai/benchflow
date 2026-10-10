Fix the bug in `app.py`.

```toml task
name = "acme/ready-ok"
version = "1.0.0"
[sandbox]
ready = { run = "curl -f localhost", interval = "5s", retries = 3 }
```
