Fix the bug in `app.py`.

```toml task
name = "acme/x-extensions"
version = "1.0.0"
[x-acme]
anything = 1

[agent]
x-retries = 3
timeout = "1m"

[sandbox]
x-gpu = "a100"
```
