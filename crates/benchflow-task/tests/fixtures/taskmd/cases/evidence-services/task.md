Fix the bug in `app.py`.

```toml task
name = "acme/evidence-services"
version = "1.0.0"
[sandbox]
outputs = [{ path = "/data", service = "db" }]
```
