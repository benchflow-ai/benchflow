Fix the bug in `app.py`.

```toml task
name = "acme/evidence-outputs"
version = "1.0.0"
[sandbox]
workdir = "/app"
outputs = [{ path = "/app/out" }, { path = "/app/fit.json", max_bytes = "10 MB" }]

[world]
kind = "virtual"

[world.record]
streams = ["camera"]
```
