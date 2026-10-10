Fix the bug in `app.py`.

```toml task
name = "acme/outputs-typo"
version = "1.0.0"
[sandbox]
workdir = "/app"
outputs = [{ pth = "/app/out.txt" }, { path = "/app/out", save_as = "/etc/x", max_bytes = -1 }, "/app/log.txt", "relative/x", "/a/../b", 5, { path = "/x", exclude = "*.pyc" }]
```
