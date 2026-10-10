Fix the bug in `app.py`.

```toml task
name = "acme/harbor-network"
version = "1.0.0"
[sandbox]
allow_internet = false

[agent]
allowed_hosts = ["pypi.org"]
```
