Fix the bug in `app.py`.

```toml task
name = "acme/services"
version = "1.0.0"
[[sandbox.services]]
name = "main"
image = "postgres:16"

[[sandbox.services]]
name = "db"
image = "postgres:16"
ports = [5432]
ready = { run = "" }

[[sandbox.services]]
name = "db"
build = "sandbox/db"

[[sandbox.services]]
image = "redis"
```
