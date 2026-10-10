Fix the bug in `app.py`.

```stage two
Two.
```

```toml task
name = "acme/host-var"
version = "1.0.0"
[sandbox]
env = { KEY = "${OPENAI_API_KEY}", MODE = "fast", BOTH = "${A}${B:-x}" }

[[sandbox.services]]
name = "db"
image = "postgres"
env = { P = "${PGPASS}" }

[stages.two]
unlock = "on_submit"

[stages.two.agent]
network = "none"
```
