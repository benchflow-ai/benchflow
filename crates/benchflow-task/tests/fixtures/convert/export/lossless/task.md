Write `done` to `/app/out.txt`.

```toml task
name = "acme/lossless"
version = "1.0.0"
description = "Export keeps everything"
authors = ["Ada Lovelace <ada@example.com>"]
keywords = ["export"]

[about]
difficulty = "easy"

[agent]
timeout = "5m"  # five minutes
user = "agent"

[sandbox]
memory = "2 GB"
network = ["pypi.org"]
workdir = "/app"
env = { MODE = "fast" }

[verifier]
timeout = "1m30s"
mount = "/tests"

[oracle]
mount = "/solution"
env = { SEED = "1" }
```
