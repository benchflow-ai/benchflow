Fix the bug in `app.py`.

```toml task
name = "acme/variants"
version = "1.0.0"
[agent]
timeout = "10m"

[variants.fast]
description = "Less time."
sandbx = {}

[variants.fast.agent]
timeout = "1m"
timout = "2m"

[variants.slow.verifier]
timeout = "nope"
```
