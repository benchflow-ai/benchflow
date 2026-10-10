Fix the bug in `app.py`.

```toml task
name = "acme/networks"
version = "1.0.0"
[sandbox]
network = "no-network"

[agent]
network = 5

[verifier]
network = { block = "x" }
```
