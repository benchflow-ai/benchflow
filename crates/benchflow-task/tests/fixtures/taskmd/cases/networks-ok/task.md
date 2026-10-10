Fix the bug in `app.py`.

```toml task
name = "acme/networks-ok"
version = "1.0.0"
[sandbox]
network = ["pypi.org", "*.github.com"]

[agent]
network = "none"

[verifier]
network = { block = ["x.com"] }
```
