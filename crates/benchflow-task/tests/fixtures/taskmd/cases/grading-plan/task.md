Fix the bug in `app.py`.

```toml task
name = "acme/grading-plan"
version = "1.0.0"
[verifier]
timeout = "5m"

[verifier.judges]
samples = 3

[verifier.judges.agent]
timeout = "10m"
budget = { tokens = "1M" }

[verifier.judges.vlm]
samples = 1
```
