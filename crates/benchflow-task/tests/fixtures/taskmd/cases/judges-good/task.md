Fix the bug in `app.py`.

```toml task
name = "acme/judges-good"
version = "1.0.0"
[verifier.judges]
samples = 3

[verifier.judges.agent]
model = "m"
harness = "judge-loop@1"
timeout = "10m"
budget = { tokens = "2M", tool_calls = 100 }
per = "rubric"
resources = { cpus = 2, memory = "4 GB" }
services = false
```
