Fix the bug in `app.py`.

```toml task
name = "acme/judges-values"
version = "1.0.0"
[verifier.judges]
samples = 0
min_samples = "x"
lazy = "no"
aggregate = "avg"
llm = 5
vlm = ""
panel = ["a", "b"]
unverified = true

[verifier.judges.agent]
model = ""
effort = 3
aggregate = "medain"
per = "task"
timeout = 600
harness = "judge-loop"
resources = { cpu = 2, memory = "lots" }
services = "yes"
views = ["/logs/agent/t.json", 5, { path = "rel", format = "atif" }, { pth = "/x", format = "trajectory-1" }, { path = "/y" }]
budget = { tokens = 0, tool_calls = -1 }
```
