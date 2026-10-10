Fix the bug in `app.py`.

```toml task
name = "acme/judges-typos"
version = "1.0.0"
[verifier.judges.agent]
model = "m"
budgett = { tokens = "2M" }
timout = "10m"
reasoning = false

[verifier.judges.llm]
harness = "claude-code@2"
budget = { tool_calls = 5 }
min_samples = 3
samples = 2
```
