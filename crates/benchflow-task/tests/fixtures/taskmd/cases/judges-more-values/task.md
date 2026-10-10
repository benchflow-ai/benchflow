Fix the bug in `app.py`.

```toml task
name = "acme/judges-more-values"
version = "1.0.0"
[verifier.judges.agent]
resources = "big"
views = "x"
budget = 5

[verifier.judges.llm]
model = ["a"]
resources = {}
views = []
budget = { cost = 1 }

[verifier.judges.panel]
model = ["a", 3]
```
