Fix the bug in `app.py`.

```toml task
name = "acme/judges-brief"
version = "1.0.0"
[verifier.judges.agent]
brief = "https://example.com/brief.md"

[verifier.judges.vlm]
brief = "sandbox/brief.md"

[verifier.judges.llm]
brief = "verifier/nope.md"

[verifier.judges.panel]
brief = ""
```
