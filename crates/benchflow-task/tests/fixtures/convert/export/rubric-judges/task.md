Write a short memo about the outage to `/app/memo.md`.

```toml task
name = "acme/memo"

[verifier]
timeout = "2m"

[verifier.judges.agent]
model = "claude-opus-5-5"
budget = { tokens = "1M" }
```
