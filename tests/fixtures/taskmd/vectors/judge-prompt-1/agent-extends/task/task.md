Fix the flaky retry logic in `src/http/client.py`. Retries should use exponential backoff (0.5 s, 1 s, 2 s, each with up to 20% jitter). Keep the change inside `src/http/`, note it in `CHANGELOG.md`, and do not add new dependencies.

```toml task
name = "vectors/agent-extends"
version = "2.1.0"

[sandbox]
image = "python:3.12-slim"
workdir = "/app"

[verifier.judges]
agent = "claude-opus-5-5"
llm = "claude-sonnet-5"
```
