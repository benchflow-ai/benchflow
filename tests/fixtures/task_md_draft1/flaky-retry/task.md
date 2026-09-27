Fix the flaky retry logic in `src/http/client.py`. Requests that time out should be retried three times with exponential backoff (0.5 s, 1 s, 2 s, each with up to 20% jitter), then raise `RetryError` with the last exception attached as its cause.

Keep the public API of `HttpClient` unchanged, and do not add new dependencies. The failing test is `tests/test_retry.py::test_timeout_retries`.

```toml task
name = "examples/flaky-retry"
title = "Fix the flaky retry in an HTTP client"
version = "3.0.0"
keywords = ["python", "networking", "bugfix"]

[about]
difficulty = "medium"
category = "software-engineering"

[agent]
timeout = "15m"
network = "none"

[sandbox]
cpus = 2
memory = "4 GB"
workdir = "/workspace"

[verifier]
timeout = "5m"
isolation = "separate"
judges = { llm = "claude-sonnet-5", samples = 3, aggregate = "median" }

[runs]
trials = 5
pinned = ["agent", "sandbox", "verifier"]
errored = "zero"
disclose = ["agent", "agent_version", "models", "reasoning_effort"]
trajectory = "required"
```
