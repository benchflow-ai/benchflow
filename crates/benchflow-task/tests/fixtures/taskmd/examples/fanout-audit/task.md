Audit the 40 Python services under `/repo/services/` for places that read secrets from environment variables without a default and crash when the variable is missing. Use subagents to cover the services in parallel if you like.

Write your findings to `/work/findings.json` as a list of `{"service", "file", "line", "variable"}` objects, one per occurrence.

```toml task
name = "examples/fanout-audit"
title = "Audit 40 services, with or without subagents"
version = "1.2.0"
keywords = ["multi-agent", "subagents", "code-audit"]

[agent]
timeout = "40m"

[sandbox]
image = "ghcr.io/example/services-monorepo:2026.09"
cpus = 4

[trajectory]
require = ["subagents"]
```
