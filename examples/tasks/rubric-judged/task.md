Our web service went down at 09:14 UTC. The log excerpt is in `/app/incident.log`. Write a short incident summary to `/app/summary.md`: what failed, the root cause, and the fix. Do not blame any person.

```toml task
name = "examples/rubric-judged"
title = "A rubric with a test gate and model-judged criteria against a reference"
version = "1.0.0"

[agent]
timeout = "5m"

[sandbox]
workdir = "/app"

[verifier]
timeout = "1m"
```
