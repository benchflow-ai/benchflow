The issue tracker export for `datekit` is in `/repo/issues/`, one JSON file per open issue. Label every issue as `bug`, `feature`, or `question`, and pick the three most urgent bugs.

Write `/repo/triage.json` as `{"labels": {"<issue id>": "<label>"}, "urgent": ["<id>", "<id>", "<id>"]}`.

```toml task
name = "examples/library-week-triage-issues"
title = "Triage the open issues"
version = "1.0.0"

[agent]
timeout = "20m"
network = "none"

[sandbox]
image = "python:3.12-slim"
workdir = "/repo"
```
