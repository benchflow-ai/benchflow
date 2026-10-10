The CLI in `/repo` loads its settings from `config.yaml`. Move it to `config.toml`: read TOML with the standard library, keep reading `config.yaml` for one release with a deprecation warning, and update the docs. Existing users must not lose any setting.

```notes
Derived from a real session in which an agent did this migration and silently dropped the `retries` setting; the user caught it three turns later. The trace was scrubbed, the repository was replaced with a synthetic one that reproduces the same shape, and the failure became the -3 criterion.
```

```toml task
name = "examples/from-trace-toml"
title = "Migrate a CLI's config from YAML to TOML"
version = "1.3.0"
keywords = ["python", "migration", "trace-derived"]

[agent]
timeout = "30m"

[sandbox]
image = "python:3.12-slim"

[provenance]
derived_from = "trace"
trace = "traces/2026-09-12-cli-config.jsonl"
trace_turns = [14, 31]
method = "trace2task"
scrubbed = ["paths", "names", "secrets"]
human_review = "2026-09-20"
failure_mode = "dropped setting during migration"
```
