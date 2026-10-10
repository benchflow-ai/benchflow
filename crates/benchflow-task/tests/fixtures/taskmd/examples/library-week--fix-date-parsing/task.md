`parse_date` in `/repo/datekit/parse.py` rejects ISO 8601 timestamps that end in `Z`, such as `2026-09-27T10:00:00Z`. Fix it so a trailing `Z` means UTC. Keep the public API unchanged; the failing test is `tests/test_parse.py::test_trailing_z`.

```toml task
name = "examples/library-week-fix-date-parsing"
title = "Accept a trailing Z in ISO timestamps"
version = "1.0.0"

[agent]
timeout = "20m"
network = "none"

[sandbox]
image = "python:3.12-slim"
workdir = "/repo"
```
