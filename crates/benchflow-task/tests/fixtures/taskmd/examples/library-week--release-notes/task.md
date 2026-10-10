Write `/repo/CHANGELOG.md` for datekit 0.4.0, covering every change made to the repository this week. Readers are users of the library, not its maintainers.

```toml task
name = "examples/library-week-release-notes"
title = "Release notes for the week's changes"
version = "1.1.0"

[agent]
timeout = "20m"
network = "none"

[sandbox]
image = "python:3.12-slim"
workdir = "/repo"
```
