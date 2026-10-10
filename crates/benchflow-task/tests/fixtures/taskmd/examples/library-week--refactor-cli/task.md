`/repo/datekit/cli.py` has grown to one long function. Split it into one function per subcommand without changing what any command prints or returns. The existing tests, and the ones you added earlier this week, must keep passing.

```toml task
name = "examples/library-week-refactor-cli"
title = "Split the CLI into one function per subcommand"
version = "1.0.0"

[agent]
timeout = "20m"
network = "none"

[sandbox]
image = "python:3.12-slim"
workdir = "/repo"
```
