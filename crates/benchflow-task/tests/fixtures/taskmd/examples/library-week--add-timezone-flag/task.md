Add a `--tz` option to the `datekit` command-line tool that converts parsed dates to the given IANA time zone before printing, for example `datekit parse 2026-09-27T10:00:00Z --tz Asia/Tokyo`. Without `--tz`, output stays as it is today. Document the option in `README.md`.

```toml task
name = "examples/library-week-add-timezone-flag"
title = "Add a --tz option to the CLI"
version = "1.0.0"

[agent]
timeout = "20m"
network = "none"

[sandbox]
image = "python:3.12-slim"
workdir = "/repo"
```
