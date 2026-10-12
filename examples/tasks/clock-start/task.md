Write the current year to `/app/year.txt` and the time zone's abbreviation to `/app/zone.txt`.

```toml task
name = "examples/clock-start"
title = "A sandbox clock that starts in 2030 (not supported yet: refused by name)"
version = "1.0.0"

[agent]
timeout = "1m"
network = "none"

[verifier]
timeout = "1m"

[sandbox]
clock = { start = "2030-01-01T00:00:00Z", advance = "real", enforce = "preload" }
```
