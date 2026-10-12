Write the current year to `/app/year.txt` and the time zone's abbreviation to `/app/zone.txt`.

```toml task
name = "examples/world-sim"
title = "A simulated world (not supported yet: refused by name)"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"

[world]
kind = "simulated"
```
