Write the current year to `/app/year.txt` and the time zone's abbreviation to `/app/zone.txt`.

```toml task
name = "examples/timezone"
title = "A sandbox time zone"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"

[sandbox]
timezone = "Asia/Tokyo"
```
