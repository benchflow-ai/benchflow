Write `42` to `/app/answer.txt`.

```toml task
name = "examples/check-errors"
title = "Two mistakes bench check finds and --fix repairs"
version = "1.0.0"

[agent]
timout = "2m"

[verifier]
timeout_sec = 60
```
