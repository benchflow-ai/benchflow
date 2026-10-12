Write `42` to `/app/answer.txt`.

```toml task
name = "examples/separate-verifier"
title = "Graded in a sandbox of its own, from the saved outputs"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
outputs = ["/app/answer.txt"]

[verifier]
timeout = "1m"
isolation = "separate"
```
