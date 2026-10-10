Write the JSON object `{"answer": 42}` to `/app/answer.json`.

```toml task
name = "fixtures/separate-verifier"
title = "Graded in an image of its own"
version = "1.0.0"

[agent]
timeout = "2m"

[verifier]
timeout = "1m"
isolation = "separate"
```
