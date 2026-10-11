Write the JSON object `{"answer": 42}` to `/app/answer.json`.

```toml task
name = "e2e/none-separate-verifier"
version = "1.0.0"

[agent]
timeout = "2m"

[verifier]
timeout = "1m"
isolation = "separate"
network = "none"
```
