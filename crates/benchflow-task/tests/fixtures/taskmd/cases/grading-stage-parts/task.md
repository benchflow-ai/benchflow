Fix the bug in `app.py`.

```stage two
Two.
```

```toml task
name = "acme/grading-stage-parts"
version = "1.0.0"
[verifier]
timeout = "2m"

[stages.two]
unlock = "on_submit"

[stages.two.verifier]
timeout = "3m"

[stages.two.verifier.judges.llm]
timeout = "1m"
samples = 2
```
