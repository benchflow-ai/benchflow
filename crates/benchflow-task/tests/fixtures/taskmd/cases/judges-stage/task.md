Fix it.

```stage two
Two.
```

```toml task
name = "acme/judges-stage"
version = "1.0.0"
[stages.two]
unlock = "on_submit"

[stages.two.verifier.judges]
samples = 2
min_samples = 3

[stages.two.verifier.judges.llm]
brief = "verifier/judge.md"
```
