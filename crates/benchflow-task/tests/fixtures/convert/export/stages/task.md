Create `/app/a.txt` holding `a`.

```stage two
Now create `/app/b.txt` holding `b`.
```

```role critic
Point out what is missing.
```

```toml task
name = "acme/two-stages"

[stages.one]
unlock = "at_start"

[stages.one.agent]
timeout = "2m"

[stages.two]
unlock = "after:one"
gate = 1.0

[verifier]
combine_stages = "final"
mount = "/tests"
```
