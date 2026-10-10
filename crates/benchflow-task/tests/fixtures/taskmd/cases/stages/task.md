Fix the bug in `app.py`.

```stage a
A.
```

```stage e
E.
```

```toml task
name = "acme/stages"
version = "1.0.0"
[stages.a]
unlock = "at_start"

[stages.b]
unlock = "at_start"

[stages.c]
submit = "x"

[stages.d]
unlock = "whenever"

[stages.e]
unlock = "after:a"
ready = { interval = "x" }

[stages.e.agent]
timeout = 5

[stages.e.verifier]
feedback = "rationales"
snapshot = [{ run = "x" }]
```
