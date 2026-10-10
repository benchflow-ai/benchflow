Fix the bug in `app.py`.

```toml task
name = "acme/choices"
version = "1.0.0"
[verifier]
isolation = "seperate"
combine_stages = "max"
unreached_stages = 0
feedback = "all"
models = "yes"

[agent]
on_timeout = "pass"
timeout_basis = "cpu"

[sandbox]
boundary = "docker"

[runs]
errored = "skip"
trajectory = "maybe"
```
