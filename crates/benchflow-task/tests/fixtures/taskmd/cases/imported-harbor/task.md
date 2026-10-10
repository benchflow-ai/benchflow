Fix the bug in `app.py`.

```toml task
name = "acme/imported-harbor"
version = "1.0.0"
[import.harbor]
trajectory = "trajectory.json"
workdirs = { one = "steps/one/workdir", two = "steps/two/workdir" }
```
