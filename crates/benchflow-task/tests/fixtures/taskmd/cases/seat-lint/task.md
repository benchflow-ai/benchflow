Fix the bug in `app.py`.

```toml task
name = "acme/seat-lint"
version = "1.0.0"
[integrity.controls]
no_op = "controls/no-op.sh"
cheats = ["controls/read.py", "cheat/solve.sh"]
```
