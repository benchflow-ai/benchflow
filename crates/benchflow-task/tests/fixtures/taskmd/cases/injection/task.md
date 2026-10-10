Fix the bug in `app.py`.

```toml task
name = "acme/injection"
version = "1.0.0"
[integrity.controls]
no_op = "controls/no-op.sh"
injection = ["no_op", "known_bad", "nearmiss"]
```
