Fix the bug in `app.py`.

```toml task
name = "acme/controls"
version = "1.0.0"
[integrity.controls]
no_op = "controls/no-op.sh"
known_bad = ["controls/missing.sh", "/abs/x.sh", "../out.sh", "", "controls/fixed-delay.patch"]
near_miss = ["controls/near.sh"]
cheats = 5
x-other = "x"
```
