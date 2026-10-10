Fix the bug in `app.py`.

```toml task
name = "acme/repr-unicode"
version = "1.0.0"
[user]
"mäx_turns" = 1
"max​_turns" = 2
"it's" = 3

[family.splits]
holdout = "seed é 3"

[sandbox]
timezone = "UTC"
env = { TZ = 5 }
```
