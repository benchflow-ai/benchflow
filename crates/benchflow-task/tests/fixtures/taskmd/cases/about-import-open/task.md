Fix the bug in `app.py`.

```toml task
name = "acme/about-import-open"
version = "1.0.0"
[about]
anything = { goes = 1 }

[import.harbor]
whatever = true

[integrity.resources]
"data/x" = "agent"

[family]
generator = "family/gen.py"

[family.params]
n = [1, 2]
```
