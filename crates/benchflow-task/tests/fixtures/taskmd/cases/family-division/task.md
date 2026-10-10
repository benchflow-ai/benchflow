Fix the bug in `app.py`.

```toml task
name = "acme/family-division"
version = "1.0.0"
[family]
generator = "family/generate.py"

[family.splits]
train = "seed // (seed - 3) > 0"
test = "seed > 100"
```
