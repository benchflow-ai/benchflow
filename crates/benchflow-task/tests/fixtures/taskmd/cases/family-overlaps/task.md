Fix the bug in `app.py`.

```toml task
name = "acme/family-overlaps"
version = "1.0.0"
[family]
generator = "family/generate.py"

[family.splits]
train = "seed % 10 < 8"
test = "seed % 10 > 6"
validation = "seed % 5 == 0"
```
