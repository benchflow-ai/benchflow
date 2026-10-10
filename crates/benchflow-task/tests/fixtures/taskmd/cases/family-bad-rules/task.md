Fix the bug in `app.py`.

```toml task
name = "acme/family-bad-rules"
version = "1.0.0"
[family]
generator = "family/generate.py"
seed_param = "n"
database = "x.db"

[family.splits]
train = "seed < 5"
test = "n $ 3"
validation = "n + 1"
x-note = "ignored"
odd = 5

[family.splits.extra]
rule = "(n < 3"
role = "test"
```
