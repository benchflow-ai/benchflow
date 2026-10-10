Fix the bug in `app.py`.

```toml task
name = "acme/family-splits-good"
version = "1.0.0"
[family]
generator = "family/generate.py"

[family.splits]
train = "seed % 10 < 8"

[family.splits.iid]
rule = "seed % 10 == 8"
role = "validation"

[family.splits.ood]
role = "test"
rule = "seed % 10 == 9"
```
