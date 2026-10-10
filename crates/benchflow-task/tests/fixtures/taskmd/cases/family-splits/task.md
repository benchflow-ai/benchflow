Fix the bug in `app.py`.

```toml task
name = "acme/family-splits"
version = "1.0.0"
[family]
generator = "family/generate.py"

[family.splits]
train = "seed % 10 < 8"
holdout = "seed % 10 == 9"

[family.splits.iid]
rule = "seed % 10 == 8"
role = "dev"

[family.splits.ood]
role = "test"
```
