Fix the bug in `app.py`.

```toml task
name = "acme/family-split-table-no-role"
version = "1.0.0"
[family]
generator = "family/generate.py"

[family.splits.holdout]
rule = "seed > 5"
```
