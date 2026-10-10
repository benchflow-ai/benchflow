Fix the bug in `app.py`.

```toml task
name = "acme/matrix-preference"
version = "1.0.0"
[matrix]
axes = []
cells = 3

[preference]
protocol = "pairwise"
raters = 3

[preference.min_agreement]
metric = "kappa"
threshold = 0.6
```
