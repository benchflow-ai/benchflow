Fix the bug in `app.py`.

```toml task
name = "acme/provenance-integrity"
version = "1.0.0"
[provenance]
trace = "x"
author = "y"

[integrity]
profile = "shared-hardened"
level = 3

[integrity.controls]
no_op = "controls/no-op.sh"
bad = "x"
```
