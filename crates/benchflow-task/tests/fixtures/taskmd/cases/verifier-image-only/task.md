Fix the bug in `app.py`.

```toml task
name = "acme/verifier-image-only"
version = "1.0.0"
[verifier]
isolation = "separate"

[verifier.sandbox]
image = "grader:1"
```
