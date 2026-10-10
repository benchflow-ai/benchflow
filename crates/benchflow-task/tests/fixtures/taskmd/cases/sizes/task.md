Fix the bug in `app.py`.

```toml task
name = "acme/sizes"
version = "1.0.0"
[sandbox]
memory = 4096
disk = "10gb"

[verifier.sandbox]
memory = "4G"

[verifier]
isolation = "separate"
```
