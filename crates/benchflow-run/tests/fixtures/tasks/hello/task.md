Create a file `/app/hello.txt` that contains exactly `Hello, world!` with no trailing newline.

```toml task
name = "fixtures/hello"
title = "Hello, world"
version = "1.0.0"

[agent]
timeout = "2m"

[verifier]
timeout = "1m"
```
