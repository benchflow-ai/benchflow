Create a file `/app/hello.txt` that contains exactly `Hello, world!` with no trailing newline.

```toml task
name = "examples/hello"
title = "Hello, world (task.md)"
version = "1.0.0"

[agent]
timeout = "2m"

[verifier]
timeout = "1m"
```
