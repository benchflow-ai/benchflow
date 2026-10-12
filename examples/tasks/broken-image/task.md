Do nothing. The Dockerfile fails to build.

```toml task
name = "examples/broken-image"
title = "A Dockerfile that fails"
version = "1.0.0"

[agent]
timeout = "1m"

[verifier]
timeout = "1m"
```
