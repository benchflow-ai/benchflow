Do nothing. The image takes 5 minutes to build and the task allows 15 seconds (Daytona adds 2 minutes for its builder to start, so it stops the build after 2m15s).

```toml task
name = "examples/build-timeout"
title = "The image build runs past its timeout"
version = "1.0.0"

[agent]
timeout = "1m"

[sandbox]
build_timeout = "15s"

[verifier]
timeout = "1m"
```
