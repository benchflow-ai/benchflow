Write a one-sentence summary of why the sky looks blue to `/app/summary.md`. Name the physical effect.

```toml task
name = "e2e/judged"
title = "One criterion a test decides, one a model judges"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
workdir = "/app"

[verifier]
timeout = "1m"
```
