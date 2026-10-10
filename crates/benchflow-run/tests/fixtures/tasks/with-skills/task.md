Create a file `/app/hello.txt` that contains exactly `Hello, world!` with no trailing newline.

```toml task
name = "fixtures/with-skills"
title = "Hello, world, with skills bundled as SkillsBench bundles them"
version = "1.0.0"

[agent]
timeout = "2m"

[sandbox]
workdir = "/app"
skills = "sandbox/skills"

[verifier]
timeout = "1m"
```
