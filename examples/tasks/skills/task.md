A skill named `secret-word` is installed for you. Follow it: write the secret word it gives to `/app/word.txt`, with no trailing newline.

```toml task
name = "examples/skills"
title = "The answer is only in a task skill"
version = "1.0.0"

[agent]
timeout = "5m"

[sandbox]
skills = "sandbox/skills"

[verifier]
timeout = "1m"
```
