Fix the bug in `app.py`.

```toml task
name = "acme/grading-script-models"
version = "1.0.0"
[verifier]
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}", MODE = "x", K = "${ANTHROPIC_BASE_URL:-http://x}" }
```
