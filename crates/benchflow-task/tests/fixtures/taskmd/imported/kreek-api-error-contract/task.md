# User Lookup Response Bug

Fix `handleUserLookup` in `src/users.js`.

Requirements:

- Distinguish missing input, missing users, and successful lookups with
  appropriate HTTP-style statuses.
- Keep successful responses and error responses predictable for callers.
- Error responses should have stable machine-readable and human-readable
  fields.

Do not add external dependencies.

Work in the provided repository. Do not add external dependencies.

```toml task
name = "consult/api-error-contract"
description = "Tighten an HTTP contract and failure response."

[about]
kind = "code"
features = ["api", "error-handling", "security", "documentation"]

[agent]
timeout = "15m"

[verifier]
timeout = "10m"
mount = "/tests"
models = true

[verifier.env]
CLAUDE_CODE_OAUTH_TOKEN = "${CLAUDE_CODE_OAUTH_TOKEN:-}"
REWARDKIT_FORCE_SUBSCRIPTION = "${REWARDKIT_FORCE_SUBSCRIPTION:-1}"
REWARDKIT_JUDGE = "${CONSULT_EVAL_JUDGE:-claude-code}"

[import.harbor]
schema_version = "1.3"

[import.harbor.rewardkit]
installs = "harbor-rewardkit==0.2.1"
scripts = ["verifier/test.sh"]
judges = ["verifier/judge/quality.toml"]

[import.harbor.originals."task.toml"]
original = "sha256:60f479897e60053727716386c214f097ac11579cd0fa4759b19ad3c5f3557e4b"
converted = "sha256:0f2098a2074ffed32960485d472692077d7f72a787b04e499645db7b1748fa96"

[import.harbor.originals."instruction.md"]
original = "sha256:5d336d486d5e8e5ffbb2ab540c70fd7cfe7f2c997c4c430533e5126d1cc34263"
converted = "sha256:5d336d486d5e8e5ffbb2ab540c70fd7cfe7f2c997c4c430533e5126d1cc34263"
```
