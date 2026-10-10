# Task

Read `/app/context.md`. Produce `/app/solution.md` that explains what
a good `recipe explain <target>` answer should tell a normal user and
reviewer.

Cover scopes, named locations, managed/unmanaged values, optional
settings groups, lifecycle, redaction/sensitivity, native
export/import summaries, and the read-only metadata boundary.

Do not run live app commands, read live values, or implement code.

```toml task
name = "local/recipe-explain-clarity"
description = "Harbor agent test case: Recipe explain clarity"
authors = ["Harbor Agent Tests"]
keywords = ["harbor-agent-tests", "dotfiles-manager-v2", "release-readiness"]

[about]
issue = "121"
purpose = "Evaluate whether recipe/support explanations are understandable, metadata-only, and complete for scopes, named locations, managed values, optional groups, lifecycle, redaction, and native summaries."

[agent]
timeout = "20m"

[sandbox]
os = "linux"
cpus = 1
memory = "4 GB"
disk = "10 GB"
gpus = 0
network = "open"
env = {}
mcp = []
build_timeout = "30m"
outputs = []
compose = "sandbox/docker-compose.yaml"

[verifier]
timeout = "20m"
user = "root"
mount = "/tests"
models = true

[verifier.env]
CODEX_AUTH_SOURCE = "/run/codex-auth-source"
VERIFIER_CODEX_HOME = "/tmp/verifier-codex-home"

[oracle.env]

[import.harbor]
schema_version = "1.2"

[import.harbor.rewardkit]
installs = "harbor-rewardkit"
scripts = ["verifier/test.sh"]
judges = ["verifier/review.toml"]

[import.harbor.originals."task.toml"]
original = "sha256:63add7de62c8029f0be653ba46a6a5a57c472eddad775e87f5c2572de31de4c9"
converted = "sha256:03635131cc2ef4d760f6aa429cfb112995c44dca3038b799204234acb2f41f58"

[import.harbor.originals."instruction.md"]
original = "sha256:54f80ab21ce4476009b10ae6a355196ed25410c7133192b5756c3f39acc909e9"
converted = "sha256:54f80ab21ce4476009b10ae6a355196ed25410c7133192b5756c3f39acc909e9"
```
