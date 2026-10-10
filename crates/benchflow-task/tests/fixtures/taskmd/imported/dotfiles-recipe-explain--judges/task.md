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

[verifier.env]
CODEX_AUTH_SOURCE = "/run/codex-auth-source"
VERIFIER_CODEX_HOME = "/tmp/verifier-codex-home"

# From verifier/review.toml by tools/convert.py --convert-judges. RewardKit 0.2.1, the newest release (the task does not pin RewardKit): its defaults for per are written out.
[verifier.judges.agent]
model = "gpt-5.5"
harness = "codex@0.130.0"
per = "rubric"
samples = 1
timeout = "10m"

[oracle.env]

[import.harbor]
schema_version = "1.2"

[import.harbor.rewardkit]
installs = "harbor-rewardkit"
scripts = ["verifier/test.sh"]
judges = []
converted = ["verifier/review.toml"]
defaults = "0.2.1"

[import.harbor.rewardkit.kept]
"tests/test.sh" = "sha256:af031b515783dcfd3439526cb22aa3cdfa56f9b9d3379b4c8b2c8cc523416e50"
"tests/review.toml" = "sha256:4337d364ec68f7561ff74f2c13de69803716869c2b5b40ca05310ae0b8c31e36"

[import.harbor.rewardkit.wrote]
"verifier/judge.md" = "sha256:a4dbe92f221cc05bed634e4cc5e71f5249703df26d701f19b213b283b5717f46"
"verifier/rubric.json" = "sha256:b43a148f864c1cfb202caeb85742024225530e094f6cb4de35240bc631b526d2"
"verifier/test.sh" = "sha256:3be330cd5b6dacf709f4a06cedc8493b10574f69ec47b76a0ddb5ee9878e046a"
"verifier.judges.agent" = "sha256:05dfc747e58e948bb166b2e12be1bb8345cd798655e69fdcd2a794336c84bf9e"

[import.harbor.originals."task.toml"]
original = "sha256:63add7de62c8029f0be653ba46a6a5a57c472eddad775e87f5c2572de31de4c9"
converted = "sha256:c5a25b3d237e2dd26de829c52b76f8f88fb870e3bff4e9babc0171c6c0846434"

[import.harbor.originals."instruction.md"]
original = "sha256:54f80ab21ce4476009b10ae6a355196ed25410c7133192b5756c3f39acc909e9"
converted = "sha256:54f80ab21ce4476009b10ae6a355196ed25410c7133192b5756c3f39acc909e9"
```
