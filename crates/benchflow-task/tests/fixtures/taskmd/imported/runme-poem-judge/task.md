# Write a Funny Runme Poem

Write a short, funny poem about Runme, notebooks, terminals, or developer
workflow.

Save the poem at:

```text
poem.txt
```

The verifier will judge the poem's humor with an LLM.

```toml task
name = "runmedev/poem-judge"
description = "Poem-writing task graded by an LLM-as-a-judge verifier."
authors = []
keywords = ["runme", "harbor", "containerless", "llm", "judge"]

[about]
difficulty = "easy"
category = "behavior"
tags = ["runme", "harbor", "llm-as-a-judge", "verifier-env"]

[agent]
timeout = "8m"

[sandbox]
cpus = 1
memory = "1 GB"
disk = "1 GB"
gpus = 0
workdir = "/app/examples/harbor/datasets/runme-llm-judge/poem-judge/workdir"
env = {}
build_timeout = "1m"

[verifier]
timeout = "15m"
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }
mount = "/tests"
models = true

[oracle]
env = {}
mount = "/solution"

[import.harbor]
schema_version = "1.1"

[import.harbor.rewardkit]
installs = "harbor-rewardkit~=0.1.0"
scripts = ["verifier/test.sh"]
judges = ["verifier/judge.toml"]

[import.harbor.originals."task.toml"]
original = "sha256:e815740c2b3ffc4e53f2047d82c472798aab398fdcb0d3dd3267409896f06d7a"
converted = "sha256:1a5fdf81961a5c549dfa11f8a91fce91320694e46f6ede0725a7a331045dc4b7"

[import.harbor.originals."instruction.md"]
original = "sha256:035289f25429530be7a8c582392c43b2a394d615e30c6a3b10564bb9899a1868"
converted = "sha256:035289f25429530be7a8c582392c43b2a394d615e30c6a3b10564bb9899a1868"
```
