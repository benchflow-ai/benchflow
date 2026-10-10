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
mount = "/tests"

# From verifier/judge.toml by tools/convert.py --convert-judges. RewardKit 0.1.7, the newest release ~=0.1.0 admits: its defaults for effort, per, timeout are written out.
[verifier.judges.llm]
model = "openai/gpt-5.6-terra"
effort = "medium"
per = "rubric"
samples = 1
timeout = "5m"

[oracle]
env = {}
mount = "/solution"

[import.harbor]
schema_version = "1.1"

[import.harbor.rewardkit]
installs = "harbor-rewardkit~=0.1.0"
scripts = ["verifier/test.sh"]
judges = []
converted = ["verifier/judge.toml"]
defaults = "0.1.7"
env = { OPENAI_API_KEY = "${OPENAI_API_KEY}" }

[import.harbor.rewardkit.kept]
"tests/test.sh" = "sha256:1b4dd117dea44035c94f247ce62ad670dd8692d12340ea0a61f92280a7d1ca52"
"tests/judge.toml" = "sha256:1d5c0d46e56a3173615196f6bf71b7180a03a94dac737545dce0135d4c876377"

[import.harbor.rewardkit.wrote]
"verifier/judge.md" = "sha256:9b11d5e2742a813f0db6d43c58d01441030153c69b4ae1865555a7340efd9770"
"verifier/rubric.json" = "sha256:163c291626e8d4fdd14bedff4f5ef863c136d26f4086deaac52969f8ad473f37"
"verifier/test.sh" = "sha256:72f98a601fd8ed60021bc3fcd4c9f965105fd7cb2e6cefe3d0c5dd334419a6ae"
"verifier.judges.llm" = "sha256:ddc2d0c38b93a2a1f19c335c58fc481aa5286de3df3d323f9850d0b38efb5a9d"

[import.harbor.originals."task.toml"]
original = "sha256:e815740c2b3ffc4e53f2047d82c472798aab398fdcb0d3dd3267409896f06d7a"
converted = "sha256:3840f52a16e0442b01fe69e0ce514281bcf0e1af535272974fab7a34a9016d85"

[import.harbor.originals."instruction.md"]
original = "sha256:035289f25429530be7a8c582392c43b2a394d615e30c6a3b10564bb9899a1868"
converted = "sha256:035289f25429530be7a8c582392c43b2a394d615e30c6a3b10564bb9899a1868"
```
