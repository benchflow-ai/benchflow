Solve the parity task. This file demonstrates that `task.md` can carry the
current Harbor-style `task.toml` schema plus BenchFlow-native roles, scenes,
and simulated-user metadata in one authoring document.

```toml task
name = "benchflow/harbor-parity"
description = "Schema-only example for Harbor-compatible task.md frontmatter"
authors = ["BenchFlow <benchflow@example.com>"]
keywords = ["parity", "task-md"]

[about]
category = "schema"
note = "This example is an authoring fixture, not a runnable task package."

[agent]
timeout = "2m"
user = "agent"
network = ["api.example.com"]

[sandbox]
image = "ghcr.io/example/task:latest"
os = "linux"
cpus = 4
memory = "4 GB"
disk = "8 GB"
gpus = 1
gpu_types = ["T4", "A100"]
tpu = { type = "v6e", topology = "2x4" }
network = ["datasets.example.com"]
workdir = "/workspace"
env = { DATASET = "${DATASET:-sample}" }
skills = "/skills"
build_timeout = "10m"

[sandbox.ready]
run = "python -m app.healthcheck"
interval = "2s"
timeout = "10s"
retries = 5

[[sandbox.outputs]]
path = "/logs/artifacts"
save_as = "artifacts"

[stages.scaffold]
unlock = "at_start"
agent = { timeout = "30s" }
gate = 0.5

[stages.scaffold.verifier]
timeout = "15s"
env = { STEP = "scaffold" }

[[stages.scaffold.outputs]]
path = "/app/scaffold.txt"

[roles.solver]

[user]
model = "scripted"

[verifier]
timeout = "1m"
user = "root"
env = { JUDGE_API_KEY = "${JUDGE_API_KEY:-test}" }
isolation = "separate"
network = "open"
combine_stages = "final"
models = true

[verifier.sandbox]
image = "ghcr.io/example/grader:latest"
cpus = 2
memory = "1 GB"
network = "none"

[oracle.env]
SOLUTION_MODE = "oracle"

[provenance]
source = "harbor/parity"

[import.v06]
schema_version = "1.3"

[import.v06.verifier]
pytest_plugins = ["pytest_playwright"]
hardening = { cleanup_conftests = false }

[import.v06.roles.solver]
agent = "claude-agent-acp"

[[import.v06.interaction.scenes]]
name = "solve"
roles = ["solver"]
```
