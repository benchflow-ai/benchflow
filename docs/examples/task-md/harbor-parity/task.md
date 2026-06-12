---
schema_version: "1.3"
source: harbor/parity
multi_step_reward_strategy: final
artifacts:
  - source: /logs/artifacts
    destination: artifacts
task:
  name: benchflow/harbor-parity
  description: Schema-only example for Harbor-compatible task.md frontmatter
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [parity, task-md]
metadata:
  category: schema
  note: This example is an authoring fixture, not a runnable task package.
agent:
  timeout_sec: 120
  user: agent
  network_mode: allowlist
  allowed_hosts: [api.example.com]
verifier:
  timeout_sec: 60
  env:
    JUDGE_API_KEY: ${JUDGE_API_KEY:-test}
  user: root
  network_mode: public
  environment_mode: separate
  pytest_plugins: [pytest_playwright]
  hardening:
    cleanup_conftests: false
  environment:
    docker_image: ghcr.io/example/grader:latest
    cpus: 2
    memory_mb: 1024
    network_mode: no-network
environment:
  network_mode: allowlist
  allowed_hosts: [datasets.example.com]
  build_timeout_sec: 600
  docker_image: ghcr.io/example/task:latest
  os: linux
  cpus: 4
  memory_mb: 4096
  storage_mb: 8192
  gpus: 1
  gpu_types: [T4, A100]
  env:
    DATASET: ${DATASET:-sample}
  skills_dir: /skills
  workdir: /workspace
  tpu:
    type: v6e
    topology: 2x4
  healthcheck:
    command: python -m app.healthcheck
    interval_sec: 2
    timeout_sec: 10
    retries: 5
oracle:
  env:
    SOLUTION_MODE: oracle
steps:
  - name: scaffold
    min_reward: 0.5
    artifacts:
      - source: /app/scaffold.txt
    agent:
      timeout_sec: 30
    verifier:
      timeout_sec: 15
      env:
        STEP: scaffold
agents:
  roles:
    solver:
      agent: claude-agent-acp
scenes:
  - name: solve
    roles: [solver]
user:
  model: scripted
---

## prompt

Solve the parity task. This file demonstrates that `task.md` can carry the
current Harbor-style `task.toml` schema plus BenchFlow-native roles, scenes,
and simulated-user metadata in one authoring document.
