---
schema_version: '1.0'
metadata:
  author_name: benchflow
  difficulty: easy
  category: sanity
  tags:
  - deterministic-integration
verifier:
  type: test-script
  timeout_sec: 60.0
agent:
  timeout_sec: 300.0
sandbox:
  build_timeout_sec: 600.0
  cpus: 1
  memory_mb: 2048
  storage_mb: 10240
  allow_internet: true
benchflow:
  environment:
    manifest: environment.toml
---

## prompt

Create a file called `hello.txt` in `/app` containing exactly `Hello, world!`.

[[fake-llm:hello-pass]]
