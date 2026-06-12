---
schema_version: '1.0'
artifacts: []
metadata:
  author_name: benchflow
  difficulty: easy
  category: demo
  tags:
  - demo
  - hello-world
verifier:
  type: test-script
  timeout_sec: 30.0
  service: main
  pytest_plugins: []
  env: {}
  judge:
    model: claude-sonnet-4-6
    rubric_path: tests/rubric.toml
    input_dir: /app
    input_type: deliverables
    context: ''
  hardening:
    cleanup_conftests: true
agent:
  timeout_sec: 120.0
environment:
  network_mode: public
  build_timeout_sec: 600.0
  os: linux
  cpus: 1
  memory_mb: 1024
  storage_mb: 10240
  gpus: 0
  mcp_servers: []
  allow_internet: true
  env: {}
oracle:
  env: {}
---

## prompt

Create a file called `hello.txt` in the current directory containing exactly:

```
Hello from benchflow!
```

That's it — just create the file with that content.
