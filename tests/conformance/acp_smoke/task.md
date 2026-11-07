---
schema_version: '1.0'
metadata:
  author_name: benchflow
  difficulty: trivial
  category: conformance
  tags: [conformance, acp, smoke]
agent:
  timeout_sec: 600
verifier:
  timeout_sec: 30
sandbox:
  docker_image: ubuntu:24.04
  cpus: 1
  memory_mb: 1024
---

## prompt

Create a file called `conformance.txt` in the current directory containing exactly:

```
ok
```

Then list the current directory's contents (via your shell tool) and stop.
