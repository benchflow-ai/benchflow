---
schema_version: "1.3"
metadata:
  author_name: benchflow tests
  environment: lab
  notes:
    environment: kept where it is
agent:
  timeout_sec: 300
verifier:
  timeout_sec: 120
sandbox:
  cpus: 2
  memory_mb: 4096
  network_mode: no-network
---

## prompt

A nested `environment` key under another mapping is not the old top-level key.
