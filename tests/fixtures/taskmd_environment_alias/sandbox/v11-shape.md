---
schema_version: '1.3'
metadata:
  author_name: Jiachen Li
  difficulty: medium
  category: industrial-physical-systems
verifier:
  type: test-script
  timeout_sec: 1800.0
  service: main
  hardening:
    cleanup_conftests: true
agent:
  timeout_sec: 1800.0
sandbox:
  network_mode: public
  build_timeout_sec: 600.0
  os: linux
  cpus: 1
  memory_mb: 2048
  storage_mb: 5120
  gpus: 0
---

## prompt

Do it.
