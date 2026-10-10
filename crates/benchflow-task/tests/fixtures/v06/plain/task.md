---
schema_version: "1.3"
task:
  name: acme/plain
  version: 1.2.0
agent:
  timeout_sec: 120
  network_mode: no-network
sandbox:
  cpus: 2
  memory_mb: 4096
  network_mode: allowlist
  allowed_hosts:
    - pypi.org
  env:
    MODE: test
verifier:
  timeout_sec: 90
  user: root
solution:
  env:
    HINT: yes-really
---
# Plain task

Write `/app/out.txt` with the answer.
