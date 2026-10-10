---
schema_version: "1.3"
agent:
  timeout_sec: 1200
verifier:
  timeout_sec: 600
environment:
  cpus: 2
  memory: 4G
  allow_internet: true
solution:
  env:
    RETRIES: "3"
benchflow:
  evidence:
    acceptance_live:
      cases:
        - name: oracle-rerun
          type: oracle
          reruns: 3
          expect:
            reward_min: 0.99
            flake_rate_max: 0.34
---

Find the line in `/app/server.log` where the regex `ERROR \d+` first matches and write its number to `/app/answer.txt`.
