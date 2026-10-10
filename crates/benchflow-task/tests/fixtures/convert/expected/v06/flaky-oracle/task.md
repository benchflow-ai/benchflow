Find the line in `/app/server.log` where the regex `ERROR \d+` first matches and write its number to `/app/answer.txt`.

```toml task
[agent]
timeout = "20m"

[sandbox]
cpus = 2
memory = "4 GB"
network = "open"

[verifier]
timeout = "10m"

[oracle.env]
RETRIES = "3"

[[x-benchflow.evidence.acceptance_live.cases]]
name = "oracle-rerun"
type = "oracle"
reruns = 3
expect = { reward_min = 0.99, flake_rate_max = 0.34 }

[import.v06]
schema_version = "1.3"
```
