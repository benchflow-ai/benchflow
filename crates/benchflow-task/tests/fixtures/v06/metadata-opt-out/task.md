---
schema_version: "1.3"
task:
  name: acme/metadata-opt-out
  version: 1.0.0
metadata:
  difficulty: easy
  x-allow-cloud-metadata: true
agent:
  timeout_sec: 120
verifier:
  timeout_sec: 90
---
# Metadata opt-out

The task opts out of bench's refusal of the cloud metadata endpoints, in the one table BenchFlow 0.8 loads with any key.
