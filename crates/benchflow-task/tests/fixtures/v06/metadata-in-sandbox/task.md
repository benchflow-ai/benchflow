---
schema_version: "1.3"
task:
  name: acme/metadata-in-sandbox
  version: 1.0.0
agent:
  timeout_sec: 120
sandbox:
  allow_cloud_metadata: true
verifier:
  timeout_sec: 90
---
# Metadata opt-out in the wrong table

BenchFlow 0.8 refuses a key it does not know in `sandbox`, so the opt-out cannot live there.
