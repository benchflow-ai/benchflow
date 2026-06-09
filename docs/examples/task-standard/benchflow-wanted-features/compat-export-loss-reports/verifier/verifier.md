---
document_version: "0.3"
verifier:
  name: compat-export-loss-reports-verifier
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      split_export: {weight: 0.35, source: deterministic}
      loss_report: {weight: 0.35, source: deterministic}
      foreign_preservation: {weight: 0.20, source: deterministic}
      docs: {weight: 0.10, source: reviewer}
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
---

## role:reviewer

Check that export reports are honest and that foreign extensions do not become
native root keys just because one fork used them.

