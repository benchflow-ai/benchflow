---
document_version: '0.3'
verifier:
  name: skillsbench-3d-scan-calc-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      geometric_mass_report:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

Validate that the submitted mass report identifies the main connected component, material ID, and mass within the accepted tolerance.
