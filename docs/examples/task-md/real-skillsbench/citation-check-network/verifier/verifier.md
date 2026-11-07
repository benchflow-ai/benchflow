---
document_version: '0.3'
verifier:
  name: skillsbench-citation-check-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      fake_citation_detection:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

Validate that the answer JSON names exactly the hallucinated citation titles and preserves the expected output shape.
