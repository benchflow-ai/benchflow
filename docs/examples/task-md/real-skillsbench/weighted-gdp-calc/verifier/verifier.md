---
document_version: '0.3'
verifier:
  name: skillsbench-weighted-gdp-calc-verifier
  default_strategy: pytest
  strategies:
    pytest:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      spreadsheet_formula_correctness:
        weight: 1.0
        source: pytest
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/ctrf.json
---

## role:reviewer

Validate that the workbook preserves the task sheet shape while computing lookup ranges, net export statistics, and weighted means within tolerance.
