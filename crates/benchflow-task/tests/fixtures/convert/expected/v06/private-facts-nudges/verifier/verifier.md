---
document_version: "0.3"
verifier:
  name: private-facts-nudges-verifier
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      recovery_files:
        weight: 1.0
        source: deterministic
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
---

## role:reviewer

Validate that the agent recovered the hidden order id through the simulated-user
interaction and wrote the required recovery files.
