---
document_version: "0.3"
verifier:
  name: clawsbench-archive-amazon-shipping-verifier
  default_strategy: gmail_state
  strategies:
    gmail_state:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      amazon_email_archived:
        weight: 1.0
        source: gmail_state
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
---

## role:reviewer

Inspect the Claw Gmail state dump and award full credit only when the unique
Amazon shipping email still exists, is not trashed or spammed, and no longer
has the `INBOX` label.
