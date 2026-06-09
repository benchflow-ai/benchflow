---
document_version: "0.3"
verifier:
  name: runtime-capability-gate-verifier
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      runtime_view: {weight: 0.35, source: deterministic}
      fail_closed: {weight: 0.35, source: deterministic}
      compatibility: {weight: 0.20, source: deterministic}
      maintainability: {weight: 0.10, source: reviewer}
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
---

## role:reviewer

Judge whether the implementation moves task selection and runtime support into
a single explicit view. Penalize any behavior that keeps silently ignoring
unsupported parsed fields.

