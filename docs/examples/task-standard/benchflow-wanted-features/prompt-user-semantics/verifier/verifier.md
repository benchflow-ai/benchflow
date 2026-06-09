---
document_version: "0.3"
verifier:
  name: prompt-user-semantics-verifier
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_sum
    dimensions:
      prompt_composition: {weight: 0.40, source: deterministic}
      user_runtime: {weight: 0.35, source: deterministic}
      privacy: {weight: 0.15, source: deterministic}
      ergonomics: {weight: 0.10, source: reviewer}
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
---

## role:reviewer

Check that prompt composition is predictable and that private simulated-user
facts cannot leak into solver prompts.

