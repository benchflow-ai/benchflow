---
document_version: "0.3"
verifier:
  name: physical-review
  default_strategy: host-review
  strategies:
    host-review:
      type: script
      command: ./test.sh
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
---
# Physical evidence verifier

Scoring is deferred to the host reviewer after the agent loses access. The robotics SDK intentionally uses skip_verify=True, preserves the native BenchFlow result, and writes assessment.json plus reward.txt in the host trial directory. Generic bench eval runs must not invent a score from agent-authored claims.
