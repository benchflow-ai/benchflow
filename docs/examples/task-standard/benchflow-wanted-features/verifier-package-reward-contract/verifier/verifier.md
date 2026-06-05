---
document_version: "0.3"
verifier:
  name: verifier-package-reward-contract
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
    rewardkit:
      type: reward-kit
      root: reward_kit/
      criteria: rubrics/verifier.toml
    judge:
      type: agent-judge
      role: verifier_judge
      inputs: [trajectory/acp_trajectory.jsonl, /logs/artifacts/diff.patch]
      isolation: verifier-only
  rubric:
    combine: weighted_sum
    dimensions:
      verifier_document: {weight: 0.30, source: deterministic}
      reward_contract: {weight: 0.35, source: deterministic}
      judge_isolation: {weight: 0.20, source: judge}
      compatibility: {weight: 0.15, source: deterministic}
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
    aggregate_policy:
      field: reward
      fallback: weighted_mean
---

## role:verifier_judge

Evaluate only declared deliverables and trajectories. Treat parser failures,
hidden fixture leakage, or missing reward details as verifier errors rather
than low-quality scored outcomes.

