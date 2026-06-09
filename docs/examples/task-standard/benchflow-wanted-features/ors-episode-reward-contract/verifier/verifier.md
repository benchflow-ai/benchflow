---
document_version: "0.4"
verifier:
  name: ors-episode-reward-contract
  default_strategy: ors
  strategies:
    ors:
      type: ors-episode
      inputs: [trajectory/ors-rewards.jsonl]
      format: jsonl
  rubric:
    combine: weighted_sum
    dimensions:
      runtime_artifact: {weight: 0.20, source: deterministic}
      declared_inputs: {weight: 0.25, source: ors}
      terminal_reward: {weight: 0.30, source: ors}
      details_preserved: {weight: 0.20, source: ors}
      fail_closed: {weight: 0.05, source: deterministic}
  outputs:
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
---

The runtime adapter writes ORS tool-output rewards into rollout trajectory
artifacts. The selected strategy consumes those declared ORS reward evidence
files and does not expose ORS-specific prompts or credentials to the solver.
