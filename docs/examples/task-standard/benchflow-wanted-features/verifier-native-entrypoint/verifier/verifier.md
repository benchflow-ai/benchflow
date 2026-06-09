---
document_version: "0.3"
verifier:
  name: verifier-native-entrypoint
  default_strategy: rewardkit
  strategies:
    rewardkit:
      type: reward-kit
      root: reward_kit
      criteria: rubrics/verifier.toml
  rubric:
    combine: weighted_mean
    dimensions:
      task_paths_entrypoint: {weight: 0.55, source: rewardkit}
      regression_tests: {weight: 0.30, source: rewardkit}
      no_test_sh_dogfood: {weight: 0.15, source: rewardkit}
  outputs:
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
    aggregate_policy:
      method: weighted_mean
      metrics:
        task_paths_entrypoint: 0.55
        regression_tests: 0.30
        no_test_sh_dogfood: 0.15
---

## verifier intent

This verifier package intentionally has no `test.sh`. The selected verifier
entrypoint is the Reward Kit runner declared above.
