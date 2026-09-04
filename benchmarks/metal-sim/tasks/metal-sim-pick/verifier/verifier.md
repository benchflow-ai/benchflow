---
document_version: "0.3"
verifier:
  name: metal-sim-pick
  default_strategy: deterministic
  strategies:
    deterministic:
      type: script
      command: ./test.sh
  rubric:
    combine: weighted_mean
    dimensions:
      task_success: {weight: 1.0, source: deterministic}
  outputs:
    reward_text: /logs/verifier/reward.txt
    reward_json: /logs/verifier/reward.json
    details_json: /logs/verifier/reward-details.json
---

## verifier intent

Runs `/app/policy.py` through the `metal_sim` embodiment on six held-out cube positions (eval seed 4242, so the seeds differ from the four the prompt suggests for local testing) with the `metal-sim-pick` task and reports the success rate as the reward. `reward-details.json` keeps the per-scene status, minimum distance and episode length; the Inspect Robots JSON log and per-episode traces are kept under `/logs/artifacts`.
