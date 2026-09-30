---
document_version: "0.3"
verifier:
  name: embodied-episode-judge
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
    details_json: /logs/verifier/reward-details.json
---

## verifier intent

The trusted episode server (the `simulator` compose service, `benchflow.embodied.server`) owns the simulation, enforces the step and wall-clock budgets and judges success from the physical state when the episode ends. This verifier runs in that same service (`verifier.service: simulator`). It closes the episode if the agent stopped without `robo done` (outcome `agent_exited`, judged after the usual settle period), waits for the server to write its record, copies the record (result.json, episode.json, steps.jsonl, trace.jsonl, frames.jsonl, video_index.json, recording.mp4, observation images) to `/logs/verifier/episode/`, and writes `reward.txt` = 1 if the server judged success, else 0, plus `reward.json` (reward, success_ever, budget_used). `reward-details.json` holds the episode result (outcome, steps used, return, the agent's final message). A missing episode result is an infrastructure error, not a zero.
