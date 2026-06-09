# ORS Episode Reward Contract Rubric

- `runtime_artifact`: ORS tool-output rewards normalize into
  `trajectory/ors-rewards.jsonl` without root task syntax.
- `declared_inputs`: `ors-episode` declares all evidence inputs in
  `verifier/verifier.md`.
- `terminal_reward`: reward evidence includes a valid terminal aggregate in
  `[0.0, 1.0]`.
- `details_preserved`: verifier output preserves the ORS response and reward
  events in `reward-details.json`.
- `fail_closed`: missing, malformed, invalid, or non-terminal ORS evidence is a
  verifier error, not an automatic zero.
