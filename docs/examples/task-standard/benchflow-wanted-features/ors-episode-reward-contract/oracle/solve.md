# Oracle: ORS Episode Reward Contract

The correct implementation adds an executable `ors-episode` verifier strategy
without making ORS a root task concept.

Acceptance evidence:

- `VerifierDocument` parses `type: ors-episode` with explicit inputs.
- `TaskPaths.is_valid()` accepts selected ORS strategies without `test.sh`.
- Sandbox capability validation treats selected ORS strategies as executable.
- `write_ors_tool_outputs_jsonl()` normalizes ORS tool-output rewards into the
  `trajectory/ors-rewards.jsonl` evidence artifact.
- `Verifier.verify()` reads declared ORS reward evidence and writes canonical
  `reward.json` plus `reward-details.json`.
- Malformed, invalid, or non-terminal ORS evidence fails closed.
