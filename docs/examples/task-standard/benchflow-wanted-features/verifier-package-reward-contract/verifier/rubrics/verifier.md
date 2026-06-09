# Verifier Package Reward Contract Rubric

- `verifier_document`: `VerifierDocument` parses strategy, rubric, output, and
  role sections from `verifier/verifier.md`, and selected `script` /
  `llm-judge` / `reward-kit` strategies execute through `Verifier.verify`.
  LLM judges honor verifier-local model, input, and context overrides.
- `reward_contract`: `reward.json` is authoritative, multi-metric maps are
  preserved, selected Reward Kit criteria govern allowed metric ids and
  aggregate weights, scalar mismatches fail closed, and `reward-details.json`
  is retained in result artifacts.
- `judge_isolation`: LLM judge execution stays in verifier scope.
  Verifier-scoped agent-judge strategies read only declared inputs, thread
  verifier credentials, and write structured reward details.
- `compatibility`: existing `verifier/test.sh` and split-layout `tests/test.sh`
  continue to run as compatibility strategies.
