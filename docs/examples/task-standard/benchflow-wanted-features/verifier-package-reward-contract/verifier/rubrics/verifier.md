# Verifier Package Reward Contract Rubric

- `verifier_document`: `VerifierDocument` parses strategy, rubric, output, and
  role sections from `verifier/verifier.md`.
- `reward_contract`: `reward.json` is authoritative, multi-metric maps are
  preserved, scalar mismatches fail closed, and `reward-details.json` is
  retained in result artifacts.
- `judge_isolation`: LLM and ACP-backed judge roles execute only in verifier
  scope with declared inputs and output schema validation.
- `compatibility`: existing `verifier/test.sh` and split-layout `tests/test.sh`
  continue to run as compatibility strategies.

