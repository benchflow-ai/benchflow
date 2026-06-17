# 2. Verifier-tamper detection via a producer-side before/after file hash

- Status: Accepted (ENG-265)
- Date: 2026-06-17

## Context

Verifier tampering — an agent mutating the score-defining (verifier) files to fake a
reward — is the load-bearing reward-hacking defense. Today `agent_judge.py` detects
it with a regex over shell **input** in the trajectory. That has confirmed false
negatives (`python -c`, heredocs, `base64 -d | sh`, a renamed grader) and false
positives (any redirect near a path containing `tests/`), and it offloads the final
call to a small LLM judge over a truncated excerpt. A regex on the agent's commands
can never be a complete account of what the agent actually did to the filesystem.

## Decision

The **producer** (the sandbox/verifier path that owns the score-defining files)
records a hash of those files **before** and **after** the agent phase. The diff
yields a definitive `verifier_files_mutated: bool` in the rollout contract
(`result.json`), surfaced to **both** the mechanical realness gate and the LLM judge.
The existing trajectory regex remains only as a cheap advisory backstop. The set of
"score-defining files" is declared per verifier/task (the files whose contents
determine the reward).

## Consequences

- (+) Deterministic, language-agnostic, and independent of the trajectory's shape or
  the judge model — catches `python -c`, base64, renamed graders, etc.
- (+) Becomes the authoritative signal; the regex is demoted to advisory.
- (−) A **producer-side change in the real benchflow source** (sandbox/verifier),
  carrying its own risk and requiring the score-defining file set to be defined.
- (−) Adds a field to the rollout contract (`result.json` schema) and to `GateResult`.

## Alternatives considered

- **Compute the before/after evidence only in the checker/scenario layer**: no
  producer change, but it can only hash what the artifacts happened to capture, so it
  is weaker and still misses in-sandbox mutation that left no artifact. Rejected as
  the primary mechanism (may still be a fallback where producer hooks are absent).
- **Keep regex only**: rejected — the false-negative classes are exploitable.
