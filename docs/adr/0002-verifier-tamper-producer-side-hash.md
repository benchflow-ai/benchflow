# 2. Verifier-tamper detection — cheap fail-closed now, producer-side hash deferred

- Status: Accepted; partially implemented in #802 / remainder deferred
- Date: 2026-06-17 (revised for #802: 2026-06-18)

## Context

> **Ported from #799 and corrected to #802's reality.** The original ADR locked
> the **producer-side before/after file hash** + a `verifier_files_mutated`
> contract field as the *primary* mechanism. #802 split this into (a) the cheap
> fail-closed fix it ships now, and (b) the producer-side hash + contract field it
> defers. The Decision below records that split; the Consequences / Alternatives
> are kept.

Verifier tampering — an agent mutating the score-defining (verifier) files to fake
a reward — is the load-bearing reward-hacking defense. The detector in #802 is
`agent_judge._scan_verifier_tamper`, a regex over shell **input** in the
trajectory. That regex has the known false-negative classes the original ADR
called out (`python -c`, heredocs, `base64 -d | sh`, a renamed grader) and
false-positive classes (any redirect near a path containing `tests/`); a regex on
the agent's commands can never be a complete account of what the agent did to the
filesystem. This is the gate surfaced as **`V-TAMPER`** in
[`../integration-review-rubric.md`](../integration-review-rubric.md).

The original ADR's two-part remedy (the cheap fail-close on the existing signal,
and the deterministic producer-side hash) had very different costs: the first is a
few lines in already-trusted-main `agent_judge.py`; the second is a change to the
**real benchflow source** (`src/benchflow/sandbox/` + the verifier path) plus a
rollout-contract (`result.json`) schema bump. #802 ships the first now and defers
the second.

## Decision

**Adopted now in #802 (Task A1 — the cheap FAIL-CLOSED fix).**
`agent_judge.realness_issues` folds the mechanically-detected tamper signal
(`evidence.flagged_actions`, the `_scan_verifier_tamper` output) into the
**judge-independent realness gate**: any flagged write/delete/chmod of a
score-defining file makes the realness gate **hard-fail regardless of the LLM
judge** (and regardless of the judge being absent or lenient). The judge prompt is
additionally told the files were flagged (`flagged_verifier_actions`). So
`V-TAMPER` is now fail-closed on the signal #802 already has, instead of being
offloaded to the judge. (Implemented in
`tests/integration/agent_judge.py`; the docstring of `realness_issues` cites this
ADR.)

**Deferred follow-up (the producer-side authority).** The **producer** (the
sandbox/verifier path that owns the score-defining files) will record a hash of
those files **before** and **after** the agent phase, yielding a definitive
`verifier_files_mutated: bool` written into the rollout contract (`result.json`)
and surfaced to both the realness gate and the LLM judge. The deterministic
trajectory regex is then demoted to an advisory backstop. The set of
"score-defining files" is declared per verifier/task. This is the deferred part
because it is a producer-side change in the real benchflow source plus a contract
schema bump.

**Coordinate the schema bump.** The optional `verifier_files_mutated` contract
field is bundled with the **deferred `network_mode` result.json serialization**
(ADR-0003) as **one** rollout-contract schema bump — both are additive,
defaulting fields, and should land together so old artifacts still parse and the
`GateResult`/`--json` shape changes once, not twice.

## Consequences

- (+) `V-TAMPER` is fail-closed **today** on the cheap signal: a flagged tamper
  rejects the rollout deterministically, without depending on the judge.
- (+) The deferred producer-side hash is deterministic, language-agnostic, and
  independent of the trajectory shape or judge model — it will catch `python -c`,
  base64, renamed graders, etc., and become the authoritative signal (regex →
  advisory).
- (−) Until the follow-up lands, detection is still bounded by what the trajectory
  regex can observe; the false-negative classes remain open (mitigated, not
  closed, by fail-closing the signal that *is* observed).
- (−) The follow-up is a **producer-side change in the real benchflow source**
  (sandbox/verifier), carrying its own risk and requiring the score-defining file
  set to be defined, plus a (coordinated) `result.json` + `GateResult` schema
  bump.

## Alternatives considered

- **Ship only the producer-side hash, skip the cheap fix:** leaves `V-TAMPER`
  judge-dependent until the heavier source change lands. Rejected — the
  fail-close on the existing signal is nearly free and strictly safer.
- **Compute the before/after evidence only in the checker/scenario layer:** no
  producer change, but it can only hash what the artifacts happened to capture, so
  it is weaker and still misses in-sandbox mutation that left no artifact. Rejected
  as the primary mechanism (may still be a fallback where producer hooks are
  absent).
- **Keep regex only (not even fail-closed):** rejected — the false-negative classes
  are exploitable, and the judge-offload is avoidable.
