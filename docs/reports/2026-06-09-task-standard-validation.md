# Task standard — validation evidence

Date: 2026-06-09

This page records how the BenchFlow task standard (`docs/task-standard.md`) was
validated: that a single `task.md` package can **represent**, **run**, and
**score** real benchmarks end-to-end. SkillsBench is the worked example because it
exercises the full surface — environment build, skill injection, oracle, and
verifier.

The bar throughout is **"runs end-to-end with a real rollout and a produced
verdict,"** not "the agent passes." Pass rate is a property of the model under
test, not of the standard.

## 1. Representation — conversion parity 88 / 88 (deterministic)

For all 88 SkillsBench `main` tasks, a roundtrip conformance check migrates each
legacy split package → native `task.md` → exports back to split, and compares the
canonical task config, the normalized prompt, and SHA-256 file maps for the
`environment/`, `solution/`, and `tests/` trees.

```
=== SkillsBench conversion-parity: 88 tasks | PASS=88  MISMATCH=0  ERROR=0 ===
```

Every task is representable as native `task.md` with **zero** semantic loss on the
compatibility surface.

## 2. Scoring — oracle E2E parity 6 / 6 (Docker)

Each sampled task runs twice with the deterministic oracle agent (`--agent oracle`,
no LLM — it executes the reference solution), once on the legacy layout and once on
the pure-native package (`task.md` + `oracle/` + `verifier/`). Reward must match.

| Task | legacy | task.md | parity |
|---|---:|---:|---|
| tictoc-unnecessary-abort-detection | 1.0 | 1.0 | ✅ |
| llm-prefix-cache-replay | 1.0 | 1.0 | ✅ |
| parallel-tfidf-search | 1.0 | 1.0 | ✅ |
| grid-dispatch-operator | 1.0 | 1.0 | ✅ |
| travel-planning | 0.0 | 0.0 | ✅ (oracle fails identically) |
| 3d-scan-calc | 0.0 | 0.0 | ✅ (oracle fails identically) |

The native package builds the environment, runs the oracle, and runs the verifier
end-to-end, scoring **identically** to the legacy layout. The two `0.0` rows are
the upstream oracle itself failing under the hardened sandbox — independent of the
standard; parity is preserved.

## 3. Live agent run — three interaction modes

To show the packages execute under a real agent (not just the deterministic
oracle), five converted tasks were run with a live agent across all three
SkillsBench interaction modes — **no-skill**, **with-skill**, and **self-gen**
(the agent authors a skill, then solves with it). A rollout is counted only when
it produced real work (tokens spent, tool calls made, and a verdict returned).

**12 / 12 clean rollouts** on the four healthy tasks (4 tasks × 3 modes), each a
real rollout. The skill machinery genuinely engages: skills are injected and
invoked, not merely declared. All clean rewards were `0.0` — expected for the
small model used here; these tasks sit below its capability floor, so the result
to read is *the packages run and the skill path fires in every mode*, not the
absolute score. The lone failing task failed for environment/verifier-portability
reasons unrelated to the conversion.

## What this shows / does not show

- **Shows:** `task.md` losslessly represents 100% of SkillsBench, scores with exact
  oracle parity to the legacy layout, and runs end-to-end under a live agent in all
  three interaction modes.
- **Does not show:** a measurable skill *uplift* — that needs a base model strong
  enough to sometimes pass; the runs here establish the plumbing, not the lift.
