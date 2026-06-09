# SkillsBench × task.md parity evidence

Date: 2026-06-08

Acceptance criterion (user): *can `task.md` actually run things and reach parity
with the original, especially SkillsBench.* Two levels were validated against the
full SkillsBench `main` suite (88 tasks, legacy Harbor layout: `task.toml` +
`instruction.md` + `environment/` + `solution/` + `tests/`).

## 1. Conversion parity — 88 / 88 (deterministic, no sandbox)

For every task, `build_harbor_roundtrip_conformance_report()` migrates
legacy → `task.md` → exports back to split and compares the supported
compatibility surface: canonical `TaskConfig`, normalized `instruction.md`
prompt, and SHA-256 file maps for `environment/`, `solution/`, and `tests/`.

```
=== SkillsBench conversion-parity: 88 tasks | PASS=88  MISMATCH=0  ERROR=0 ===
```

Every SkillsBench task is representable as native `task.md` with **zero**
semantic loss on the Harbor-compatible surface. Repro:
`experiments/skillsbench-taskmd-parity/skillsbench_parity.py`.

## 2. Oracle E2E parity — 6 / 6 sample (Docker)

Each sampled task is run twice with `bench eval create --agent oracle --sandbox
docker` — once on the legacy layout, once after `migrate_task_to_task_md(...,
remove_legacy=True)` produces a pure native package (`task.md` + `oracle/` +
`verifier/`). The oracle agent runs `solve.sh` with no LLM, so the reward is
deterministic. Assertion: `reward_legacy == reward_taskmd`.

| Task | legacy | task.md | parity |
|---|---:|---:|---|
| tictoc-unnecessary-abort-detection | 1.0 | 1.0 | ✅ |
| llm-prefix-cache-replay | 1.0 | 1.0 | ✅ |
| parallel-tfidf-search | 1.0 | 1.0 | ✅ |
| grid-dispatch-operator | 1.0 | 1.0 | ✅ |
| travel-planning | 0.0 | 0.0 | ✅ (oracle fails identically) |
| 3d-scan-calc | 0.0 | 0.0 | ✅ (oracle `solve.sh` rc=1) |

`PARITY(legacy==taskmd) = 6/6`. The native `task.md` package builds the
environment, runs the oracle, and runs the verifier end-to-end on Docker, scoring
**identically** to the legacy layout. The two `0.0` tasks are the SkillsBench
oracle itself failing (e.g. `solve.sh` exits non-zero, likely network/deps under
the hardened sandbox) — independent of `task.md`; parity is preserved. Repro:
`experiments/skillsbench-taskmd-parity/skillsbench_oracle_parity.py`.

## What this does and does not show

- **Shows:** `task.md` losslessly represents 100% of SkillsBench, and runs +
  scores end-to-end with exact oracle parity to legacy. Tier-1 ("BenchFlow
  self-adopts the standard") is validated for SkillsBench at the deterministic
  level.
- **Does not show:** full *agent-run* parity at scale — running a real agent
  (e.g. Gemini 3.1 Flash Lite) across all 88 tasks on Daytona at concurrency 100
  and comparing reward distributions legacy-vs-task.md. That is the next
  validation; it needs `DAYTONA_API_KEY` and a model budget, and reward variance
  must be handled statistically rather than as exact equality.
- **Surfaced (separate from task.md):** at least 2 SkillsBench oracles
  (`travel-planning`, `3d-scan-calc`) score 0.0 under the hardened sandbox — worth
  filing against SkillsBench independently.
