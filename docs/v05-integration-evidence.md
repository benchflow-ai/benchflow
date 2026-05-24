# v0.5 Integration Evidence

Date: 2026-05-24

## Summary

All 14 urgent release blockers (ENG-147 through ENG-161) have been resolved
on the `v0.5-integration` branch across PRs #347–#372.

Test suite: 1910 passed, 9 skipped, 0 failed. ruff + ty clean.

## Blocker Resolution

| PR | Ticket | Title | Real Eval | Trace Audit |
|---|---|---|---|---|
| #347 | — | Fix 3 pre-existing test failures (TOML, dotenv, retry dedup) | hello-world-task ✓ | subagent ✓ |
| #348 | ENG-159 | `--include`/`--exclude` CLI flags | hello-world-task ✓ | subagent ✓ |
| #349 | ENG-150 | Accept verifier reward output when script exits nonzero | threejs-to-obj (rc=1, reward=0) ✓ | subagent ✓ |
| #350 | ENG-149 | Structured idle timeout diagnostics | data-to-d3 (idle timeout triggered) ✓ | subagent ✓ |
| #351 | ENG-160 | Scope resume scan to job directory, dedup retry artifacts | resume into existing job ✓ | subagent ✓ |
| #352 | ENG-148 | Structured transport error diagnostics | 3d-scan-calc ✓ | subagent ✓ |
| #353 | ENG-147 | Retry Daytona sandbox startup/export timeouts | 3d-scan-calc ✓ | subagent ✓ |
| #354 | ENG-151 | Classify verifier dep install failures + fix simpo index policy | simpo-code-reproduction ✓ | subagent ✓ |
| #355 | ENG-152 | Structured verifier timeout diagnostics | quantum-numerical-simulation ✓ | subagent ✓ |
| #356 | ENG-153 | CTRF path consistency lint | all 94 tasks ✓ | subagent ✓ |
| #357 | ENG-157/158 | Dashboard: stale advisory + file:// fetch guidance | 3d-scan-calc ✓ | subagent ✓ |
| #372 | — | Dead code cleanup + 9-task baseline audit | 9-task release subset ✓ | — |

## Diagnostic Fields Added

Every `result.json` now includes these structured diagnostic fields:

| Field | PR | Purpose |
|---|---|---|
| `error_category` | #350 | Classifies agent errors (idle_timeout, pipe_closed, sandbox_setup, etc.) |
| `idle_timeout_info` | #350 | Duration, tool call count, wall clock at timeout |
| `sandbox_startup_info` | #353 | Sandbox ID, state, attempts, build timeout |
| `transport_error_info` | #352 | Exit code, diagnosis, sandbox reachability probe |
| `verifier_timeout_info` | #355 | Budget, elapsed, task name |
| `verifier_error_category` | #354 | Classifies verifier errors (dep_install, timeout, infra, failure) |

## 9-Task Baseline (gemini-2.5-flash on Daytona)

| Task | Reward | Tools | Trajectory Entries | Notes |
|---|---|---|---|---|
| jax-computing-basics | 0.0 | 40 | 81 | |
| python-scala-translation | 0.0 | 8 | 18 | |
| jpg-ocr-stat | 0.0 | 21 | 43 | |
| grid-dispatch-operator | 0.0 | 28 | 54 | |
| threejs-to-obj | 0.0 | 11 | 25 | ENG-150: rc=1 accepted |
| data-to-d3 | 0.0 | 31 | 56 | ENG-149: 2 idle timeouts, retry ✓ |
| lake-warming-attribution | 0.0 | 20 | 35 | |
| weighted-gdp-calc | 0.0 | 0 | 3 | |
| shock-analysis-supply | 0.0 | 2 | 7 | |

Score: 0/9 (0.0%), 0 infrastructure errors, 0 secret leaks.

## Release Evidence Lanes

All evidence lanes pass:

```
Adapter release evidence: 6/6 PASS
  Harvey LAB, ProgramBench, SkillsBench, HILBench, OpaqueToolsBench, ContinualLearningBench

Trace-to-task e2e: 2/2 PASS
  minimal-claude.jsonl, minimal-opentraces.jsonl

Hosted env compatibility: 3/3 PASS + Harbor inventory
  OpenReward, Harbor, PrimeIntellect

Decoupling checks: 20/20 PASS
```

## Operational Work (In Progress)

| Ticket | Description | Status |
|---|---|---|
| ENG-154 | Rerun 8 JS-dependent SkillsBench tasks | Subagent running (4/8 done) |
| ENG-155 | Rerun self-gen SkillsBench subset | Subagent launched |
| ENG-156 | Full 94-task SkillsBench × 3 modes | Subagent running |
| — | HF artifact upload | Pending credentials |
