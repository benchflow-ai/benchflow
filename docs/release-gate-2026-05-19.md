# Trial-Ready Release Gate Evidence

Date: 2026-05-19

## Verdict

The focused trial-ready release gate is green for the current blocker set.
BenchFlow can run the selected real-suite evidence through the current
Rollout/Sandbox/Reward path, and the remaining work is release mechanics:
merge the clean adapter PRs plus the hosted-env source adapter, then cut the
release as `1.0.0`.

Do not tag the release from the open-PR state. Land the adapter and hosted-env
PRs first.

## Current Gate

| Area | Evidence |
|---|---|
| Trace-to-task e2e | `trace-to-task-e2e` generated tasks from JSONL and OpenTraces fixtures, checked them, and ran oracle evals with `reward=1.0`. |
| Release manifest | `uv run python tests/integration/run_suite.py --profile full-release --dry-run --fail-on-todo` passes. |
| Adapter release set | Harvey LAB, ProgramBench, SkillsBench, HILBench, OpaqueToolsBench, and CLBench all pass `adapter-release-set` evidence. |
| HILBench eval parity | PR #279 downloaded the HF bucket image, loaded/tagged it in Docker, built the generated task image, produced unsolved reward `0.000000`, and produced ground-truth reward `1.000000`. |
| Hosted env board | OpenReward, Harbor, and PrimeIntellect selected `env_uid` metadata is recorded; Harbor inventory emits `env_uid` and `hub_url`. |
| Docker/Daytona | Docker and Daytona smoke evidence exists for shared sandbox and Terminal-Bench-style tasks. |
| Modal | Optional follow-up evidence only; not a current release blocker. |
| Firecracker/K8s | Backlog only; not a current release blocker. |

## Adapter PR State

| PR | Benchmark | Head | GitHub state |
|---|---|---|---|
| #279 | HILBench | `d626d95bc304dd8256015d2d465aac55cd92bf31` | mergeable clean, `test` success |
| #280 | OpaqueToolsBench | `358fcfacd46505beb10f03f7d6f42de6c37073a4` | mergeable clean, `test` success |
| #283 | CLBench | `1415a9c04a04c1bfe75a5fb0c4104003482db9fe` | mergeable clean, `test` success |
| #290 | Hosted env source adapter | `41322da7d7b124695fad1b03ff9f06242b06a194` | `test` success; auxiliary review checks still pending |

## Commands

```bash
uv run python tests/integration/run_suite.py --profile full-release --dry-run --fail-on-todo

uv run python tests/integration/run_suite.py \
  --lane adapter-release-set \
  --execute-adapter-evidence \
  --skillsbench-result dogfood/2026-05-19-release-gate/jobs-skillsbench-docker/2026-05-19__02-16-37/jax-computing-basics__f921d900/result.json \
  --open-pr-root HILBench=/tmp/benchflow-release-pr279-final \
  --open-pr-root OpaqueToolsBench=/tmp/benchflow-release-pr280-20260519-021530 \
  --open-pr-root CLBench=/tmp/benchflow-release-pr283-final
```

## Artifact Pointers

Detailed run artifacts are intentionally under ignored `dogfood/` paths:

- `dogfood/2026-05-19-release-gate/report.md`
- `dogfood/2026-05-19-release-gate/adapter-evidence-refresh.txt`
- `dogfood/2026-05-19-release-gate/hilbench-hf-bucket-access.json`
- `dogfood/2026-05-19-release-gate/hilbench-pr279-eval/eval-parity-summary.json`
- `dogfood/2026-05-19-trace-to-task-e2e/trace-evidence.json`
- `dogfood/2026-05-19-release-gate/hosted-envs/hosted-env-evidence.json`
- `dogfood/2026-05-19-release-gate/hosted-envs/harbor-registry-inventory.jsonl`

## Versioning Call

Call this release `1.0.0` once PRs #279, #280, #283, and #290 land on
`main`. The evidence and product intent are trial-ready rather than another
internal `0.4.0` refactor cut.

Release sequence after merge approval:

1. Merge PRs #279, #280, #283, and #290.
2. Merge the release-gate evidence PR.
3. Bump `pyproject.toml` on `main` to `1.0.0`.
4. Tag `v1.0.0` on `main`.
5. Bump `main` to the next `.dev0`.
