# Trial-Ready Release Gate Evidence

Date: 2026-05-19

## Verdict

The focused trial-ready release gate is green for the current blocker set.
BenchFlow can run the selected real-suite evidence through the current
Rollout/Sandbox/Reward path, and the remaining work is release mechanics:
update the release-blocker PR branches after the v0.4 squash merge, merge the
adapter and hosted-env source PRs, then cut the release as `1.0.0`.

Do not tag the release from the open-PR state. Land the adapter, hosted-env, and
release-gate PRs first.

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

## PR State After v0.4 Merge

PR #294 merged `refactor/v0.4` into `main` as a squash merge. The resulting
`main` tree matches `refactor/v0.4`, but several older PR branches now need a
branch update before GitHub will mark them mergeable again.

| PR | Area | Remote head | GitHub state | Local resolved head |
|---|---|---|---|---|
| #279 | HILBench | `d626d95bc304dd8256015d2d465aac55cd92bf31` | dirty/conflicting after #294; prior `test` success | `4c9e1c0` |
| #280 | OpaqueToolsBench | `358fcfacd46505beb10f03f7d6f42de6c37073a4` | mergeable clean, `test` success | no update needed |
| #283 | CLBench | `1415a9c04a04c1bfe75a5fb0c4104003482db9fe` | dirty/conflicting after #294; prior `test` success | `fe25d54` |
| #290 | Hosted env source adapter | `41322da7d7b124695fad1b03ff9f06242b06a194` | dirty/conflicting after #294; prior `test` success, Cursor Bugbot neutral | `bf1ba2b` |
| #291 | Pi ACP provider/model fix | `ba32d0b3d1dbf2839e834fcf64cb8aee96f8f999` | dirty/conflicting after #294; Devin Review success | `18fc9be` |
| #292 | Release gate evidence | `bde9fa66e9a66c68e8b5eea0ead5fefddc934f3a` | dirty/conflicting after #294; remote base still `refactor/v0.4` until retarget succeeds | current branch `HEAD` |

## Commands

Validation commands:

```bash
uv run python tests/integration/run_suite.py --profile full-release --dry-run --fail-on-todo

uv run python tests/integration/run_suite.py \
  --lane trace-to-task-e2e \
  --execute-trace-evidence \
  --run-trace-eval

uv run python tests/integration/run_suite.py \
  --lane hosted-env-compatibility-board \
  --execute-hosted-env-evidence

uv run python tests/integration/run_suite.py \
  --lane adapter-release-set \
  --execute-adapter-evidence \
  --skillsbench-result dogfood/2026-05-19-release-gate/jobs-skillsbench-docker/2026-05-19__02-16-37/jax-computing-basics__f921d900/result.json \
  --open-pr-root HILBench=/tmp/benchflow-release-pr279-final \
  --open-pr-root OpaqueToolsBench=/tmp/benchflow-release-pr280-20260519-021530 \
  --open-pr-root CLBench=/tmp/benchflow-release-pr283-final
```

Remote update commands prepared locally:

```bash
# Retarget #292 to the branch that now ships the v0.4 tree.
gh api \
  --method PATCH \
  repos/benchflow-ai/benchflow/pulls/292 \
  -f base=main

# Refresh dirty release-blocker PR branches after #294's squash merge.
git push origin handoff/pr279-hilbench-v04-main:devin/1778983541-hilbench-adapter
git push origin handoff/pr283-clbench-v04-main:devin/1779000478-clbench-adapter
git push origin handoff/pr290-hosted-env-v04-main:codex/hosted-env-adapter

# #291 is a fork PR with maintainer edits enabled. Push to the contributor fork.
git push https://github.com/Kfkcome/benchflow handoff/pr291-pi-acp-v04-main:fix/pi-acp-set-model-provider-prefix

# Publish this release-gate branch update.
git push origin HEAD:codex/trial-ready-release-gate
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

Call this release `1.0.0` once the dirty release-blocker PR branches are updated
and PRs #279, #280, #283, #290, #291, and #292 land on `main`. The evidence and
product intent are trial-ready rather than another internal `0.4.0` refactor
cut.

Release sequence after merge approval:

1. Retarget #292 from `refactor/v0.4` to `main`.
2. Push the prepared branch updates for #279, #283, #290, #291, and #292.
3. Wait for GitHub checks to rerun cleanly.
4. Merge PRs #279, #280, #283, #290, #291, and #292.
5. Bump `pyproject.toml` on `main` to `1.0.0`.
6. Tag `v1.0.0` on `main`.
7. Bump `main` to the next `.dev0`.
