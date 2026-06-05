# Trial-Ready Release Gate Evidence

Date: 2026-05-19

> **Historical note:** this dated gate predates the shipped public `0.5.2`
> release and the public/internal preview publishing split. Current
> release-channel mechanics live in [`docs/release.md`](./release.md).

## Verdict

The focused trial-ready release gate is green for the current blocker set.
BenchFlow can run the selected real-suite evidence through the current
Rollout/Sandbox/Reward path, and the remaining work is release mechanics:
merge the adapter, hosted-env, Pi ACP, and release-gate PRs, then cut the
release as `1.0.0`.

Do not tag the release from the open-PR state. Land the adapter, hosted-env, and
release-gate PRs first.

## Current Gate

| Area | Evidence |
|---|---|
| Trace-to-task e2e | `trace-to-task-e2e` generated tasks from JSONL and OpenTraces fixtures, checked them, and ran oracle evals with `reward=1.0`. |
| Release manifest | `uv run python tests/integration/run_suite.py --profile full-release --dry-run --fail-on-todo` passes. |
| Adapter release set | Harvey LAB, ProgramBench, SkillsBench, HILBench, OpaqueToolsBench, and ContinualLearningBench all pass `adapter-release-set` evidence. |
| HILBench eval parity | PR #279 downloaded the HF bucket image, loaded/tagged it in Docker, built the generated task image, produced unsolved reward `0.000000`, and produced ground-truth reward `1.000000`. |
| Hosted env board | OpenReward, Harbor, and PrimeIntellect selected `env_uid` metadata is recorded; Harbor inventory emits `env_uid` and `hub_url`. |
| Docker/Daytona | Docker and Daytona smoke evidence exists for shared sandbox and Terminal-Bench-style tasks. |
| Modal | Optional follow-up evidence only; not a current release blocker. |
| Firecracker/K8s | Backlog only; not a current release blocker. |

## PR State After v0.5 Merge Packet (2026-05-20)

The release-blocker adapter and CLI PRs landed on `main` via the merge queue:

| PR | Area | Status |
|---|---|---|
| #290 | Hosted env source adapter | merged |
| #296 | Codex subscription auth in Daytona | merged |
| #298 | v0.5 dev bump | merged |
| #299 | CLI tasks generate flags | merged |
| #300 | Pi ACP provider prefix fix | merged (supersedes #291) |
| #301 | OpaqueToolsBench adapter | merged (supersedes #280) |
| #302 | HILBench adapter | merged (supersedes #279) |
| #303 | ContinualLearningBench adapter (ENG-103) | merged (supersedes #283) |
| #295 | Release gate evidence refresh | in queue |

Stale PRs closed: #297 (superseded by #298), #291 (superseded by #300).

## PR State After v0.4 Merge

PR #294 merged `refactor/v0.4` into `main` as a squash merge. The resulting
`main` tree matches `refactor/v0.4`. The release-blocker PR branches have now
been updated against `main`; GitHub marks them mergeable, with the repository
`test` check green where GitHub runs checks for the branch.

| PR | Area | Remote head | GitHub state | Local resolved head |
|---|---|---|---|---|
| #279 | HILBench | `f6f9c4d` | mergeable, `test` success | `f6f9c4d` |
| #280 | OpaqueToolsBench | `358fcfacd46505beb10f03f7d6f42de6c37073a4` | mergeable clean, `test` success | no update needed |
| #283 | ContinualLearningBench | `1bd8bc4` | mergeable, `test` success | `1bd8bc4` |
| #290 | Hosted env source adapter | `ceef534` | mergeable, repository `test` success | `ceef534` |
| #291 | Pi ACP provider/model fix | `c57c95c` | mergeable; fork PR has no check rollup | `c57c95c` |
| #292 | Release gate evidence | this PR | mergeable, repository `test` success | this PR |

## Integrated Release Candidate

The local branch `handoff/trial-ready-release-integrated` merges the current
`origin/main` tree with #279, #280, #283, #290, #291, and #292 plus two
integration fixups discovered during the local merge proof. Use
`git rev-parse handoff/trial-ready-release-integrated` for the current local
head because this handoff branch moves as the release-gate notes are refreshed.

- preserve executable bits on existing shell fixtures touched by #279/#283;
- reconcile the hosted-env CLI additions with release-gate CLI error/help
  behavior.

Validation on that integrated branch:

- `uv run --extra dev python -m pytest tests/`: 1164 passed, 14 skipped, 1 deselected.
- `uv run --extra dev ruff check .`: passed.
- `uv run --extra dev ty check src/`: passed.
- `uv run python tests/integration/run_suite.py --profile full-release --dry-run --fail-on-todo`: passed.

## Merge Sequence

The release PRs are individually mergeable against the current `main`, but the
release should not be landed by blindly merging raw #292 after #279, #280,
#283, #290, and #291. A local sequence simulation showed that order can expose a
`src/benchflow/cli/main.py` conflict between the hosted-env CLI additions and
the release-gate eval error-handling changes.

PR #292 has already been updated to a sequence-safe release-gate refresh. The
repository `test` check is green on the pushed code-evidence head `8c58403`;
later docs-only refreshes do not change the tested release-gate code path. The
guarded local helper now documents the approval-gated merge flow:

```bash
RUN_PREFLIGHT=1 bash dogfood/2026-05-19-release-gate/remote-handoff/merge-release-prs-after-approval.sh
RUN_MERGE=1 bash dogfood/2026-05-19-release-gate/remote-handoff/merge-release-prs-after-approval.sh
```

Use GitHub merge commits for this release packet. If earlier PRs are
squash-merged instead, refresh #292 again on the post-squash `main` before
merging it.

The packaging-only branch `handoff/release-1.0.0-rc-refresh` bumps
`pyproject.toml` and the local package entry in `uv.lock` to `1.0.0`.
`dogfood/2026-05-19-release-gate/remote-handoff/package-rc-preflight.sh`
builds that RC head, runs `twine check`, and verifies both wheel `METADATA` and
sdist `PKG-INFO` report `Version: 1.0.0`.

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
  --open-pr-root ContinualLearningBench=/tmp/benchflow-release-pr283-final
```

Remote update status:

- Completed: #292 was retargeted from `refactor/v0.4` to `main`.
- Completed: #279, #283, #290, #291, and #292 were fast-forwarded to resolved
  heads after #294's squash merge.
- Completed: #279, #283, #290, and #292 reran the GitHub `test` workflow
  successfully after the branch updates. #291 is a fork PR with no check rollup.
- Third-party automation such as Cursor Bugbot is tracked separately from the
  release gate; the gating repository `test` workflow has passed.
- Handoff bundle: `dogfood/2026-05-19-release-gate/remote-handoff/release-pr-handoff.bundle`
  contains the prepared PR heads, the integrated proof branch, and the 1.0.0 RC
  branch.
- Guarded push helper: `dogfood/2026-05-19-release-gate/remote-handoff/push-release-pr-heads.sh`
  verifies exact local heads and defaults to dry-run. It was used to perform the
  release-blocker branch updates.

Remote branch update commands used locally:

```bash
RUN_PUSH=1 bash dogfood/2026-05-19-release-gate/remote-handoff/push-release-pr-heads.sh
```

## Artifact Pointers

Detailed run artifacts are intentionally under ignored `dogfood/` paths:

- `dogfood/2026-05-19-release-gate/report.md`
- `dogfood/2026-05-19-release-gate/adapter-evidence-refresh.txt`
- `dogfood/2026-05-19-release-gate/hilbench-hf-bucket-access.json`
- `dogfood/2026-05-19-release-gate/hilbench-pr279-eval/eval-parity-summary.json`
- `dogfood/2026-05-19-trace-to-task-e2e/trace-evidence.json`
- `dogfood/2026-05-19-trace-to-task-e2e/jobs-oracle-jsonl-fixture/2026-05-19__05-47-57/create-hello-txt-with-exactly-hello-from-b2c25762__a7f9dc42/result.json`
- `dogfood/2026-05-19-trace-to-task-e2e/jobs-oracle-opentraces-fixture/2026-05-19__05-48-10/create-a-file-named-opentraces-txt-with-9f87605e__a7e18623/result.json`
- `dogfood/2026-05-19-release-gate/hosted-envs/hosted-env-evidence.json`
- `dogfood/2026-05-19-release-gate/hosted-envs/harbor-registry-inventory.jsonl`
- `dogfood/2026-05-19-release-gate/package-rc-preflight.json`
- `dogfood/2026-05-19-release-gate/remote-handoff/README.md`
- `dogfood/2026-05-19-release-gate/remote-handoff/release-pr-handoff.bundle`

## Versioning Call

This dated gate originally proposed `1.0.0` once PRs #279, #280, #283, #290,
#291, and #292 landed on `main`. The shipped public release is `0.5.2`; this
document is retained as historical evidence for the trial-ready gate and the
internal v0.4 refactor cut.

Release sequence after merge approval:

> Historical note: this dated gate predates the public/internal preview
> publishing split. Current release-channel mechanics live in
> [`docs/release.md`](./release.md).

1. Merge PRs #279, #280, #283, #290, #291, and #292.
2. Bump `pyproject.toml` on `main` to `1.0.0`.
3. Tag `v1.0.0` on `main`.
4. Bump `main` to the next `.dev0`.
