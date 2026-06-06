# Task Standard Dogfood: BenchFlow Wanted Features

Date: 2026-06-05

This dogfood pass used the draft task standard to package real BenchFlow
features we want next. The goal was not to invent benchmark toys; it was to ask
whether `task.md` plus a first-class `verifier/` package can describe work an
implementation agent could actually pick up.

## Packages Created

| Package | Real feature | Standard surfaces exercised | Current validation |
|---|---|---|---|
| [runtime-capability-gate](../examples/task-standard/benchflow-wanted-features/runtime-capability-gate/task.md) | `TaskPackage` / `TaskRuntimeView`, sandbox-aware capability validation, native/compat alias drift | F1-F4/F7, `benchflow.prompt`, `benchflow.verifier`, calibration evidence, multi-scene roles | `TaskDocument` parses; `bench tasks check` passes |
| [verifier-package-reward-contract](../examples/task-standard/benchflow-wanted-features/verifier-package-reward-contract/task.md) | `VerifierDocument`, `verifier/verifier.md`, Reward Kit, agent judge, `reward.json` precedence, `reward-details.json` | F3/F8, verifier package, structured rubric, hybrid strategies | `TaskDocument` parses; `bench tasks check` passes |
| [compat-export-loss-reports](../examples/task-standard/benchflow-wanted-features/compat-export-loss-reports/task.md) | Harbor/Pier split export, degraded export reports, foreign extension preservation | F1/F2/F6/F7, compatibility map, export losses | `TaskDocument` parses; `bench tasks check` passes |
| [prompt-user-semantics](../examples/task-standard/benchflow-wanted-features/prompt-user-semantics/task.md) | Prompt append/replace semantics, document-declared user/nudge loop, private fact handling | F5/F7, `benchflow.prompt`, `user`, `## user-persona`, `benchflow.nudges` | `TaskDocument` parses; `bench tasks check` passes |

All packages live under
[docs/examples/task-standard/benchflow-wanted-features](../examples/task-standard/benchflow-wanted-features/README.md).

## Checks Run

Parsed every package with the current `TaskDocument` parser:

```text
OK compat-export-loss-reports/task.md
OK prompt-user-semantics/task.md
OK runtime-capability-gate/task.md
OK verifier-package-reward-contract/task.md
```

Ran the current structural checker:

```text
bench tasks check docs/examples/task-standard/benchflow-wanted-features/*/
```

Result: all four task directories were reported valid.

The verifier `test.sh` files are intentionally future acceptance scripts. They
should not pass today because they check for wanted files such as
`src/benchflow/task/runtime_capabilities.py`,
`src/benchflow/task/verifier_document.py`, and export tests that do not exist
yet.

## What Worked

The root `task.md` shape handled real BenchFlow feature work well:

- Harbor-compatible config stayed small and parseable.
- `benchflow:` carried draft semantics without root-key creep.
- Roles/scenes gave each task a realistic implementation/review workflow.
- `oracle/solve.md` was a useful place for reference implementation intent.
- `verifier/verifier.md` made the acceptance contract much easier to review
  than a bare `test.sh`.

The `solution -> oracle` and `tests -> verifier` naming also held up: the
feature tasks read better when `oracle/` means reference behavior and
`verifier/` means the whole evaluator-side contract.

## What Broke Or Felt Thin

1. Current validation is too shallow.

   `bench tasks check` passes packages whose `verifier/verifier.md` semantics
   are not parsed or executable. That is correct for today's structural check,
   but it proves we need a separate capability-aware check such as:

   ```bash
   bench tasks check --sandbox docker --capabilities
   ```

2. `verifier/verifier.md` is not first-class yet.

   The dogfood packages can include verifier strategies, rubrics, judge roles,
   Reward Kit pointers, and `reward-details.json`, but current code ignores
   those files. This validates the need for `VerifierDocument`.

3. The standard needs a clear "future acceptance script" convention.

   These wanted-feature packages include `verifier/test.sh` scripts that are
   supposed to fail until the feature is implemented. The package standard
   should distinguish:

   - structural validity
   - runnable task validity
   - feature acceptance validity
   - leaderboard publication validity

4. The prompt/user task exposed a hard runtime question.

   `user`, `## user-persona`, and `benchflow.nudges` are natural to author, but
   they are not useful unless `TaskRuntimeView` can say whether a selected
   rollout backend will actually compile them into a user loop.

5. Export tasks need a typed loss-report artifact.

   The compatibility package can describe desired losses in `benchflow`, but
   the standard should eventually name a concrete output file such as
   `/logs/verifier/export-loss-report.json` or
   `compatibility/export-report.json`.

## Standard Changes Suggested By Dogfood

1. Add `VerifierDocument` to the implementation roadmap before broadening
   verifier strategies.
2. Define validation levels:
   `structural`, `runtime-capability`, `acceptance`, and `publication-grade`.
3. Give future-acceptance packages an explicit status field, for example:

   ```yaml
   benchflow:
     status: wanted-feature
     runnable_today: false
   ```

4. Add a typed compatibility loss report schema.
5. Add `TaskRuntimeView` before implementing prompt/user semantics, because
   prompt composition, user loops, and runtime support all need one selected
   view of the task package.

## Implementation Follow-Up (2026-06-05, same branch)

The wanted-feature dogfood packages now have partial runtime backing on
`cursor/task-standard-runtime-gaps-e453`:

| Wanted feature | Landed | Still open |
|---|---|---|
| `TaskPackage` / `TaskRuntimeView` | `src/benchflow/task/package.py`, rollout upload + runtime summary logging | Source-hash provenance, full prompt composition on base `/instruction.md` |
| Runtime capability gates | `src/benchflow/task/runtime_capabilities.py`, docker/modal `environment.workdir`, healthcheck fail-closed; `bench tasks check --sandbox` | daytona workdir, K8s/podman matrices, GPU/private-mount gates |
| `VerifierDocument` | `src/benchflow/task/verifier_document.py`, script/reward-kit/agent-judge routing in `Verifier` | ORS/hybrid strategies, full rubric-dimension orchestration |
| Export + loss reports | `src/benchflow/task/export.py`, `bench tasks export/import/round-trip` | Pier-specific losses; full foreign import preservation |
| Reward contract | JSON-first precedence, multi-metric maps, `reward-details.json` in `result.json` | Reward Kit execution, aggregate policy from verifier metadata |
| Prompt/user semantics | `prompt_composition.py`, `user_loop.py`, multi-scene `user-loop` auto-wiring | Branchable nudges, scene-level user-loop metadata |

Regression guard: `tests/test_task_standard_dogfood.py` exercises all four
wanted-feature packages through structural check, `Task()` load,
`TaskRuntimeView`, export losses, and docker capability rejection for
`runtime-capability-gate`.

## Decision

The standard is useful for real BenchFlow work. The core missing product
surface — runtime view plus verifier package parser — now exists in draft form.
Remaining work is execution depth (ORS/hybrid verifier strategies,
branchable nudges, K8s/podman capability matrices), not another authoring-format spike.

