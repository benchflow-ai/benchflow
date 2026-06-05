# BenchFlow Wanted Feature Dogfood Tasks

These packages dogfood the draft task standard against real BenchFlow work we
want to do next. They are intentionally outside `docs/examples/task-md/**`
because they use proposed `benchflow:` and `verifier/verifier.md` fields that
parse as metadata today but are not fully executable runtime features yet.

Each package dogfoods a wanted BenchFlow feature. Partial runtime backing now
exists on the task-standard branch (see the dogfood report follow-up section):

- `runtime-capability-gate`: **partial** — `TaskPackage` / `TaskRuntimeView`,
  docker/daytona capability gates, `bench tasks check --sandbox`.
- `verifier-package-reward-contract`: **partial** — `VerifierDocument` parser,
  `reward.json` precedence, `reward-details.json` copy-through; strategy
  execution still open.
- `compat-export-loss-reports`: **partial** — `bench tasks export` with typed
  loss list; round-trip parity still open.
- `prompt-user-semantics`: **authoring only** — prompt composition and
  user/nudge runtime semantics not executable yet.

See the dogfood report:
[2026-06-05-task-standard-benchflow-dogfood.md](../../../reports/2026-06-05-task-standard-benchflow-dogfood.md).
