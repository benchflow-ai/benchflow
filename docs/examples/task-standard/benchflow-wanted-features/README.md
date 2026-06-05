# BenchFlow Wanted Feature Dogfood Tasks

These packages dogfood the draft task standard against real BenchFlow work we
want to do next. They are intentionally outside `docs/examples/task-md/**`
because they use proposed `benchflow:` and `verifier/verifier.md` fields that
parse as metadata today but are not fully executable runtime features yet.

Each package dogfoods a wanted BenchFlow feature. Partial runtime backing now
exists on the task-standard branch (see the dogfood report follow-up section):

- `runtime-capability-gate`: **mostly landed** — `TaskPackage` / `TaskRuntimeView`,
  docker/daytona capability gates, `bench tasks check --sandbox`.
- `verifier-package-reward-contract`: **mostly landed** — `VerifierDocument`,
  JSON-first rewards, multi-metric maps, strategy routing (script only);
  Reward Kit / agent-judge execution still open.
- `compat-export-loss-reports`: **mostly landed** — `bench tasks export` with
  `compatibility/export-report.json`; round-trip parity still open.
- `prompt-user-semantics`: **partial** — append/replace composition on scene
  turns; user/nudge loops fail-closed on docker/daytona until executable.

See the dogfood report:
[2026-06-05-task-standard-benchflow-dogfood.md](../../../reports/2026-06-05-task-standard-benchflow-dogfood.md).
