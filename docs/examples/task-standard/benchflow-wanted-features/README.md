# BenchFlow Wanted Feature Dogfood Tasks

These packages dogfood the draft task standard against real BenchFlow work we
want to do next. They are intentionally outside `docs/examples/task-md/**`
because they use proposed `benchflow:` and `verifier/verifier.md` fields that
parse as metadata today but are not fully executable runtime features yet.

Each package is a task for a future implementation agent:

- `runtime-capability-gate`: build `TaskPackage` / `TaskRuntimeView` and
  fail-closed runtime capability validation.
- `verifier-package-reward-contract`: add `VerifierDocument`,
  `verifier/verifier.md`, Reward Kit preservation, and `reward.json`
  precedence.
- `compat-export-loss-reports`: add native-to-Harbor/Pier export and degraded
  export reports.
- `prompt-user-semantics`: implement prompt composition and document-declared
  user/nudge runtime semantics.

See the dogfood report:
[2026-06-05-task-standard-benchflow-dogfood.md](../../../reports/2026-06-05-task-standard-benchflow-dogfood.md).
