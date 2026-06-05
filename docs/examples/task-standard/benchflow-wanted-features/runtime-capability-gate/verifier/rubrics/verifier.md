# Runtime Capability Gate Rubric

- `runtime_view`: `TaskPackage` / `TaskRuntimeView` exists and owns selected
  entrypoint, prompt, verifier/oracle dirs, source hashes, scenes, and
  compatibility metadata.
- `fail_closed`: sandbox-aware validation rejects unsupported parsed features
  before sandbox creation.
- `compatibility`: Harbor/Pier split-layout imports remain valid and
  native/compatibility drift is detected.
- `maintainability`: rollout, path helpers, and adapters call the same runtime
  view instead of reimplementing selection locally.

