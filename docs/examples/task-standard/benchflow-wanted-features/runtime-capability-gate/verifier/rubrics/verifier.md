# Runtime Capability Gate Rubric

- `runtime_package`: `TaskPackage` is the package-level boundary wrapping the
  selected `TaskRuntimeView`, verifier document metadata, compatibility state,
  source hashes, and sandbox runtime issues.
- `fail_closed`: sandbox-aware validation rejects unsupported parsed features
  before sandbox creation.
- `acceptance_live`: `acceptance-live` owns executable oracle/verifier reruns,
  persisted live reports, flake-threshold enforcement, and generated live
  calibration cases from static calibration reports.
- `compatibility`: Harbor/Pier split-layout imports remain valid and
  native/compatibility drift is detected.
- `runtime_view`: rollout, path helpers, authoring checks, and adapters call the
  same package/view boundary instead of reimplementing selection locally.
