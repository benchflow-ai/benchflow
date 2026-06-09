# BenchFlow Wanted Feature Dogfood Tasks

These packages dogfood the draft task standard against real BenchFlow work we
want to do next. They are intentionally outside `docs/examples/task-md/**`
because they use proposed `benchflow:` and `verifier/verifier.md` fields that
parse today but are not all fully executable runtime features yet.

Each package is a task for an implementation agent. Some are now partially
implemented; the verifier scripts remain acceptance contracts for the full
feature. All six pass `bench tasks check` and the static native package gate
`bench tasks check --level publication-grade`; publication-grade includes an
explicit `reward_json` contract. They also pass the combined pre-publication
gate `bench tasks check --level publication-grade --sandbox docker`, which adds
sandbox-specific runtime capability validation to the native package checks.
All six also dogfood the static
`bench tasks check --level acceptance` evidence shape. That gate checks
declared oracle, verifier, review, calibration, trajectory, and artifact
evidence, including a verifier stability report with concrete rerun records,
and a calibration report with no-op, known-bad, partial, and reference cases.
It validates the declared JSON artifact contents against the metadata and
requires SHA-256 pins for primary evidence files. `runtime-capability-gate` also
dogfoods `bench tasks check --level acceptance-live --sandbox docker`, which
executes live verifier and oracle/reference cases for a leaderboard-oriented
report. `verifier-package-reward-contract` dogfoods a narrower acceptance-live
reference-verifier rerun for the native verifier package path, not a
leaderboard-suitability report. Use
`--report-output` when you want a fresh live report without refreshing a
checked-in report artifact.

- `runtime-capability-gate`: first slice implemented with `TaskRuntimeView`,
  `TaskPackage`, dedicated runtime capability tests, and fail-closed sandbox
  launch validation; prompt compilation now lives in `TaskPackage`, while
  local acceptance evidence under `evidence/acceptance/` proves the new static
  evidence gate, semantic artifact checks, verifier stability report contract,
  calibration report contract, and primary evidence hash pins on a real
  wanted-feature task; richer document-declared user/nudge runtime execution
  remains target work.
- `verifier-package-reward-contract`: parser plus runtime selection for
  `script`, `llm-judge`, `reward-kit`, and verifier-scoped `agent-judge`
  strategies are implemented with `VerifierDocument` and `verifier/verifier.md`;
  metrics-map aggregate policy execution is implemented, and selected Reward
  Kit criteria now govern metric ids, aggregate weights, and reward mismatch
  failures; fuller Harbor Reward Kit parity remains target work.
- `verifier-native-entrypoint`: a native verifier package with no
  `verifier/test.sh`; it selects a Reward Kit runner from `verifier/verifier.md`
  and dogfoods `TaskPaths.is_valid` following the selected verifier strategy
  instead of the legacy script filename.
- `compat-export-loss-reports`: native-to-Harbor/Pier split export, degraded
  export reports, and first foreign-extension preservation round trip are
  implemented; fuller import/export CLIs and target-specific adapters remain
  target work.
- `ors-episode-reward-contract`: selected `ors-episode` verifier strategies
  now consume declared ORS reward evidence, preserve reward events/details, and
  fail closed on missing, malformed, invalid, or non-terminal evidence; runtime
  helpers also normalize ORS tool-output rewards into
  `trajectory/ors-rewards.jsonl`; fuller OpenReward environment import/export
  and AgentBeats assessor lifecycles remain target work.
- `prompt-user-semantics`: append/replace prompt composition now compiles into a
  package-level prompt plan; deterministic and bounded model-linear user loops
  compile into `RolloutConfig.user` for supported linear single- and
  multi-scene document flows; the first sequential shared-workspace team
  handoff subset executes explicit multi-role scene turns and records handoff
  metadata; human confirmation policy installs a fail-closed permission gate
  unless an explicit handler is supplied; `branch_execution:
  option-kinds-preserved` names the supported branchable option-preservation
  slice; real forked branch execution, interactive approval UI, parallel teams,
  and rich handoff artifacts remain target work.

See the dogfood report:
[2026-06-05-task-standard-benchflow-dogfood.md](../../../reports/2026-06-05-task-standard-benchflow-dogfood.md).
