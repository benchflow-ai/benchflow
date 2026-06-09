---
profile: [code-change, multi-agent, acceptance-live, leaderboard-local]
source: benchflow/wanted-features/runtime-capability-gate
name: benchflow-wanted/runtime-capability-gate
image: ubuntu:24.04
verifier: verifier/
oracle: oracle/
task:
  description: Validate the TaskPackage/TaskRuntimeView boundary and extend runtime capability validation.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, runtime, capability-gate]
metadata:
  feature_area: task-runtime
---

## prompt

Implement the next BenchFlow task-runtime slice in this repository.

BenchFlow now has a first `TaskRuntimeView` and `TaskPackage` boundary. Validate
that package as the single executable task view every runner can consult before
launching a sandbox, then extend it only where real runtime semantics still leak
out to one-off call sites.

Add a fail-closed runtime capability validator for parsed fields that are not
implemented by the selected sandbox yet: `steps`, artifacts, network allowlists,
separate verifier environments, Windows, TPU, healthcheck, workdir, non-`main`
verifier service, native/compatibility alias drift, `user`, `benchflow.nudges`,
and unsupported `benchflow.runtime_policy`.

## role:architect

Review whether the landed `TaskPackage` owns enough selected-state evidence for
rollout, verifier, adapter export, and authoring checks. Keep adapter import and
native authoring validation separate.

## role:implementer

Write the code and regression tests. Preserve existing passing behavior unless
the new capability validator intentionally rejects a previously silent
parse-only field in sandbox-aware validation.

## role:reviewer

Review the diff for hidden silent-degrade paths, missing negative tests, and
any accidental deprecation of Harbor/Pier split-layout compatibility.
