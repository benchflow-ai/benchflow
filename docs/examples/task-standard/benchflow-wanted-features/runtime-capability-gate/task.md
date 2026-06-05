---
schema_version: "1.3"
source: benchflow/wanted-features/runtime-capability-gate
task:
  name: benchflow-wanted/runtime-capability-gate
  description: Add TaskPackage/TaskRuntimeView and fail-closed runtime capability validation.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, runtime, capability-gate]
metadata:
  category: benchflow-feature
  feature_area: task-runtime
agent:
  timeout_sec: 7200
  network_mode: no-network
verifier:
  timeout_sec: 1200
  user: root
environment:
  docker_image: ghcr.io/astral-sh/uv:python3.12-bookworm
  network_mode: no-network
  cpus: 4
  memory_mb: 8192
  storage_mb: 10240
  workdir: /repo
agents:
  roles:
    architect:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: xhigh
      capabilities: [code-edit, tests]
    implementer:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: xhigh
      capabilities: [code-edit, tests]
    reviewer:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
      capabilities: [review]
scenes:
  - name: design
    turns:
      - role: architect
  - name: implement
    turns:
      - role: implementer
  - name: review
    turns:
      - role: reviewer
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F1, F2, F3, F4, F7]
    user_story: "BenchFlow needs one executable task view that selects native vs compatibility files and refuses parsed runtime semantics that the chosen sandbox cannot honor."
    acceptance:
      - "one TaskRuntimeView answers selected task entrypoint, prompt, verifier dir, oracle dir, source hashes, and compatibility metadata"
      - "runtime capability validation fails closed for parsed unsupported Harbor fields before sandbox creation"
      - "native/compatibility alias drift is detected instead of silently preferring native paths"
  prompt:
    composition: append
    order: [base, role, scene, turn]
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/verifier.md
    entrypoint: verifier/test.sh
    implementation:
      type: test-script
      outputs:
        reward_json: /logs/verifier/reward.json
        reward_details: /logs/verifier/reward-details.json
  evidence:
    calibration:
      oracle_reward: 1.0
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
---

## prompt

Implement the next BenchFlow task-runtime slice in this repository.

Build a single executable task view that every runner can consult before
launching a sandbox. It should select the authoritative task definition,
materialized prompt, verifier directory, oracle directory, source hashes, scene
metadata, and compatibility status.

Add a fail-closed runtime capability validator for parsed fields that are not
implemented by the selected sandbox yet: `steps`, artifacts, network allowlists,
separate verifier environments, Windows, TPU, healthcheck, workdir, non-`main`
verifier service, native/compatibility alias drift, `user`, `benchflow.nudges`,
and unsupported `benchflow.runtime_policy`.

## role:architect

Propose the smallest module boundary that can become the future
`TaskPackage` / `TaskRuntimeView`. Keep adapter import and native authoring
validation separate.

## role:implementer

Write the code and regression tests. Preserve existing passing behavior unless
the new capability validator intentionally rejects a previously silent
parse-only field in sandbox-aware validation.

## role:reviewer

Review the diff for hidden silent-degrade paths, missing negative tests, and
any accidental deprecation of Harbor/Pier split-layout compatibility.

