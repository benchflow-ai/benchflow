---
schema_version: "1.3"
source: benchflow/wanted-features/runtime-capability-gate
task:
  name: benchflow-wanted/runtime-capability-gate
  description: Validate the TaskPackage/TaskRuntimeView boundary and extend runtime capability validation.
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
  docker_image: ubuntu:24.04
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
      - "one TaskPackage exposes the selected task entrypoint, prompt, verifier dir, oracle dir, source hashes, compatibility metadata, verifier document, and runtime support issues"
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
    oracle_runs:
      required_reward: 1.0
      artifact: evidence/acceptance/oracle-run.json
    verifier:
      reruns: 5
      flake_rate: 0.0
      report: evidence/acceptance/verifier-stability-report.json
    review:
      anti_cheat: passed
      instruction_alignment: passed
      reviewer: benchflow-task-standard-dogfood
      artifact: evidence/acceptance/review.json
    calibration:
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      partial_solution_range: [0.4, 0.8]
      report: evidence/acceptance/calibration-report.json
      human_or_reference_examples:
        - name: gold-runtime-boundary
          expected_reward: 1.0
          artifact: evidence/acceptance/gold-result.json
    acceptance_live:
      report: evidence/acceptance/live-report.json
      workspace:
        source: current-worktree
        target: /repo
      calibration:
        from: calibration.report
        reruns: 1
        flake_rate_max: 0.0
      leaderboard:
        required: true
        max_flake_rate: 0.0
      cases:
        - name: live-oracle-rerun
          type: oracle
          reruns: 1
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
        - name: live-reference-verifier
          type: reference
          reruns: 1
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
    trajectories:
      - path: evidence/acceptance/gold-trajectory.jsonl
        sha256: 0c207c024f0bed409c534c6e7603fb7399b7797d1ef81a45ff94b67d74be755f
    artifacts:
      - path: evidence/acceptance/oracle-run.json
        sha256: 7f4d2a632c5ad1e8d87218b7a9f98109a1cc714af94dbb4e55eda21d5d244d91
      - path: evidence/acceptance/gold-result.json
        sha256: 314550225da3542c9671d7b8c80d1010e300b692a8c31e398787f9c9f8450a7d
      - path: evidence/acceptance/calibration-report.json
        sha256: d37de11fa5984d9b647622aa27c57105c77e7bfd0b615d13dae7530e6075b65c
      - path: evidence/acceptance/verifier-stability-report.json
        sha256: 21b39867b6d9d7b8077e258691154f109db0fc20c8124e5f32e776d5f705f904
      - path: evidence/acceptance/review.json
        sha256: ab38687d8896eb091936ab5bdadf73cd2f9fa9be4e509fdc3ef5d817a68c10be
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
