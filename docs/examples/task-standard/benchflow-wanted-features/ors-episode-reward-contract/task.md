---
schema_version: "1.3"
source: benchflow/wanted-features/ors-episode-reward-contract
task:
  name: benchflow-wanted/ors-episode-reward-contract
  description: Add a verifier-scoped ORS episode reward strategy for task.md packages.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, verifier, openreward, ors, rewards]
metadata:
  category: benchflow-feature
  feature_area: verifier
agent:
  timeout_sec: 7200
  network_mode: no-network
verifier:
  timeout_sec: 1200
  type: test-script
environment:
  network_mode: no-network
  allow_internet: false
  docker_image: ghcr.io/astral-sh/uv:python3.12-bookworm
  cpus: 2
  memory_mb: 4096
  storage_mb: 10240
agents:
  roles:
    implementer:
      agent: codex
      responsibilities:
        - normalize ORS tool-output rewards into trajectory evidence
        - implement verifier-scoped ORS reward evidence normalization
        - preserve reward events and terminal aggregate in reward-details.json
    reviewer:
      agent: codex
      responsibilities:
        - verify malformed ORS evidence fails closed
        - ensure ORS support stays under verifier scope
scenes:
  - name: implement
    roles: [implementer]
    prompt: Build the smallest executable ORS episode verifier strategy.
  - name: review
    roles: [reviewer]
    prompt: Audit the ORS reward contract, tests, docs, and dogfood package.
benchflow:
  document_version: "0.4"
  compatibility:
    harbor:
      export: degraded
      losses: [ors_episode_strategy, reward_event_stream]
    ors:
      export: native
      runtime_tool_outputs: trajectory/ors-rewards.jsonl
      reward_events: required
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/verifier.md
    implementation:
      type: hybrid
      strategies:
        ors_episode: trajectory/ors-rewards.jsonl
    outputs:
      reward_json: /logs/verifier/reward.json
      reward_details_json: /logs/verifier/reward-details.json
  evidence:
    oracle_runs:
      required_reward: 1.0
      artifact: evidence/acceptance/oracle-run.json
    verifier:
      reruns: 3
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
        - name: gold-ors-episode
          expected_reward: 1.0
          artifact: evidence/acceptance/gold-result.json
      fixtures:
        - verifier/fixtures/ors-tool-outputs.example.jsonl
        - verifier/fixtures/ors-rewards.example.jsonl
    trajectories:
      - path: evidence/acceptance/gold-trajectory.jsonl
        sha256: a887ddd383d5f7a3f05e0422312b749c79e2eb32a57afbac59ca5682d7baa833
    artifacts:
      - path: evidence/acceptance/oracle-run.json
        sha256: 1c26be25853db3a2fbf9aee2104164c7184ba2347899614a62e42e13c36eb5c5
      - path: evidence/acceptance/gold-result.json
        sha256: 4e92a1999b6801d22d8c578f8354d021b29526d01852b796cc0c4a6d37b2baa4
      - path: evidence/acceptance/calibration-report.json
        sha256: e1b1de755f883535c39e2a9fb192430ba065f2dfadb7ff2eb2f281781725830d
      - path: evidence/acceptance/verifier-stability-report.json
        sha256: a91f2afc0bee4239df2483d67bab9c6bb57a75466042f3239880f2d52b8f3e2d
      - path: evidence/acceptance/review.json
        sha256: 0366f9fbd99606469daca3a37329cd3e8f710d37fc8d96b1aa88a6b3b7979439
      - path: verifier/fixtures/ors-tool-outputs.example.jsonl
        sha256: 9f36310e5992ae734969d37b45e170fd72c5c2858f4735fbb13a5e83a22a1e16
      - path: verifier/fixtures/ors-rewards.example.jsonl
        sha256: 31b52d0701a58a3e25685d778d9e653152e234fc6ca49926ae499bdf97b6a9b8
---

## prompt

Implement first-class support for an `ors-episode` strategy in
`verifier/verifier.md`.

The runtime side must be able to normalize ORS tool-output rewards into
`trajectory/ors-rewards.jsonl`. The verifier side must read declared evidence
inputs, normalize ORS reward responses or reward-event streams into BenchFlow's
reward envelope, emit `/logs/verifier/reward.json`, preserve the ORS response
in `/logs/verifier/reward-details.json`, and fail closed on malformed or
non-terminal reward evidence.

## role:implementer

Keep ORS support inside verifier scope. Do not add root task syntax for ORS
episodes when `verifier/verifier.md` can declare the strategy.

## role:reviewer

Check the declared-input boundary. Relative evidence paths must resolve under
rollout evidence, absolute paths must be explicitly downloaded, and missing or
invalid evidence must be a verifier error.

## scene:implement

Build the parser/runtime slice and focused tests.

## scene:review

Run the dogfood package through structural and sandbox-aware checks, then
summarize remaining ORS/AgentBeats gaps honestly.
