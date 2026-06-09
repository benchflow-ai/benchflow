---
schema_version: "1.3"
source: benchflow/wanted-features/verifier-native-entrypoint
task:
  name: benchflow-wanted/verifier-native-entrypoint
  description: Make verifier/verifier.md a true native verifier entrypoint, not just metadata beside test.sh.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, verifier, reward-kit, native-entrypoint]
metadata:
  category: benchflow-feature
  feature_area: verifier
agent:
  timeout_sec: 3600
  network_mode: no-network
verifier:
  timeout_sec: 900
  user: root
environment:
  docker_image: ghcr.io/astral-sh/uv:python3.12-bookworm
  network_mode: no-network
  cpus: 2
  memory_mb: 4096
  storage_mb: 4096
  workdir: /repo
agents:
  roles:
    implementer:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: high
      capabilities: [code-edit, tests]
scenes:
  - name: implement
    turns:
      - role: implementer
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F3, F7, F8]
    user_story: "Verifier authors need native verifier packages to be valid when verifier/verifier.md selects an executable strategy, even if no legacy test.sh exists."
    acceptance:
      - "TaskPaths.is_valid accepts a native verifier package whose selected reward-kit strategy has a valid runner and criteria file but no verifier/test.sh"
      - "TaskPaths.is_valid rejects the same package when the selected reward-kit runner is missing"
      - "bench tasks check accepts this dogfood package structurally and with --sandbox docker"
  verifier:
    spec: verifier/verifier.md
    structured_rubric: verifier/rubrics/verifier.toml
    reward_kit: verifier/reward_kit/
    implementation:
      type: hybrid
      strategies:
        rewardkit: verifier/reward_kit/reward.py
      outputs:
        reward_json: /logs/verifier/reward.json
        reward_details: /logs/verifier/reward-details.json
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
        - name: gold-native-entrypoint
          expected_reward: 1.0
          artifact: evidence/acceptance/gold-result.json
    trajectories:
      - path: evidence/acceptance/gold-trajectory.jsonl
        sha256: d9bfb2a83a684d2d50105ab356a0729b5e87b7d1620c81b9ee4da9016a452eb7
    artifacts:
      - path: evidence/acceptance/oracle-run.json
        sha256: 6bbdae3af63d860fd15f11b74b5b97b2974fef1231ccd270176bb77e08ad1988
      - path: evidence/acceptance/gold-result.json
        sha256: 2309df8b66b9b141c73f383c85667e8fe6a827c74380c92afcd0c35487618e96
      - path: evidence/acceptance/calibration-report.json
        sha256: 4912efb68ebe113b319ded39af60bd144bf9f2a8ac4a307eec1605015730773f
      - path: evidence/acceptance/verifier-stability-report.json
        sha256: 36314f4b293cb2973a5907793742f0b24bff6158540f6d705e0890be98121e51
      - path: evidence/acceptance/review.json
        sha256: c2cfa6be52cb3f649e83cc5ea8980b2e73d0fbbf4d390eb19b948137884a7804
---

## prompt

Make native verifier packages valid when `verifier/verifier.md` selects an
executable strategy, even when the package intentionally has no
`verifier/test.sh`.

Start with the concrete mismatch: `Verifier.verify()` can execute selected
`reward-kit`, `llm-judge`, and `agent-judge` strategies, but shared path
validity still treats `test.sh` as mandatory.

## role:implementer

Keep legacy `tests/test.sh` and native `verifier/test.sh` behavior working.
Add the narrowest structural validity logic needed for selected verifier
strategies, with fail-closed behavior for malformed documents and missing
Reward Kit runners.
