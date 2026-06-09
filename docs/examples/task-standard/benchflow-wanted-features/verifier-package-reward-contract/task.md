---
schema_version: "1.3"
source: benchflow/wanted-features/verifier-package-reward-contract
task:
  name: benchflow-wanted/verifier-package-reward-contract
  description: Add verifier/verifier.md, verifier strategies, and rich reward artifact preservation.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, verifier, reward-kit, agent-judge]
metadata:
  category: benchflow-feature
  feature_area: verifier
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
    implementer:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: xhigh
      capabilities: [code-edit, tests]
    judge_designer:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
      capabilities: [review]
scenes:
  - name: implement
    turns:
      - role: implementer
  - name: rubric-review
    turns:
      - role: judge_designer
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F3, F8]
    user_story: "Verifier authors need a first-class package for deterministic checks, Reward Kit criteria, LLM judges, agent judges, multi-metric reward.json, and reward-details.json."
    acceptance:
      - "verifier/verifier.md is parsed into a VerifierDocument with strategies, rubric files, outputs, and judge roles"
      - "script, llm-judge, and agent-judge strategies selected from verifier/verifier.md execute through Verifier.verify"
      - "llm-judge strategies can declare verifier-local model, input_dir, and context_file overrides"
      - "Reward Kit strategies execute through a verifier-scoped reward.py runner contract"
      - "declared Reward Kit criteria are parsed before launch and govern reward.json metrics"
      - "metrics-only reward.json maps can compute canonical reward through verifier.outputs.aggregate_policy"
      - "reward.json is authoritative when present; reward.txt is scalar compatibility"
      - "Harbor Reward Kit multi-metric maps and reward-details.json are preserved"
      - "agent-judge strategy is verifier-scoped, not a solver scene"
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/verifier.md
    structured_rubric: verifier/rubrics/verifier.toml
    entrypoint: verifier/test.sh
    reward_kit: verifier/reward_kit/
    implementation:
      type: hybrid
      strategies:
        deterministic: verifier/test.sh
        llm_judge: verifier/rubrics/verifier.toml
        rewardkit: verifier/reward_kit/
        agent_judge: verifier/judges/reviewer.md
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
        - name: gold-verifier-package
          expected_reward: 1.0
          artifact: evidence/acceptance/gold-result.json
      judge_agreement:
        required: true
        sample_count: 5
        min_pairwise_agreement: 0.8
    acceptance_live:
      report: evidence/acceptance/live-report.json
      workspace:
        source: current-worktree
        target: /repo
      cases:
        - name: live-reference-verifier
          type: reference
          reruns: 1
          expect:
            reward_min: 0.99
            flake_rate_max: 0.0
    trajectories:
      - path: evidence/acceptance/gold-trajectory.jsonl
        sha256: 9baef13f7fc603cdb7d0c213cc9472754babea6d2651502a97ac36743ad449f6
    artifacts:
      - path: evidence/acceptance/oracle-run.json
        sha256: f548b71f7402c29eb86e22566c8aeaa734aeaa5446ba91fa947389ddb6fc0f14
      - path: evidence/acceptance/gold-result.json
        sha256: 745769fa58d9d0d2b849f1cf2607808ed262c303d05e1cf0139de63a4711125c
      - path: evidence/acceptance/calibration-report.json
        sha256: 269453495de2c5b1e21c07188092c932d7530812ea216f5b71da818b24ee5b02
      - path: evidence/acceptance/verifier-stability-report.json
        sha256: f4887dabf310949cdbeba296ca353b3bac24f24adf75355cc1a337945def34f6
      - path: evidence/acceptance/review.json
        sha256: c250ac0577dcb5550a5089dc52fb515cc834eeda18196d9390907fd60e10973b
---

## prompt

Implement native verifier package support.

Add `VerifierDocument` runtime strategy selection for `verifier/verifier.md`,
with `verifier/task.toml` as an optional compatibility projection. `test.sh`
should be one verifier strategy, not the whole verifier model. Script,
LLM-judge, verifier-scoped agent-judge, and Reward Kit strategies should
execute when selected. Preserve Reward Kit criteria directories, parse declared
criteria before launch, require metrics to match selected criteria when a
Reward Kit strategy declares them, preserve LLM judge configs including
model/input/context overrides, agent-judge roles, multi-metric `reward.json`,
`reward-details.json`, and verifier evidence artifacts.

## role:implementer

Keep the current script verifier behavior working. Add verifier package
selection and reward artifact handling behind explicit selection and
compatibility tests.

## role:judge_designer

Review the verifier isolation model. Judge agents must be hidden from solver
prompts, read only declared inputs, and write structured evidence only under
`/logs/verifier`.
