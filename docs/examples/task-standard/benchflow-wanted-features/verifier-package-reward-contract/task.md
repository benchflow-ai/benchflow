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
        rewardkit: verifier/reward_kit/
        agent_judge: verifier/judges/reviewer.md
      outputs:
        reward_json: /logs/verifier/reward.json
        reward_details: /logs/verifier/reward-details.json
  evidence:
    calibration:
      oracle_reward: 1.0
      no_op_reward_max: 0.0
      known_bad_reward_max: 0.2
      judge_agreement:
        required: true
        sample_count: 5
        min_pairwise_agreement: 0.8
---

## prompt

Implement native verifier package support.

Add a `VerifierDocument` parser for `verifier/verifier.md`, with
`verifier/task.toml` as an optional compatibility projection. `test.sh` should
be one verifier strategy, not the whole verifier model. Preserve Reward Kit
criteria directories, LLM judge configs, agent-judge roles, multi-metric
`reward.json`, `reward-details.json`, and verifier evidence artifacts.

## role:implementer

Keep the current script verifier behavior working. Add the new package parser
and reward artifact handling behind explicit selection and compatibility tests.

## role:judge_designer

Review the verifier isolation model. Judge agents must be hidden from solver
prompts, read only declared inputs, and write structured evidence only under
`/logs/verifier`.

