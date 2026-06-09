---
schema_version: "1.3"
source: benchflow/wanted-features/prompt-user-semantics
task:
  name: benchflow-wanted/prompt-user-semantics
  description: Implement explicit prompt composition plus document-declared simulated user/nudge semantics.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, scenes, user, nudges]
metadata:
  category: benchflow-feature
  feature_area: interaction
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
    scene_engineer:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: high
      capabilities: [code-edit, tests]
    ux_reviewer:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
      capabilities: [review]
scenes:
  - name: prompt-composition
    turns:
      - role: scene_engineer
  - name: user-loop
    turns:
      - role: scene_engineer
  - name: team-handoff
    turns:
      - role: scene_engineer
        prompt: Summarize how the document user runtime preserves private facts and records role handoff metadata.
      - role: ux_reviewer
        prompt: Review the handoff transcript shape for privacy leaks, missing role metadata, and unsupported team semantics.
user:
  model: claude-haiku
  stop_rule: satisfied-or-5-rounds
  private_facts:
    hidden_need: "Document user loops must be executable or fail capability validation."
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F5, F7]
    user_story: "Multi-scene and simulated-user tasks need prompt guardrails to compose predictably; linear document-declared user/nudge loops must execute, human confirmation policy must fail closed without a handler, and branchable ask_user choices must preserve option kinds for future branch execution."
    acceptance:
      - "benchflow.prompt.composition append preserves base, role, scene, and turn prompts in order"
      - "replace composition works only when explicitly requested"
      - "document-declared linear multi-scene user/nudge loops compile into RolloutConfig and pass sandbox-aware validation"
      - "sequential shared-workspace team handoff executes explicit multi-role scene turns and records handoff metadata"
      - "human confirmation policy installs a fail-closed permission handler unless a caller supplies an explicit on_ask_user handler"
      - "branchable ask_user requests preserve option IDs and option kinds so reject/allow semantics are explicit"
      - "full branch execution, interactive approval UI, parallel teams, and rich handoff artifacts remain explicit future work"
  teams:
    standard_review:
      handoff:
        mode: sequential
        workspace_visibility: shared
        trajectory_visibility: metadata
  prompt:
    composition: append
    order: [base, role, scene, turn]
  nudges:
    mode: simulated-user
    branchable: true
    branch_execution: option-kinds-preserved
    nudge_budget: 5
    confirmation_policy:
      destructive_actions: human
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/verifier.md
    entrypoint: verifier/test.sh
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
        - name: gold-user-semantics
          expected_reward: 1.0
          artifact: evidence/acceptance/gold-result.json
    trajectories:
      - path: evidence/acceptance/gold-trajectory.jsonl
        sha256: 22b44de50bdf345ec09f23f589264c3d1cd8b4bf8bda486783076009c477f11e
    artifacts:
      - path: evidence/acceptance/oracle-run.json
        sha256: 40a3368b55dba88acebb0ebc9721c656fb4219802e16b079ca8fcad06fc972c3
      - path: evidence/acceptance/gold-result.json
        sha256: ddc1ffa2ff43e5a3d73db452965a854ff84cadfd7ce553511ac45bcbf15086ad
      - path: evidence/acceptance/calibration-report.json
        sha256: c19c8569dee89ced0be3e342efb686c5df8add60c44c12022df739287565a058
      - path: evidence/acceptance/verifier-stability-report.json
        sha256: 0a5a079180a06b18a25da205144a228f413dc27aed9f6200d034869bd9e882d6
      - path: evidence/acceptance/review.json
        sha256: 8be8caf6c4ea0dc12641f7253fb8844dc78fd818c868ede44c7f47fa5d7de96d
---

## prompt

Implement explicit prompt composition and document-declared user semantics.

Today scene prompts can shadow role/base prompts, and the deterministic
scripted plus bounded model-linear private-fact subsets of `user` metadata
compile into a concrete linear user loop across document scenes. Add the
`benchflow.prompt.composition` contract and make supported document user/nudge
metadata executable while branchable `ask_user` option kinds are preserved for
policy/branch scoring. The first team handoff slice should execute explicit
multi-role scene turns sequentially in a shared workspace and record handoff
metadata, while full branch execution, interactive approval UI, parallel teams,
and rich handoff artifacts fail closed or remain explicit future work.

## role:scene_engineer

Keep prompt composition deterministic and testable. Add regression tests for
append order, explicit replace, dangling role prompts, and scene/turn override
behavior.

## scene:user-loop

Connect `user` and `## user-persona` to the rollout user loop only when the
runtime can honor the declared stop rule, private facts, and linear scene
sequence. Otherwise emit a clear unsupported-feature validation result.

## scene:team-handoff

Execute explicit multi-role scene turns only for the supported sequential
shared-workspace team handoff subset. Persist enough metadata for audit without
sharing hidden full trajectories across roles.

## role:ux_reviewer

Review the generated prompts for accidental guardrail loss and check that
private user facts are never exposed to solver prompts before the simulated user
reveals them.

## user-persona

You are a task author trying to model NudgeBench-style feedback. You reveal
private facts only when the agent asks a targeted clarification question.
