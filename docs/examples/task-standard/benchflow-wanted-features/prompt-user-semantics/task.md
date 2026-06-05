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
      - role: ux_reviewer
user:
  model: claude-haiku
  stop_rule: satisfied-or-5-rounds
  private_facts:
    hidden_need: "Document user loops must be executable or fail capability validation."
benchflow:
  document_version: "0.3"
  traceability:
    need_ids: [F5, F7]
    user_story: "Multi-scene and simulated-user tasks need prompt guardrails to compose predictably and user/nudge metadata to either execute or fail closed."
    acceptance:
      - "benchflow.prompt.composition append preserves base, role, scene, and turn prompts in order"
      - "replace composition works only when explicitly requested"
      - "document-declared user/nudge loops compile into RolloutConfig or sandbox-aware validation rejects them as metadata-only"
  prompt:
    composition: append
    order: [base, role, scene, turn]
  nudges:
    mode: simulated-user
    branchable: true
    nudge_budget: 5
    confirmation_policy:
      destructive_actions: human
  verifier:
    spec: verifier/verifier.md
    rubric: verifier/rubrics/verifier.md
    entrypoint: verifier/test.sh
---

## prompt

Implement explicit prompt composition and document-declared user semantics.

Today scene prompts can shadow role/base prompts, and `user` /
`## user-persona` is parsed but not compiled into a concrete user loop. Add
the `benchflow.prompt.composition` contract and make document user/nudge
metadata either executable or fail sandbox-aware capability validation.

## role:scene_engineer

Keep prompt composition deterministic and testable. Add regression tests for
append order, explicit replace, dangling role prompts, and scene/turn override
behavior.

## scene:user-loop

Connect `user` and `## user-persona` to the rollout user loop only when the
runtime can honor the declared stop rule and private facts. Otherwise emit a
clear unsupported-feature validation result.

## role:ux_reviewer

Review the generated prompts for accidental guardrail loss and check that
private user facts are never exposed to solver prompts before the simulated user
reveals them.

## user-persona

You are a task author trying to model NudgeBench-style feedback. You reveal
private facts only when the agent asks a targeted clarification question.

