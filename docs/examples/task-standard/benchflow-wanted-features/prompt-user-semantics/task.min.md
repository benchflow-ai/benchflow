---
profile: code-change
source: benchflow/wanted-features/prompt-user-semantics
name: benchflow-wanted/prompt-user-semantics
image: ghcr.io/astral-sh/uv:python3.12-bookworm
verifier: verifier/
oracle: oracle/
task:
  description: Implement explicit prompt composition plus document-declared simulated user/nudge semantics.
  authors:
    - name: BenchFlow
      email: benchflow@example.com
  keywords: [task-standard, scenes, user, nudges]
metadata:
  feature_area: interaction
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
  teams:
    standard_review:
      handoff:
        mode: sequential
        workspace_visibility: shared
        trajectory_visibility: metadata
  nudges:
    mode: simulated-user
    branchable: true
    branch_execution: option-kinds-preserved
    nudge_budget: 5
    confirmation_policy:
      destructive_actions: human
---

## prompt

Implement explicit prompt composition and document-declared user semantics.

Today scene prompts can shadow role/base prompts, and the deterministic
scripted plus bounded model-linear private-fact subsets of `user` metadata
compile into a concrete linear user loop across document scenes.

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
