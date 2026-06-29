---
name: benchflow-create-task
description: Create native BenchFlow task.md benchmark tasks. Use when asked to create a new benchmark task, write a verifier, set up a task environment, or define multi-agent roles/scenes/adapters.
---

# BenchFlow Create Task Skill

Use this skill to create benchmark tasks that BenchFlow can run. Prefer the native `task.md` package format. The older split layout (`task.toml` plus `instruction.md` plus `tests/` plus `solution/`) is compatibility input/output, not the recommended authoring surface.

## When to use

Activate this skill when the user asks to:

- create a new benchmark task;
- write a verifier or rubric;
- set up an environment Dockerfile;
- convert an existing problem into a BenchFlow task;
- define multi-agent roles, scenes, simulated-user loops, or external workflow adapters.

## Native task structure

A minimal native package has:

- `task.md` — YAML frontmatter plus the prompt body;
- `environment/Dockerfile` — sandbox setup;
- `verifier/test.sh` or `verifier/verifier.md` — reward contract;
- optional `oracle/solve.sh` — held-out reference behavior;
- optional `prompts/role.<name>.md`, `prompts/scene.<name>.md`, and `prompts/user-persona.md` sidecars for multi-prompt tasks.

## Minimal task.md frontmatter

Use this shape for a simple task:

```yaml
---
schema_version: "1.3"
task:
  name: benchflow/my-task
  description: Short task description
  authors:
    - name: Your Name
agent:
  timeout_sec: 600
verifier:
  timeout_sec: 300
environment:
  cpus: 1
  memory_mb: 4096
---

Write the task prompt here. The prompt body is what the agent sees.
```

## Verifier contract

The verifier must write a scalar reward from `0.0` to `1.0` to `/logs/verifier/reward.txt`. Prefer also writing a structured `/logs/verifier/reward.json` when the task has rubric details, metrics, or evidence.

Verifier guidance:

- deterministic checks should use `verifier/test.sh`;
- richer scoring should declare `verifier/verifier.md` with a selected strategy;
- hidden fixtures and judge prompts belong under `verifier/`, not in the agent prompt;
- nonzero verifier exit without a fresh reward is infrastructure failure, not a task score.

## Multi-agent native roles and scenes

Use root `agents` and `scenes` for stable BenchFlow-native orchestration:

```yaml
agents:
  roles:
    planner:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
    implementer:
      agent: codex-acp
      model: gpt-5.5
    reviewer:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
scenes:
  - name: plan
    turns:
      - role: planner
  - name: implement
    turns:
      - role: implementer
  - name: review
    turns:
      - role: reviewer
```

BenchFlow uses the shared sandbox as the explicit handoff medium. Ask roles to write handoff artifacts under `/app`, such as `/app/plan.md` or `/app/review-feedback.md`.

## External multi-agent workflow adapters

For external frameworks such as LangGraph, AutoGen, CrewAI, Harbor, or a generic OpenAI-compatible workflow, keep the stable role declarations above and put experimental adapter metadata under `benchflow.multi_agent`:

```yaml
benchflow:
  multi_agent:
    adapter: langgraph
    mode: external-workflow
    entrypoint: workflows.design_review:run
    workflow_root: workflow/
    trace:
      llm_proxy: litellm
      capture_raw_llm: required
      capture_framework_events: best-effort
      relationship_graph: required
    agents:
      mapping:
        planner: {role: planner, framework_node: plan_node}
        implementer: {role: implementer, framework_node: implement_node}
        reviewer: {role: reviewer, framework_node: review_node}
```

The adapter should emit:

- `trajectory/llm_raw.jsonl` for LiteLLM proxy request/response records;
- `trajectory/multiagent_events.jsonl` for normalized relationship-aware events;
- `trajectory/agent_graph.json` for agent, team, subgraph, and handoff relationships;
- `trajectory/index.json` for counts, checksums, coverage, and diagnostics.

If raw LLM capture is required but no LiteLLM calls are recorded, the run should fail closed instead of silently reporting a partial trace.

## Prompt sidecars

Use prompt sidecars when roles or scenes need distinct instructions:

- `prompts/role.planner.md` — planner-specific role prompt;
- `prompts/role.implementer.md` — implementation role prompt;
- `prompts/role.reviewer.md` — review role prompt;
- `prompts/scene.review.md` — scene-level review guidance.

Prompt precedence is: inline turn prompt, scene prompt, role prompt, then base prompt.

## Local validation

Use schema validation for authoring-only fixtures and structural validation for runnable tasks.

- Schema-only: `bench tasks check tasks/my-task --level schema`
- Runnable package: `bench tasks check tasks/my-task`
- Publication-grade package: `bench tasks check tasks/my-task --level publication-grade`

## Tips

- Set `agent.timeout_sec` on every published task.
- Keep prompts self-contained and specify exact file paths.
- Put task data in `environment/` and copy it from the Dockerfile.
- Keep verifier code deterministic unless the task intentionally uses an LLM or agent judge.
- Keep adapter-specific fields under `benchflow:` until the task standard promotes them.
- Compare multi-agent workflows against a matched single-agent baseline with the same task set, tool access, answer contract, usage accounting, and trajectory logging coverage.
