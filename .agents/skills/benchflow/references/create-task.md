---
name: benchflow-create-task
description: Create native BenchFlow task.md benchmark tasks, including real multi-agent roles and handoff tracking.
---

# BenchFlow Create Task Skill

Prefer native `task.md` packages. The legacy split layout is compatibility input/output only.

A native package usually contains `task.md`, `environment/Dockerfile`, `verifier/test.sh` or `verifier/verifier.md`, and optionally `oracle/solve.sh` plus prompt sidecars under `prompts/`.

## Minimal task.md

```yaml
---
schema_version: "1.3"
task:
  name: benchflow/my-task
agent:
  timeout_sec: 600
verifier:
  timeout_sec: 300
environment:
  cpus: 1
  memory_mb: 4096
---

Write the task prompt here.
```

## Real multi-agent tasks

Use `agents` and `scenes` to start real agent sessions. These are not LangGraph nodes or plain model calls. Each role should map to an actual BenchFlow agent runtime.

```yaml
agents:
  roles:
    planner:
      agent: claude-agent-acp
    implementer:
      agent: codex-acp
    reviewer:
      agent: gemini
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

BenchFlow should give each role a fresh session and capture an isolated trajectory. Handoffs should be explicit artifacts or messages.

```yaml
benchflow:
  multi_agent:
    runtime: real-agent-sessions
    mode: sequential-artifact-handoff
    trace:
      per_agent_trajectories: required
      native_transcripts: best-effort
      workspace_diffs: required
      raw_llm_proxy: optional
    handoffs:
      - id: plan-to-implementer
        from: planner
        to: implementer
        artifacts: [/app/plan.md]
      - id: implementation-to-reviewer
        from: implementer
        to: reviewer
        artifacts: [/app]
```

Expected artifacts are `trajectory/sessions.jsonl`, `trajectory/handoffs.jsonl`, `trajectory/multiagent_events.jsonl`, `trajectory/agent_graph.json`, and per-agent session files under `trajectory/agents/`.

LiteLLM capture is optional evidence. It must not replace the real agent transcript unless a task explicitly asks for provider-level evidence.

## Validation

- Schema-only: `bench tasks check tasks/my-task --level schema`
- Runnable package: `bench tasks check tasks/my-task`
- Publication-grade package: `bench tasks check tasks/my-task --level publication-grade`
