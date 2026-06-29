---
schema_version: "1.3"
task:
  name: benchflow/multi-agent-real-agents
  description: Schema-only fixture for real agent sessions with isolated trajectories
agent:
  timeout_sec: 900
verifier:
  timeout_sec: 180
environment:
  cpus: 2
  memory_mb: 4096
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
benchflow:
  multi_agent:
    runtime: real-agent-sessions
    mode: sequential-artifact-handoff
    trace:
      per_agent_trajectories: required
      native_transcripts: best-effort
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
---
# Real multi-agent session fixture

BenchFlow should start different real agents, capture one isolated trajectory per agent session, and add explicit handoff edges between those sessions.
