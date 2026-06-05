---
version: "1.0"
metadata:
  author_name: benchflow
  difficulty: hard
  category: simulated-user
  tags: [task-md, nudgebench, agent-team]
agent:
  timeout_sec: 1200
verifier:
  timeout_sec: 240
environment:
  cpus: 2
  memory_mb: 4096
agents:
  roles:
    concierge:
      agent: claude-agent-acp
      model: claude-sonnet-4-6
      capabilities: [tool-use, user-dialogue]
    operator:
      agent: codex-acp
      model: gpt-5.5
      reasoning_effort: high
      capabilities: [tool-use, code-edit]
scenes:
  - name: intake
    turns:
      - role: concierge
  - name: fulfill
    turns:
      - role: operator
user:
  model: claude-haiku
  stop_rule: satisfied-or-5-rounds
  tools: [ask-clarifying-question]
---
# NudgeBench Order Recovery

## prompt

Help a simulated customer recover a missing order. The final answer must include
the order id, the likely failure point, and the exact next action.

## role:concierge

Interview the customer. Ask for missing details when the request is ambiguous,
but do not expose internal implementation details.

## role:operator

Inspect the available order data and prepare the final recovery action.

## user-persona

You are impatient because this is the second support attempt. You remember the
item and delivery window, but you only reveal the order id if the agent asks a
specific question that makes it clear why the id is needed.
