---
schema_version: "1.3"
metadata:
  author_name: benchflow
  difficulty: medium
  category: multi-scene
  tags: [task-md, scenes]
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
      capabilities: [tool-use]
    executor:
      agent: claude-agent-acp
      capabilities: [tool-use, code-edit]
    reviewer:
      agent: claude-agent-acp
      capabilities: [tool-use, review]
scenes:
  - name: plan
    turns:
      - role: planner
  - name: implement
    turns:
      - role: executor
  - name: review
    turns:
      - role: reviewer
---
# Multi-Scene Refactor

## prompt

Refactor the tiny service so it keeps the same public behavior while splitting
request parsing, business logic, and output formatting into separate modules.

## scene:plan

Read the task, inspect the code, and write a concise implementation plan.

## scene:implement

Apply the plan. Keep the patch small and run the verifier before finishing.

## scene:review

Review the final diff for behavior drift, missing tests, and unnecessary churn.
