---
schema_version: "1.3"
task:
  name: benchflow/private-facts-nudges
  description: Runnable task.md fixture for simulated-user nudges.
metadata:
  category: simulated-user
  tags: [task-md, simulated-user, private-facts]
agent:
  timeout_sec: 600
verifier:
  timeout_sec: 120
environment:
  network_mode: no-network
  cpus: 1
  memory_mb: 2048
  workdir: /workspace
agents:
  roles:
    support_solver:
      agent: claude-agent-acp
      capabilities: [tool-use, file-write, user-dialogue]
scenes:
  - name: triage
    roles: [support_solver]
    turns:
      - role: support_solver
        prompt: Ask the simulated user for the order id when it is missing, then write it to /workspace/order_id.txt.
  - name: finish
    roles: [support_solver]
    turns:
      - role: support_solver
        prompt: Write /workspace/recovery.json with order_id, failure_point, and next_action fields.
user:
  model: scripted
  stop_rule: satisfied-or-3-rounds
  private_facts:
    order_id: BF-1042
benchflow:
  prompt:
    composition: append
    order: [base, role, scene, turn]
  nudges:
    mode: simulated-user
    nudge_budget: 3
---
# Private Facts Nudge Fixture

## prompt

Recover the simulated customer's missing order. The final workspace must contain
the recovered order id and a short machine-readable recovery plan.

## role:support_solver

Treat the user as the only source of the hidden order id. Ask a targeted
clarifying question before recording any private fact, and keep the final files
limited to the requested support recovery fields.

## scene:triage

Establish the missing order id from the simulated user and persist it exactly.

## scene:finish

Summarize the likely failure point and the next action a support operator should
take.

## user-persona

You are a frustrated customer on your second support attempt. You know the order
id, but you reveal it only after the agent asks a specific question that makes it
clear why the id is needed.
