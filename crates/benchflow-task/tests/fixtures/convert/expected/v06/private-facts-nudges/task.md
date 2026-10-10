Recover the simulated customer's missing order. The final workspace must contain
the recovered order id and a short machine-readable recovery plan.

```role support_solver
Treat the user as the only source of the hidden order id. Ask a targeted
clarifying question before recording any private fact, and keep the final files
limited to the requested support recovery fields.
```

```stage triage
Establish the missing order id from the simulated user and persist it exactly.
```

```stage finish
Summarize the likely failure point and the next action a support operator should
take.
```

```user
You are a frustrated customer on your second support attempt. You know the order
id, but you reveal it only after the agent asks a specific question that makes it
clear why the id is needed.
```

```toml task
name = "benchflow/private-facts-nudges"
description = "Runnable task.md fixture for simulated-user nudges."

[about]
category = "simulated-user"
tags = ["task-md", "simulated-user", "private-facts"]

[agent]
timeout = "10m"

[sandbox]
cpus = 1
memory = "2 GB"
network = "none"
workdir = "/workspace"

[roles.support_solver]

[user]
model = "scripted"

[verifier]
timeout = "2m"

[x-benchflow.prompt]
composition = "append"
order = ["base", "role", "scene", "turn"]

[x-benchflow.nudges]
mode = "simulated-user"
nudge_budget = 3

[import.v06]
schema_version = "1.3"

[import.v06.roles.support_solver]
agent = "claude-agent-acp"
capabilities = ["tool-use", "file-write", "user-dialogue"]

[import.v06.user]
stop_rule = "satisfied-or-3-rounds"
private_facts = { order_id = "BF-1042" }

[[import.v06.interaction.scenes]]
name = "triage"
roles = ["support_solver"]

[[import.v06.interaction.scenes.turns]]
role = "support_solver"
prompt = "Ask the simulated user for the order id when it is missing, then write it to /workspace/order_id.txt."

[[import.v06.interaction.scenes]]
name = "finish"
roles = ["support_solver"]

[[import.v06.interaction.scenes.turns]]
role = "support_solver"
prompt = "Write /workspace/recovery.json with order_id, failure_point, and next_action fields."

[import.v06.verifier_md]
path = "verifier/verifier.md"
default_strategy = "deterministic"
strategies = { deterministic = "script" }
roles = ["reviewer"]
```
