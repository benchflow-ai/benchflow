Help a simulated customer recover a missing order. The final answer must include
the order id, the likely failure point, and the exact next action.

```role concierge
Interview the customer. Ask for missing details when the request is ambiguous,
but do not expose internal implementation details.
```

```role operator
Inspect the available order data and prepare the final recovery action.
```

```user
You are impatient because this is the second support attempt. You remember the
item and delivery window, but you only reveal the order id if the agent asks a
specific question that makes it clear why the id is needed.
```

```toml task
[about]
author_name = "benchflow"
difficulty = "hard"
category = "simulated-user"
tags = ["task-md", "nudgebench", "agent-team"]

[agent]
timeout = "20m"

[sandbox]
cpus = 2
memory = "4 GB"

[roles.concierge]

[roles.operator]

[user]
model = "scripted"

[verifier]
timeout = "4m"

[import.v06]
schema_version = "1.3"
user = { stop_rule = "satisfied-or-5-rounds", tools = ["ask-clarifying-question"] }

[import.v06.roles.concierge]
agent = "claude-agent-acp"
capabilities = ["tool-use", "user-dialogue"]

[import.v06.roles.operator]
agent = "claude-agent-acp"
capabilities = ["tool-use", "code-edit"]

[[import.v06.interaction.scenes]]
name = "intake"

[[import.v06.interaction.scenes.turns]]
role = "concierge"

[[import.v06.interaction.scenes]]
name = "fulfill"

[[import.v06.interaction.scenes.turns]]
role = "operator"
```
