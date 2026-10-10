Refactor the tiny service so it keeps the same public behavior while splitting
request parsing, business logic, and output formatting into separate modules.

```stage plan
Read the task, inspect the code, and write a concise implementation plan.
```

```stage implement
Apply the plan. Keep the patch small and run the verifier before finishing.
```

```stage review
Review the final diff for behavior drift, missing tests, and unnecessary churn.
```

```toml task
[about]
author_name = "benchflow"
difficulty = "medium"
category = "multi-scene"
tags = ["task-md", "scenes"]

[agent]
timeout = "15m"

[sandbox]
cpus = 2
memory = "4 GB"

[roles.planner]

[roles.executor]

[roles.reviewer]

[verifier]
timeout = "3m"

[import.v06]
schema_version = "1.3"

[import.v06.roles.planner]
agent = "claude-agent-acp"
capabilities = ["tool-use"]

[import.v06.roles.executor]
agent = "claude-agent-acp"
capabilities = ["tool-use", "code-edit"]

[import.v06.roles.reviewer]
agent = "claude-agent-acp"
capabilities = ["tool-use", "review"]

[[import.v06.interaction.scenes]]
name = "plan"

[[import.v06.interaction.scenes.turns]]
role = "planner"

[[import.v06.interaction.scenes]]
name = "implement"

[[import.v06.interaction.scenes.turns]]
role = "executor"

[[import.v06.interaction.scenes]]
name = "review"

[[import.v06.interaction.scenes.turns]]
role = "reviewer"
```
