Using the benchflow SDK, run the task at `/app/sample-task` with the `claude-agent-acp` agent and the `claude-haiku-4-5-20251001` model on the `daytona` environment.

Write the result to `/app/output.json` with these fields:
- `reward`: the reward value from result.rewards (e.g., 1.0 or 0.0)
- `tool_calls`: the number of tool calls (result.n_tool_calls)
- `error`: the error string if any, or null
- `agent`: the agent name from result.agent_name

The ANTHROPIC_API_KEY and DAYTONA_API_KEY are already set in the environment.

```toml task
[about]
author_name = "benchflow"
difficulty = "easy"
category = "benchflow-skill-validation"
tags = ["benchflow", "sdk", "skill-test"]

[agent]
timeout = "5m"

[sandbox]
cpus = 1
memory = "4 GB"
disk = "10 GB"
build_timeout = "5m"

[verifier]
timeout = "1m"

[import.harbor]
version = "1.0"

[import.harbor.originals."task.toml"]
original = "sha256:223b20596f43ca0fee13eedb99840c53265e7873f0f99cebb732894ee8c95905"
converted = "sha256:5b6f52b337afd5178affe5ba10a8d92145c8f4100d739cebcf6f20b9952f6fcd"

[import.harbor.originals."instruction.md"]
original = "sha256:d8f526ea526fc8b7c85c2a59e061af375257698df90f72bf1d4341afd4e23a20"
converted = "sha256:d8f526ea526fc8b7c85c2a59e061af375257698df90f72bf1d4341afd4e23a20"
```
