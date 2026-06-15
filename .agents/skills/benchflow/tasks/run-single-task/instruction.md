Using the benchflow SDK, run the task at `/app/sample-task` with the `claude-agent-acp` agent and the `claude-haiku-4-5-20251001` model on the `daytona` environment.

Write the result to `/app/output.json` with these fields:
- `reward`: the reward value from result.rewards (e.g., 1.0 or 0.0)
- `tool_calls`: the number of tool calls (result.n_tool_calls)
- `error`: the error string if any, or null
- `agent`: the agent name from result.agent_name

The ANTHROPIC_API_KEY and DAYTONA_API_KEY are already set in the environment.
