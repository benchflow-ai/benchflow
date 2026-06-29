# Real multi-agent adapter fixture

This directory is a schema-only `task.md` fixture for launching different real agent sessions and preserving isolated per-agent trajectories.

The implementation slice in this PR writes real-agent trace artifacts for native Scene / Role / Turn execution: `sessions.jsonl`, `handoffs.jsonl`, `multiagent_events.jsonl`, `agent_graph.json`, and per-session `trajectory/agents/<role>/<session>/acp.jsonl` files.

Validate this fixture with `uv run --extra dev bench tasks check docs/examples/task-md/multi-agent-adapter --level schema`.

The fixture is not runnable because it intentionally omits `environment/`, `verifier/`, and workflow sidecars.
