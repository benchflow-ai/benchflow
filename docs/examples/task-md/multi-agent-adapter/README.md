# Real multi-agent adapter fixture

This directory is a schema-only `task.md` fixture for launching different real agent sessions and preserving isolated per-agent trajectories.

Validate it with `uv run --extra dev bench tasks check docs/examples/task-md/multi-agent-adapter --level schema`.

The fixture is not runnable because it intentionally omits `environment/`, `verifier/`, and workflow sidecars.
