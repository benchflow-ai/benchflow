# Multi-agent adapter fixture

This directory is a schema-only `task.md` fixture for the proposed uniform multi-agent adapter surface.

Validate it with `uv run --extra dev bench tasks check docs/examples/task-md/multi-agent-adapter --level schema`.

The fixture is not a runnable eval task because it intentionally omits `environment/`, `verifier/`, and workflow implementation sidecars.
