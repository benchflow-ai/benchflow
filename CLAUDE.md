# benchflow

Multi-turn agent benchmarking. Harbor + ACP.

## Key files

- `src/benchflow/sdk.py` — SDK.run() orchestrates everything
- `src/benchflow/job.py` — Job with concurrency, retries, resume
- `src/benchflow/agents/registry.py` — agent configs + skill_paths
- `src/benchflow/acp/client.py` — ACP JSON-RPC client
- `src/benchflow/process.py` — DockerProcess, DaytonaProcess

## Testing

```bash
pytest tests/     # unit tests (no Docker needed)
```

Use Haiku 4.5 (`claude-haiku-4-5-20251001`) for smoke tests.

## Cross-dev with smolclaws

benchflow is a dep of `smolclaws/packages/clawbench/pyproject.toml`. After pushing benchflow changes, update the pinned commit hash there.
