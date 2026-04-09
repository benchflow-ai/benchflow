# benchflow

Multi-turn agent benchmarking with ACP.

## Key files

- `src/benchflow/sdk.py` — SDK.run() orchestrates everything
- `src/benchflow/job.py` — Job with concurrency, retries, resume
- `src/benchflow/agents/registry.py` — agent configs + skill_paths
- `src/benchflow/acp/client.py` — ACP JSON-RPC client
- `src/benchflow/process.py` — DockerProcess, DaytonaProcess

## Architecture

```
SDK.run()          — orchestrates: env → install agent → ACP connect → prompt → verify → cleanup
Job                — wraps SDK.run() with concurrency, retries, resume
registries         — AGENTS (registry.py), PROVIDERS (providers.py) — add dict entries, not code
acp/               — JSON-RPC client + session state + container stdio transport
trajectories/      — capture via ACP native, HTTP proxy, or OTel; ATIF export (backlog)
agents/            — agent configs, provider resolution, openclaw shim, user_agent (backlog)
process.py         — DockerProcess / DaytonaProcess live stdio pipes
metrics.py         — post-hoc aggregation from result.json files
_scoring.py        — shared pure functions: reward extraction, error classification, pass rates
```

## Testing

Requires Python >=3.12. Use `uv` for environment and dependency management:

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/     # unit tests (no Docker needed)
```

Use Haiku 4.5 (`claude-haiku-4-5-20251001`) for smoke tests.

## Test policy

Do not modify or delete existing test files. If a test needs updating because
the code it tests changed shape (e.g., function extracted to new module), that
is acceptable. Write new tests in `tests/test_<module>.py`.
