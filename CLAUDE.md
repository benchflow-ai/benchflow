# benchflow

Multi-turn agent benchmarking with ACP.

## Key files

- `src/benchflow/sdk.py` — SDK.run() orchestrates via private methods
- `src/benchflow/models.py` — RunResult, AgentInstallError, AgentTimeoutError, TrajectorySource
- `src/benchflow/_trajectory.py` — trajectory capture and parsing
- `src/benchflow/_env_setup.py` — Dockerfile staging, skills injection, DinD patching
- `src/benchflow/job.py` — Job with concurrency, retries, resume
- `src/benchflow/agents/registry.py` — AgentConfig (env_mapping, credential_files, home_dirs)
- `src/benchflow/acp/client.py` — ACP JSON-RPC client
- `src/benchflow/process.py` — DockerProcess, DaytonaProcess
- `src/benchflow/_scoring.py` — reward extraction, error classification, pass rates
- `src/benchflow/metrics.py` — post-hoc aggregation from result.json files
- `src/benchflow/environments.py` — ServiceConfig, Dockerfile service detection
- `src/benchflow/skills.py` — skill install and discovery
- `src/benchflow/tasks.py` — task validation and init

## Architecture

```
SDK.run()          — orchestrates via private methods (each 10–80 lines)
  models.py        — RunResult, AgentInstallError, AgentTimeoutError, TrajectorySource (public, re-exported from benchflow)
  _trajectory.py   — ACP native, agent-scraped, Gemini trajectory capture
  _env_setup.py    — Dockerfile dep staging, skills injection, DinD detection/patching
Job                — wraps SDK.run() with concurrency, retries, resume
registries         — AGENTS (registry.py), PROVIDERS (providers.py) — add dict entries, not code
  AgentConfig      — env_mapping, credential_files, home_dirs, skill_paths
  ProviderConfig   — credential_files (e.g. Vertex ADC)
acp/               — JSON-RPC client + session state + container stdio transport
trajectories/      — capture via ACP native, HTTP proxy, or OTel; ATIF model implemented, not wired
agents/            — agent configs, provider resolution, openclaw shim, user_agent implemented, not wired
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

## Known issues

### P1 — Fix Soon
- **Harbor private attributes** — `process.py` accesses `env._sandbox`, `env._strategy`, `env._docker_compose_paths`. No public APIs exist in Harbor. Blocked on upstream.

### P2 — Backlog
- **No e2e integration tests** — SDK.run(), Job.run() have no end-to-end coverage with real environments. Job._run_task_loop has mocked async tests.
- **Job resume config scoping** — warns on agent mismatch, but other config fields (model, concurrency) still unscoped.
- **YAML config parity with Harbor** — job YAML covers agent, model, env vars, concurrency, retries, prompts, skills_dir, sandbox_user. Gap: Harbor task-level fields not overridable from job YAML (resource limits, timeouts, allow_internet).

### Later
- OpenRouter provider support
- Daytona snapshots (pre-bake agent, eliminate install time)
- Prebuilt SkillsBench images
- Wire ATIF export and user_agent into SDK/CLI
- MCP pass-through
- E2B/Modal environments
- `benchflow jobs list`, task filtering

## Smoke test checklist

Future smoke tests must verify:
1. Task execution — reward > 0
2. Trajectory — non-empty `acp_trajectory.jsonl` with tool calls
3. Skills — for SkillsBench tasks, verify agent uses skill content
4. Multi-agent — same tasks on claude-agent-acp, pi-acp, openclaw
5. Errors — 0 infra errors
