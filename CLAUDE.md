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

## Conventions

Rules a fresh agent won't infer from the code alone:

- **Minimal fix.** Fix what was asked, nothing more. "Leave as is" is a valid
  outcome. No speculative abstractions, no drive-by cleanups, no adding
  docstrings/types/comments to code you didn't touch. Generalize after the
  third repetition, not the first.
- **Registry over hardcode.** Adding an agent or provider = a dict entry in
  `agents/registry.py` or `providers.py`, not a new code path. Narrow
  boundary exceptions exist (e.g. the `oracle` agent is special-cased in
  `sdk.py` because it bypasses the agent loop entirely) — don't add more
  without the same justification.
- **Single source of truth.** Scoring logic lives in `_scoring.py`, trajectory
  parsing in `_trajectory.py`, env setup in `_env_setup.py`. Don't duplicate;
  import.
- **Greppable modules, small private methods.** `SDK.run()` is the
  orchestrator; the private methods it calls are each 10–80 lines. Match
  that shape when extending — add a new private method, don't inline more
  logic into `run()`.
- **Don't modify or delete existing tests.** Updating a test because the code
  it tests changed shape is fine. Writing new tests in
  `tests/test_<module>.py` is fine. Rewriting a passing test to match new
  behavior without understanding why it was written is not.
- **No tautological tests.** Don't test `@dataclass` field reads, stdlib
  behavior, helper self-consistency, or "does it construct." Tests should
  exercise real code paths.
- **Don't auto-commit.** Stop after each logical unit and wait for explicit
  approval. Don't chain commits across unrelated fixes.

## Testing

Requires Python >=3.12. Use `uv` for environment and dependency management:

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/pre-commit install                           # one-time per clone — runs ruff on commit
.venv/bin/python -m pytest tests/                      # unit tests (no Docker needed)
.venv/bin/python -m pytest -m live tests/test_smoke.py # e2e smoke (real Docker + API)
```

Pre-commit hook runs `ruff format` + `ruff check` on staged files in `src/`
and `tests/`, matching what CI gates. After cloning, run `pre-commit install`
once. Bypassing with `--no-verify` defeats the local guard but CI still
enforces the same checks — fix the underlying issue rather than skipping.

Before pushing, run the full local gate: `ruff format`, `ruff check`,
`pytest`, and `ty check src/`. CI enforces all four — running pytest alone
has missed lint regressions before.

### `ty` as a navigation tool

`ty` (installed via `[dev]`) is both a type checker and the fastest way to
discover blast radius. After any signature change, run:

```bash
.venv/bin/ty check src/
```

The error list *is* the caller list — but only because the tree is clean
today (zero pre-existing errors). Treat any new `ty` error as signal, not
noise, and keep the baseline at zero. Cheaper than grepping for a symbol
that flows through registries or dynamic dispatch; use it before
hand-rolling a find-references search.

Use Haiku 4.5 (`claude-haiku-4-5-20251001`) for smoke tests.

The `live` marker is excluded by default (`addopts = "-m 'not live'"`). Live
tests require docker daemon + (`ANTHROPIC_API_KEY` or `~/.claude/.credentials.json`)
and skip otherwise via the `smoke_prereqs` fixture. Inside DinD devcontainers
the test uses a workspace-rooted `jobs_dir` (see `smoke_jobs_dir` fixture)
because pytest's `tmp_path` lives on the container overlay and can't be
bind-mounted by the host docker daemon. To add a new live test: opt into
`@pytest.mark.live`, depend on `smoke_prereqs` and `smoke_jobs_dir`, never
call `resolve_agent_env` from the skip path (it's part of the system under
test).

## Test policy

Do not modify or delete existing test files. If a test needs updating because
the code it tests changed shape (e.g., function extracted to new module), that
is acceptable. Write new tests in `tests/test_<module>.py`.

## Known issues

### P1 — Fix Soon
- **Harbor private attributes** — `process.py` accesses `env._sandbox`, `env._strategy`, `env._docker_compose_paths`. No public APIs exist in Harbor. Blocked on upstream.

### P2 — Backlog
- **e2e coverage partial** — `tests/test_smoke.py::test_hello_world_smoke` exercises `SDK.run()` end-to-end against claude-agent-acp + Haiku 4.5 (live-marker, local/manual). Still open: multi-agent (codex/pi/openclaw/gemini), SkillsBench tasks, CI wiring.
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
