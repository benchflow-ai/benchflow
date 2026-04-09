# Benchflow — Project Status

## What Works

### Infrastructure
- Harbor superset with ACP — Docker and Daytona environments
- SDK.run(), Job orchestration (concurrency 64, retries, resume)
- CLI: `benchflow run`, `benchflow job`, `benchflow agents`, `benchflow metrics`, `benchflow view`
- YAML job config: `Job.from_yaml()` (benchflow-native + Harbor-compatible)
- Auto-env: API keys inherited from `os.environ`
- Result persistence: result.json, prompts.json, acp_trajectory.jsonl per trial
- Metrics: `collect_metrics()` with pass rates, tool calls, timing, error breakdowns
- Skills: benchflow-run, benchflow-create-task (eval'd at reward 1.0)
- SDK refactored: run() decomposed into 14 private methods, 3 modules extracted (_models, _trajectory, _env_setup)
- Data-driven agent config: env_mapping, credential_files, home_dirs — new agent = registry edit only
- Unit tests: 232 pass (no Docker needed)

### Agents

| Agent | Execution | Trajectory | Skills | Notes |
|-------|-----------|------------|--------|-------|
| claude-agent-acp | Working | Full (ACP native) | ~/.claude/skills/ | Primary agent |
| pi-acp | Working | Full (ACP native) | ~/.claude/skills/ | Outperformed claude on some tasks |
| openclaw | Working | Full (via shim JSONL parsing) | .claude/skills/ → workspace/skills/ | ACP shim wraps `openclaw agent --local` |
| codex-acp | Registered | — | — | Needs OPENAI_API_KEY |
| gemini | Registered | — | — | Needs GOOGLE_API_KEY |

### Benchmark Results

| Benchmark | Model | Score | Reference |
|-----------|-------|-------|-----------|
| TB2 single-turn | Sonnet 4.6 | **58.4%** (52/89) | 59.1% (Anthropic) |
| TB2 multi-turn | Haiku 4.5 | **37.1%** (33/89) | 27.5% (tbench.ai*) |

*Confounding variables: different Claude Code versions + different prompting strategy. See [docs/parity/RESULTS.md](docs/parity/RESULTS.md).

---

## Open Issues

### P1 — Fix Soon
- **Harbor private attributes** — `process.py` accesses `env._sandbox`, `env._strategy`, `env._docker_compose_paths`. No public APIs exist in Harbor. Blocked on upstream.

### P2 — Backlog
- **No integration tests** — SDK.run(), Job.run() have zero end-to-end coverage. SDK internals (_resolve_agent_env, _init_trial, _write_config, _build_result) are unit-tested; async methods need mock env for coverage.
- **Job resume config scoping** — warns on agent mismatch, but other config fields (model, concurrency) still unscoped.
- **YAML config parity with Harbor** — job YAML already covers agent, model, env vars, concurrency, retries, prompts, skills_dir, sandbox_user. The real gap is Harbor *task-level* fields not overridable from job YAML: resource limits (cpus, memory_mb, storage_mb, gpus), timeouts (agent.timeout_sec, build_timeout_sec), and allow_internet. MCP servers and verifier config are inherently per-task, not job-level.

### Benchmarks To Run
- TB2 multi-turn with Sonnet (the parity number that matters)
- SkillsBench full run (87 tasks)
- Multi-agent comparison (20+ tasks, all 3 agents)

### Later
- OpenRouter provider support
- Daytona snapshots (pre-bake agent, eliminate install time)
- Prebuilt SkillsBench images
- ATIF export, MCP pass-through
- E2B/Modal environments
- `benchflow jobs list`, task filtering

---

## Smoke Test Checklist

Future smoke tests must verify:
1. Task execution — reward > 0
2. Trajectory — non-empty `acp_trajectory.jsonl` with tool calls
3. Skills — for SkillsBench tasks, verify agent uses skill content
4. Multi-agent — same tasks on claude-agent-acp, pi-acp, openclaw
5. Errors — 0 infra errors

