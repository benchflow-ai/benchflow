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
- 45 unit tests

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
- **Harbor private attributes** — `process.py` accesses `env._sandbox`, `env._strategy`. Fragile.
- **API keys in `ps aux`** — Docker exec `-e K=V` visible in process list.

### P2 — Backlog
- **No integration tests** — SDK.run(), Job.run() have zero coverage.
- **Job resume no config scoping** — same jobs_dir + different config silently skips tasks.
- **`from harbor import *`** — namespace collision risk (benchflow.Job shadows harbor.Job).

### Fixed (2026-03-24/25)
- ~~**Harbor unpinned**~~ — Pinned to commit `6c2c293`.
- ~~**No timeout on initialize/session_new**~~ — 60s timeout on both.
- **UTF-8 decode crash** in ContainerTransport — now uses `errors="replace"`.
- **Timeout variable scoping** in SDK.run() — defined before try block.
- **Permission handler fragility** — falls back to first option.
- **Job.run() race condition** — now passes job_name to SDK.run().
- **Notification handler crashes** — wrapped in try/except.
- **Viewer JSON parsing** — tolerates corrupted trajectory files.
- **ToolCallStatus enum** — catches invalid status strings.
- **Dead code cleanup** — removed container.py, unused ACPClient fields.
- **#86** — context_root, sandbox_user, pre_agent_hooks, skills_dir Dockerfile injection, DinD path translation.
- **#88** — Pre-create trial dirs to avoid root ownership PermissionError.
- **#89** — Oracle agent support (run solution/solve.sh directly).
- **#90** — Per-phase timing.json, config.json, agent stdout/stderr capture.
- **#91** — Oracle writes agent/oracle.txt inside container (avoids root-ownership).
- **#92** — pre_agent_hooks run before oracle/ACP branch.
- **#94** — Scrape agent-native trajectory as fallback + auto-create tool call records for Gemini.
- **openclaw-gemini** agent variant + GEMINI_API_KEY forwarding.
- **register_agent()** API for custom agents at runtime.
- **Sandbox user** copies .openclaw/ and .gemini/ config dirs.

---

## Roadmap

### Next Up
- ~~**Skills support (SDK-level)**~~ — ✅ Done. `skills_dir` param in SDK.run(), Job, CLI.
- **YAML config parity with Harbor** — agent params, environment config, dataset config
- **Registry architecture** — agent shims as first-class registry entries

### Benchmarks To Run
- TB2 multi-turn with Sonnet (the parity number that matters)
- SkillsBench full run (87 tasks)
- Multi-agent comparison (20+ tasks, all 3 agents)

### Later
- OpenRouter / Vertex AI provider support
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

---

## Key Facts

| Fact | Value |
|------|-------|
| claude-agent-acp | v0.22.2 (Claude Code v2.1.76) |
| Default model | Sonnet 4.6 (set via ACP session/set_model) |
| TB2 tasks | 89 |
| SkillsBench tasks | 87 |
| Max Daytona concurrency tested | 64 |
| Unit tests | 66 |
| Working agents | 3 (claude, pi-acp, openclaw) |
