# Benchflow v2 — Plan

## Completed

### Infrastructure (pushed to `main`)
- Harbor superset: import harbor as dependency, re-export everything
- ACP client: initialize, session/new, session/prompt, session/config/update, permission auto-approve
- Container transport: live stdio pipe via Docker compose exec or Daytona SSH
- SDK.run(): Harbor env (Docker or Daytona) + ACP agent + Harbor verifier
- Multi-turn: multiple prompts to same ACP session
- Multi-agent registry: claude-agent-acp, pi-acp, openclaw, codex-acp, gemini
- Model config: set via ACP `session/set_model` (env var ignored by claude-agent-acp)
- Result persistence: result.json, prompts.json, acp_trajectory.jsonl per trial
- Viewer: `benchflow view` renders HTML trajectory viewer
- CLI: `benchflow run`, `benchflow view`
- Trajectory capture: ACP native
- Daytona environment support (DaytonaProcess via SSH, LiveProcess abstraction)
- Job orchestration: concurrency, retries (`RetryConfig`), resume, `summary.json`
- Metrics: `collect_metrics()` — pass rates, tool calls, timing, error breakdowns
- Auto-env: SDK auto-inherits API keys from `os.environ`
- Bug fixes: pipefail, DEBIAN_FRONTEND, node version check, dynamic WORKDIR, 10MB buffer, token limit, stderr capture

### TB2 Single-Turn (Step 1) — Done
- **52/89 (58.4%)** with Sonnet 4.6 via claude-agent-acp (Claude Code v2.1.76)
- Parity: official 59.1%, tbench.ai 59.55% — **within ~1%, pipeline validated**
- 14 errors: 9 timeouts, 5 Daytona install (fixed with DEBIAN_FRONTEND)

### TB2 Multi-Turn (Step 2) — Done
- **33/89 (37.1%)** with Haiku 4.5, multi-turn recheck prompt
- Reference: tbench.ai Haiku 4.5 = 27.5% (Claude Code v2.0.31) — our newer v2.1.76 is ~10pp better
- 9 errors: all timeouts, 0 install failures
- Prompts: `[instruction, "Review your solution. Check for errors, test it, and fix any issues."]`

### Multi-Agent Testing — Done
- **claude-agent-acp**: TB2-14 = 4/14 (28.6%), 0 errors, 12.7min
- **pi-acp**: TB2-14 = 7/14 (50.0%), 2 timeouts, 60.7min — outperformed claude on same tasks/model
- **openclaw**: Working via ACP shim — reward 1.0 on log-summary-date-ranges (Haiku 4.5)
  - openclaw's native ACP bridge is incompatible (needs gateway chat sessions)
  - Solution: ACP shim wraps `openclaw agent --local` + workspace symlink to task dir
  - Tool calls happen internally (not visible via ACP updates), but tasks are solved
- **codex-acp**: Not tested (needs OPENAI_API_KEY)
- **gemini**: Not tested (needs GOOGLE_API_KEY)

### SkillsBench Smoke Test — Done
- 2/4 passed (50%) with Haiku 4.5 on Daytona
- Skills auto-loaded: Dockerfiles copy skills to `~/.claude/skills/`, Claude Code discovers them
- Partial rewards work (dialogue-parser got 0.333)
- Dockerfile builds on Daytona work (~30-60s extra per task, no prebuilt images)

### Dogfood — Done
- All SDK features tested end-to-end: SDK.run(), Job, collect_metrics, list_agents, viewer
- Concurrency 64 validated on Daytona
- Docs: `docs/dogfood/DOGFOOD.md`, `docs/GAP_ANALYSIS.md`

### Skills & Skill Validation — Done
- Created `skills/benchflow-run/` — skill for running benchmarks (SDK, Job, metrics)
- Created `skills/benchflow-create-task/` — skill for creating Harbor-format tasks
- Validation tasks eval'd through benchflow (Haiku 4.5, Daytona):
  - `create-simple-task`: reward 1.0 — agent read skill, created valid task with all files
  - `benchflow-knowledge`: reward 1.0 (6/6) — agent read skills, answered all questions correctly
- Skills baked into Dockerfiles via `COPY skills /root/.claude/skills/`, auto-discovered by Claude Code

### Code Review Fixes — Done
- `asyncio.gather` with `return_exceptions=True` — one crash no longer kills entire job
- Removed per-task `_prune_docker()` — was racing with container setup at concurrency > 1
- Added logging to silent exception blocks in job.py and metrics.py
- Simplified job counting — normalize all results to dicts, eliminated dead RunResult branches
- YAML job config: `Job.from_yaml()` supports benchflow-native and Harbor-compatible formats
- CLI: `benchflow job --config/-f` for YAML configs
- 45 unit tests (23 new): metrics, job counting, YAML parsing

### Smoke Tests — Done
Post-fix validation (Haiku 4.5, Daytona, concurrency 64):

| Run | Passed | Total | Score | Errors |
|-----|--------|-------|-------|--------|
| TB2 single-turn | 2 | 5 | 40% | 0 |
| TB2 multi-turn | 2+ | 5 | 40%+ | 0 |
| SkillsBench | 2 | 4 | 50% | 0 |

All 0 infra errors, trajectories have real tool calls, scores match expected range for Haiku.

---

## Next Steps

### Step 3: SkillsBench Full Run
- [ ] Run all 87 tasks with claude-agent-acp + Haiku 4.5 on Daytona
- [ ] Run 20-task subset with pi-acp for comparison
- [ ] Compare with reference trajectories / official scores
- Smoke test done, skills loading confirmed, Dockerfile builds work

### Step 4: TB2 Multi-Turn with Sonnet
- [ ] Run all 89 tasks with Sonnet 4.6 + recheck prompt
- [ ] Compare: single-turn Sonnet (58.4%) vs multi-turn Sonnet (?)
- [ ] This is the number that matters for parity — Haiku multi-turn was just validation

### Step 5: Parity Report
- [ ] Update `docs/parity/RESULTS.md` with:
  - SkillsBench results
  - Multi-turn Sonnet results
  - Multi-agent comparison table
  - Full error analysis across all runs
- [ ] Agent/model version matrix

### CLI & SDK Polish
- [x] `benchflow agents` — list registered agents
- [x] `benchflow job` / `benchflow metrics` — CLI commands
- [x] Viewer: prints URL when serving
- [x] YAML job config: `Job.from_yaml()`, `benchflow job --config`
- [ ] `benchflow jobs list` — show past job results
- [ ] Task filtering: `Job(tasks=[...])` or `task_glob` parameter

### Skills Support (next feature)
- [ ] SDK-level skills loading — detect skills in task env, copy to agent-specific paths
- [ ] Skills finding — discover skills from registry, symlinks, or task config
- [ ] Skills for all agents: claude (~/.claude/skills/), openclaw (~/.openclaw/workspace/.claude/skills/), codex (~/.codex/skills/), gemini (~/.gemini/skills/)
- Harbor's approach: `skills_dir` in EnvironmentConfig, each agent copies to its own config dir

### Multi-Agent Expansion
- [ ] codex-acp: test with OPENAI_API_KEY
- [ ] gemini: test with GOOGLE_API_KEY
- [x] openclaw: working via ACP shim with trajectory parsing

### LLM Provider Support
- [ ] OpenRouter — route to any model via OPENROUTER_API_KEY
- [ ] Google Vertex AI — enterprise Gemini/Claude access

### Future Infrastructure
- [ ] Daytona snapshots — pre-bake agent to eliminate install time
- [ ] Prebuilt images for SkillsBench — speed up from ~90s to ~30s per task
- [ ] ATIF export for trajectory interop
- [ ] MCP pass-through to agents
- [ ] E2B/Modal environment backends

---

## Key Facts

| Fact | Value |
|------|-------|
| claude-agent-acp version | 0.22.2 |
| Embedded Claude Code | v2.1.76 |
| Claude Agent SDK | v0.2.76 |
| Default model | Sonnet 4.6 (NOT Haiku) |
| ACP protocol SDK | 0.16.1 |
| TB2 task count | 89 |
| SkillsBench task count | 87 |
| Our TB2 score (Sonnet 4.6, single-turn) | 52/89 = 58.4% |
| Our TB2 score (Haiku 4.5, multi-turn) | 33/89 = 37.1% |
| Official TB2 (Sonnet 4.6) | 59.1% |
| Official TB2 (Haiku 4.5, old CC) | 27.5% |
| Daytona max concurrency tested | 64 |
