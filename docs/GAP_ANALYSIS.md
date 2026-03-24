# Benchflow v2 — Gap Analysis

Findings from dogfood testing on 2026-03-23/24.

## Multi-Agent Testing Results

### Agent Smoke Tests (log-summary-date-ranges)

| Agent | Result | Tools | Notes |
|-------|--------|-------|-------|
| claude-agent-acp | PASS (1.0) | 5-6 | Works reliably |
| pi-acp | PASS (1.0) | 6 | Works after install fix |
| openclaw | PASS (1.0) | 0* | Via ACP shim. *Tool calls internal to openclaw, not visible via ACP. |
| codex-acp | Not tested | — | Needs OPENAI_API_KEY |
| gemini | Not tested | — | Needs GOOGLE_API_KEY |

### TB2-14 Comparison (Haiku 4.5, concurrency 64, Daytona)

| Agent | Passed | Failed | Errored | Score | Time |
|-------|--------|--------|---------|-------|------|
| claude-agent-acp | 4 | 10 | 0 | 28.6% | 12.7min |
| pi-acp | 7 | 5 | 2 | 50.0% | 60.7min |

pi-acp outperformed claude-agent-acp on the same 14-task subset with the same model (Haiku 4.5). pi-acp passed cancel-async-tasks, bn-fit-modify, adaptive-rejection-sampler, and distribution-search where claude failed. pi-acp had 2 timeouts (write-compressor, circuit-fibsqrt) vs claude's 0, but took 5x longer overall.

### SkillsBench Smoke Test (Haiku 4.5, Daytona, claude-agent-acp)

- 2/4 passed (50%), 0 errors (1 of 5 tasks had a broken symlink)
- SkillsBench tasks build from Dockerfile on Daytona (no prebuilt images)
- Build time adds ~30-60s per task but works
- Partial rewards work (dialogue-parser got 0.333)

## Bugs Found & Fixed

### Fixed
1. **Auto-env inheritance** — `SDK.run()` now auto-inherits `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` from `os.environ`. Previously required explicit `agent_env` every time.
2. **pi-acp install** — Install command only installed the ACP wrapper, not the underlying `pi` agent. Added `@mariozechner/pi-coding-agent` to install chain.

### Known Issues
3. **openclaw incompatible** — ACP bridge requires sessions via gateway's `/acp spawn`, not standard ACP `session/new`. Architectural mismatch, not a bug we can fix.
4. **No task filtering on Job** — `Job(tasks_dir=...)` runs ALL tasks. No way to pass a task list or glob filter. Workaround: symlink directory.
5. ~~**Viewer doesn't print URL**~~ — Fixed: viewer prints `Trajectory viewer: http://localhost:{port}`.
6. **hello-world has no prebuilt image** — Running hello-world on Daytona triggers a full Docker build (~5min). Only TB2 tasks have prebuilt images.

## Orchestration Features Tested

| Feature | Status | Notes |
|---------|--------|-------|
| SDK.run() single task | ✓ Works | Tested with 3 agents |
| Job orchestration | ✓ Works | Tested 14+ concurrent tasks |
| Concurrency 64 | ✓ Works | All 14 tasks launched simultaneously on Daytona |
| Retries (max_retries=1) | ✓ Works | Errored tasks retried correctly |
| Resume (re-run same jobs_dir) | ✓ Works | Errored tasks re-run, passed tasks skipped |
| collect_metrics | ✓ Works | Accurate pass/fail/error/tool counts |
| list_agents | ✓ Works | 5 agents with metadata |
| Viewer | ✓ Works | Starts and serves, no crash |
| Result persistence | ✓ Works | result.json, prompts.json, trajectory, verifier outputs |
| ACP trajectory capture | ✓ Works | Real tool calls with content in acp_trajectory.jsonl |
| Model selection (Haiku) | ✓ Works | Confirmed via ACP config/update |
| Daytona environments | ✓ Works | Prebuilt images + Dockerfile builds |
| Multi-agent (same model) | ✓ Works | claude + pi-acp on same tasks, same model |
| SkillsBench tasks | ✓ Works | Dockerfile build on Daytona, partial rewards |

## Gaps Remaining

### Fixed
- **openclaw**: Native ACP bridge incompatible (needs gateway chat sessions). Fixed via ACP shim that wraps `openclaw agent --local` + workspace symlink. Reward 1.0 on log-summary-date-ranges.
- **Task filtering**: Job should accept `tasks: list[str]` or `task_glob: str` parameter

### Should Fix
- **Prebuilt images for SkillsBench**: Would speed up runs from ~90s to ~30s per task
- **summary.json overwrites**: Re-running a job overwrites summary.json but leaves old trial dirs. Should either clean up or append.

### Nice to Have
- **hello-world prebuilt image**: For quick agent smoke tests
- **Agent validation**: `list_agents()` could check which agents are actually installable vs just registered
- **Progress callback**: Job could emit progress events for monitoring

## Dogfood Documents

- `docs/dogfood/DOGFOOD.md` — End-to-end test prompt for agents/humans
- `docs/CLAUDE.md` — Project documentation for AI assistants
- `PLAN.md` — Project roadmap and status
