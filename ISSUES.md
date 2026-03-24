# Benchflow — Open Issues & Backlog

## Active Issues

### P1 — Fix Soon

- **Harbor private attributes** — `process.py:103,182` accesses `env._sandbox`, `env._strategy`. Fragile coupling to Harbor internals. Fix: contribute public API to Harbor.
- **Harbor unpinned** — `pyproject.toml` pins Harbor to git HEAD. Pin to commit hash or tag.
- **API keys in `ps aux`** — Docker exec passes env vars as CLI flags (`-e K=V`). Visible in process list. Fix: use `--env-file`.

### P2 — Backlog

- **No integration tests** — SDK.run(), Job.run() have zero test coverage. Only unit tests for data structures.
- **Job resume no config scoping** — `_get_completed_tasks()` matches by task name across ALL subdirs. Different configs on same jobs_dir silently skip tasks.
- **No timeout on initialize/session_new** — ACP handshake can hang forever if agent process is stuck.
- **`from harbor import *`** — wildcard import creates namespace collision risk. benchflow.Job shadows harbor.Job.

## Completed (this session)

- [x] P0: `RunResult.trial_name` default — prevented crash in asyncio.gather exception handler
- [x] `return_exceptions=True` on `asyncio.gather` — one crash no longer kills entire job
- [x] Per-task `_prune_docker()` race — removed from per-task path
- [x] Silent exception swallowing — added logging
- [x] Openclaw ACP shim — task execution, skill loading, trajectory capture all working
- [x] JSONL parser — handles openclaw's `toolCall` format (not `tool_use`)
- [x] Session handler — supports `text_update` and `agent_thought` (not just chunked)
- [x] Dead code cleanup, type annotation fixes, docstring fixes

## Feature Backlog

### Next Up
- **Skills support (SDK-level)** — detect skills in task env, copy to agent-specific paths. Currently only handled by Dockerfile COPY and openclaw shim.
- **YAML config parity with Harbor** — our configs are simpler than Harbor's. Need: agent params, environment config, dataset config.
- **Registry architecture** — shims should be part of the registry system, not standalone files.

### Later
- **OpenRouter** — route to any model via OPENROUTER_API_KEY
- **Google Vertex AI** — enterprise Gemini/Claude access
- **Daytona snapshots** — pre-bake agent to eliminate install time
- **Prebuilt images for SkillsBench** — speed up from ~90s to ~30s per task
- **ATIF export** — trajectory interop format
- **MCP pass-through** — forward MCP servers to agents
- **E2B/Modal environments** — alternative sandbox backends
- **`benchflow jobs list`** — show past job results
- **Task filtering** — `Job(tasks=[...])` or `task_glob` parameter

## Smoke Test Requirements

Future smoke tests must verify:
1. Task execution — reward > 0
2. Trajectory — non-empty `acp_trajectory.jsonl` with tool calls
3. Skills — for SkillsBench tasks, verify agent references skill content
4. Multi-agent parity — same tasks run on claude-agent-acp, pi-acp, openclaw
5. Error count — 0 infra errors
