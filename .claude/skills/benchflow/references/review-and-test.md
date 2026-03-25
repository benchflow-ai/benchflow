# Review, Fix, and Test — benchflow Development Workflow

This reference describes the workflow for code review, bug fixing, and validation
of the benchflow codebase. Use when asked to review benchflow, fix bugs, or run
quality checks.

## 1. Council Review

Launch 4 independent subagents in parallel to review different subsystems:

### Agent 1: SDK & Job Core
- `src/benchflow/sdk.py` — SDK.run() orchestration
- `src/benchflow/job.py` — Job concurrency, retries, resume
- `src/benchflow/metrics.py` — collect_metrics()
- Focus: race conditions, error handling, edge cases

### Agent 2: ACP Subsystem
- `src/benchflow/acp/client.py` — JSON-RPC client
- `src/benchflow/acp/container_transport.py` — stdio pipe transport
- `src/benchflow/acp/session.py` — tool call tracking
- Focus: protocol errors, timeouts, hangs, memory leaks

### Agent 3: Process & Agents
- `src/benchflow/process.py` — DockerProcess, DaytonaProcess
- `src/benchflow/agents/registry.py` — agent configs
- `src/benchflow/viewer.py` — trajectory viewer
- `src/benchflow/cli/main.py` — CLI
- Focus: resource leaks, Harbor private attrs, CLI issues

### Agent 4: Test Coverage
- All files in `tests/`
- Compare against `src/benchflow/` for gaps
- Focus: missing coverage, flaky tests, poor assertions

Each agent should cite `file:line` for every issue. Research only — no edits.

## 2. Review the Reviews

After all agents report:
1. Deduplicate findings across agents
2. Prioritize: Critical (hangs, crashes, data loss) > P1 (fragility) > P2 (quality)
3. Discard false positives (verify each finding before acting)
4. Create task list for fixes

## 3. Fix

Apply fixes in priority order:
1. Make each fix as a discrete edit
2. Run `uv run python -m pytest tests/ -v` after each batch
3. All 45+ tests must pass before proceeding

## 4. Smoke Test on SkillsBench

After fixes pass unit tests, run a real smoke test:

```bash
source .env

# Pick 5 diverse tasks (cross-env if available)
benchflow run -t .ref/skillsbench/task-1 -a claude-agent-acp -e daytona -s skills/ -m claude-haiku-4-5-20251001
benchflow run -t .ref/skillsbench/task-2 -a pi-acp -e daytona -s skills/ -m claude-haiku-4-5-20251001
benchflow run -t .ref/skillsbench/task-3 -a openclaw -e daytona -s skills/ -m claude-haiku-4-5-20251001
```

Verify for each agent:
- [ ] Task completes without error
- [ ] `result.json` has rewards
- [ ] `acp_trajectory.jsonl` is non-empty with tool calls
- [ ] Skills were discoverable by the agent (check trajectory for skill references)

## 5. Update STATUS.md

After all fixes and smoke tests:
- Move fixed issues from "Open Issues" to "Fixed" with date
- Update roadmap (cross off completed items)
- Add any new issues discovered during review

## 6. Post Progress

If Discord thread is active, post:
- Council review summary (issue count by severity)
- Fix batch summary (what was fixed, tests passing)
- Smoke test results (pass/fail per agent)
