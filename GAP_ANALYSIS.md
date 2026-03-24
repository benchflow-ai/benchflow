# benchflow vs Harbor — Gap Analysis

benchflow wraps Harbor as a dependency and adds ACP (Agent Client Protocol) for multi-turn agent communication. This document tracks what Harbor has, what benchflow adds, and what's missing.

## Current State

- **benchflow**: ~3,800 lines across 22 source files
- **Harbor**: ~32,000 lines (used as dependency)
- **Relationship**: benchflow `from harbor import *` and extends with ACP + SDK

## What benchflow adds (Harbor doesn't have)

| Feature | Module | Lines |
|---------|--------|-------|
| ACP protocol client (JSON-RPC 2.0) | `acp/client.py` | 243 |
| ACP session state tracking | `acp/session.py` | 94 |
| ACP types (tools, prompts, content) | `acp/types.py` | 248 |
| ACP transports (stdio, container) | `acp/transport.py`, `acp/container_transport.py` | 163 |
| Live stdio pipes (Docker + Daytona SSH) | `process.py` | 255 |
| Unified SDK (`sdk.run()`) | `sdk.py` | 455 |
| Job orchestration with retries | `job.py` | 200 |
| Metrics collection/aggregation | `metrics.py` | 160 |
| Agent registry (5 ACP agents) | `agents/registry.py` | 130 |
| Trajectory capture (HTTP proxy) | `trajectories/proxy.py` | 415 |
| Trajectory capture (OTel) | `trajectories/otel.py` | 263 |
| Trajectory ATIF converter | `trajectories/claude_code.py` | 246 |
| Trajectory viewer (HTML) | `viewer.py` | 364 |
| Interactive user agent | `agents/user_agent.py` | 58 |

## Gap Analysis

### 1. Orchestration

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Local orchestrator | `LocalOrchestrator` with async workers | `Job` class with semaphore | **Sufficient** — our Job covers the use case |
| Queue orchestrator | `QueueOrchestrator` for distributed runs | None | **Skip** — not needed for single-machine |
| Retry config | `RetryConfig` with include/exclude exceptions, backoff | `RetryConfig` with retry_on_install/pipe/acp | **Sufficient** |
| Trial hooks | `TrialEvent` callbacks (start, end, error) | `on_result` callback in Job | **Add**: pre-trial and error hooks |
| Concurrency control | Configurable n_concurrent | `concurrency` param in JobConfig | **Done** |
| Resume/checkpoint | Skips completed trials | `get_done_tasks()` skips completed | **Done** |

**Action**: Add trial hooks (on_start, on_error). Otherwise sufficient.

### 2. Agent Management

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Agent interface | `BaseAgent` ABC (setup/run/version) | ACP protocol handles this | **Different approach** — we use ACP, not agent classes |
| Agent factory | `AgentFactory` with 17+ agents | `agents/registry.py` with 5 ACP agents | **Expand**: add more agents as ACP ecosystem grows |
| Agent install | Jinja2 templates (`install-*.sh.j2`) | Bash commands in registry | **Sufficient** |
| Agent version | `agent.version()` method | Detected from ACP `initialize` response | **Done** |
| ATIF support flag | `SUPPORTS_ATIF` class var | All ACP agents support trajectory capture | **Done** |
| Agent context | `AgentContext` with MCP servers, skills | Not implemented | **Add**: pass MCP config through ACP |

**Action**: Add MCP server support. Agent registry is sufficient for ACP agents.

### 3. Environments

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Docker | Full compose lifecycle | Uses Harbor's `DockerEnvironment` + `DockerProcess` pipe | **Done** |
| Daytona | Full DinD + direct modes | Uses Harbor's `DaytonaEnvironment` + `DaytonaProcess` SSH pipe | **Done** |
| E2B | `E2BEnvironment` | None | **Add later** if needed |
| Modal | `ModalEnvironment` | None | **Add later** if needed |
| Runloop | `RunloopEnvironment` | None | **Skip** |
| GKE | `GKEEnvironment` | None | **Skip** |
| Environment factory | `EnvironmentFactory` | `_create_environment()` in sdk.py | **Refactor**: extract to factory |

**Action**: Extract environment factory. E2B/Modal are future work.

### 4. CLI

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Run command | `harbor run` with full config | `benchflow run` (basic) | **Expand**: add --concurrency, --retry, --environment flags |
| View command | `harbor traces view` | `benchflow view` (basic HTML) | **Sufficient** |
| Job management | `harbor jobs list/status/results` | None | **Add**: `benchflow jobs` subcommand |
| Dataset commands | `harbor datasets list/download` | None | **Skip** — use task_path directly |
| Agent commands | None | None | **Add**: `benchflow agents list` |

**Action**: Expand `benchflow run` flags. Add `benchflow jobs` and `benchflow agents` commands.

### 5. Verification

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Verifier | `Verifier` class | Uses Harbor's directly | **Done** |
| Reward parsing | text + JSON | Delegated to Harbor | **Done** |
| Custom verifiers | LLM-based verifiers via config | Not exposed | **Add later** |

**Action**: None needed now.

### 6. Trajectory / ATIF

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| ATIF format | Full ATIF v1.6 schema | Basic ATIF types | **Expand**: full ATIF export from ACP sessions |
| HTTP proxy capture | N/A | `TrajectoryProxy` | **Done** |
| OTel capture | N/A | `OTelCollector` | **Done** |
| ACP native capture | N/A | `ACPSession.tool_calls` → JSONL | **Done** |
| Stream-json viewer | N/A | `viewer.py` renders Claude Code format | **Done** |

**Action**: Improve ATIF export to include full session state.

### 7. Metrics

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Metric types | Sum, Min, Max, Mean, UvScript | `BenchmarkMetrics` with pass rate, tool calls, timing | **Sufficient** for now |
| Metric factory | `MetricFactory` | None | **Skip** — not needed yet |
| Per-agent aggregation | Built into orchestrator | `collect_metrics()` per results dir | **Add**: multi-agent comparison |

**Action**: Add multi-agent comparison to metrics.

### 8. Job Persistence / Config

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Job config (YAML/TOML) | Full `JobConfig` pydantic model | `JobConfig` dataclass | **Expand**: add YAML loading |
| Trial config | `TrialConfig` with attempt tracking | Uses Harbor's `TrialPaths` | **Sufficient** |
| Result export | JSON + JSONL | `result.json` per trial + `summary.json` | **Done** |

**Action**: Add YAML config file support for jobs.

### 9. Dataset / Registry

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Dataset client | Download from registry | None | **Skip** — use local task_path |
| Registry | Harbor/JSON registry clients | None | **Skip** |
| Task download + caching | Automatic | None | **Skip** |

**Action**: None — we use local task directories.

### 10. Viewer / UI

| Feature | Harbor | benchflow | Plan |
|---------|--------|-----------|------|
| Web UI | FastAPI + React viewer | None | **Skip** — use local HTML files |
| HTML trajectory viewer | N/A | `viewer.py` serves HTML | **Done** |
| Job comparison | Side-by-side grid | None | **Add later**: comparison mode |

**Action**: None for now.

---

## Priority Roadmap

### P0 — Do Now
- [x] ~~DEBIAN_FRONTEND=noninteractive~~ (fixed)
- [x] ~~Job orchestration with retries~~ (done)
- [x] ~~Metrics collection~~ (done)
- [x] ~~Agent registry~~ (done)
- [x] ~~Model selection via ACP~~ (done)
- [ ] Extract environment factory from sdk.py

### P1 — Next Sprint
- [ ] Expand CLI: `benchflow run` flags (--concurrency, --retry, --environment, --model)
- [ ] Add `benchflow agents list` command
- [ ] Add `benchflow jobs list` command
- [ ] YAML job config file support
- [ ] Trial hooks (on_start, on_error)
- [ ] Multi-agent comparison in metrics
- [ ] Full ATIF export from ACP sessions

### P2 — Future
- [ ] MCP server config pass-through
- [ ] E2B environment support
- [ ] Modal environment support
- [ ] Job comparison viewer
- [ ] LLM-based custom verifiers
- [ ] Daytona snapshots for pre-baked agent images

### Not Planned (Harbor-only)
- Queue orchestrator (distributed)
- GKE/Runloop environments
- Dataset registry
- Metric factory
- Agent classes (we use ACP protocol instead)
