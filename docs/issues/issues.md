# Benchflow Issues — Gap Analysis (2026-04-10)

Grounded in: source code on `main` (0.2.0) + `feat/skill-eval-multi-agent` + `feat/harden`,
smoke test results from kywch (9/9 tasks clean on harden), reviewer findings from GPT-5.4 + Opus.

---

## Issue 1: Job orchestrator lacks backpressure, priority scheduling, and per-environment concurrency limits

**Labels:** enhancement, orchestration, P1

**Current state (code):**
- `job.py:372` — concurrency is a single `asyncio.Semaphore(cfg.concurrency)`
- No backpressure: if tasks start failing (Docker OOM, Daytona quota), the semaphore keeps launching new ones
- No priority: all tasks treated equally, no way to prioritize flaky tasks for retry first
- No per-environment limits: `concurrency=64` applies globally, but Docker can only handle ~4 while Daytona can do 64+
- No task-level timeout: relies entirely on agent-level `timeout_sec` from task.toml

**What Harbor does:**
- Queue-based orchestration with backpressure (stops launching when error rate > threshold)
- Per-environment concurrency routing (Docker: 4, Modal: 100, etc.)
- Priority queue for retries
- Artifact collection from sandboxes after each trial

**Proposed:**
1. Add error-rate backpressure: pause launching when >20% of recent tasks errored
2. Per-environment concurrency config: `{"docker": 4, "daytona": 64}`
3. Priority queue: retries go to front of queue
4. Task-level timeout override from job config

**Affected files:** `src/benchflow/job.py`

---

## Issue 2: LLM judge verifier API key not forwarded to verifier container

**Labels:** bug, skill-eval, P0

**Current state (code):**
- `skill_eval.py` generates tasks with `judge.py.tmpl` that calls `anthropic.Anthropic()` or `openai.OpenAI()`
- These SDKs need `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in the verifier container
- Harbor verifiers run in a separate container context — SDK does NOT forward API keys to the verifier
- `sdk.py` only sets agent env vars via `_resolve_agent_env()` — verifier env is handled by Harbor's `Verifier` class
- **Result:** Every LLM judge call will fail silently and return 0.0. Users will see all-zero scores with no explanation.

**Proposed fix:**
- Add `verifier_env` parameter to `SDK.run()` that gets forwarded to Harbor's verifier
- OR: generate a wrapper `test.sh` that re-exports the host's API key into the verifier env
- Short-term hack: bake `ARG ANTHROPIC_API_KEY` into the generated Dockerfile and pass via `--build-arg`

**Affected files:** `src/benchflow/sdk.py`, `src/benchflow/skill_eval.py`, `src/benchflow/templates/test.sh.tmpl`

---

## Issue 3: `benchflow eval` and `benchflow skill-eval` naming collision

**Labels:** UX, CLI, P1

**Current state (code):**
- `cli/main.py:270` — `benchflow eval` runs tasks with optional skill injection (wrapper around Job)
- `cli/main.py:492` — `benchflow skill-eval` generates ephemeral tasks from evals.json (new feature)
- Both show up in `benchflow --help` with similar descriptions
- A user running `benchflow eval` when they mean `skill-eval` (or vice versa) will get wrong behavior silently

**Proposed:**
- Rename `eval` → `run-batch` or `bench` (it's just a simpler Job wrapper)
- Or: merge `eval` into `skill-eval` with `--tasks-dir` fallback (if no evals.json, behave like current `eval`)
- Or: deprecate `eval` entirely since `job` covers the same use case with more options

**Affected files:** `src/benchflow/cli/main.py`

---

## Issue 4: Multi-agent sandbox (Docker Compose + MCP orchestrator)

**Labels:** feature, multi-agent, P1

**Current state (code):**
- `process.py` only supports single-agent-per-container (DockerProcess, DaytonaProcess)
- `sdk.py:run()` manages one ACP session per call
- `_env_setup.py` detects docker-compose but only for services (gmail-sim, gcal-sim), not for multi-agent
- Harbor's simulated-user cookbook uses Docker Compose + FastMCP sidecar — we can extend this pattern

**Design doc:** `docs/multi-agent-sandbox-design.md` (on `feat/skill-eval-multi-agent` branch)

**Key engineering problems (from reviewer feedback):**
1. **Caller identity:** MCP doesn't expose who called a tool. Options: separate endpoints per agent, include caller in args, or use session metadata
2. **Deadlock prevention:** Agent A calls ask_B while B calls ask_A simultaneously → deadlock. Need async message queues or turn-based protocol
3. **Turn limit scoping:** Global turn counter vs per-agent-pair limits
4. **State management:** Global variables in orchestrator need locks for concurrent access
5. **Daytona support:** Docker Compose doesn't work on Daytona. Need Option B (single container, process isolation)

**Depends on:** Harbor internalization (Docker Compose generation from task.toml)

**Affected files:** `src/benchflow/sdk.py`, `src/benchflow/process.py`, new `src/benchflow/orchestrator.py`

---

## Issue 5: ATIF (Agent Trajectory Interchange Format) export not wired into SDK

**Labels:** enhancement, trajectory, P2

**Current state (code):**
- `trajectories/atif.py` — full ATIF data model defined (ATIFDocument, AgentStep, ToolCall, etc.)
- `trajectories/claude_code.py` — converter from Claude Code stream-json to ATIF
- Neither is called from `sdk.py:run()` or any CLI command
- `_trajectory.py` captures ACP-native trajectory but does not convert to ATIF
- Harbor is standardizing on ATIF — we need parity for interop

**Proposed:**
1. Wire ATIF conversion into SDK.run() — after ACP trajectory capture, convert to ATIF and save as `trajectory/atif.json`
2. Add `benchflow export --format atif` CLI command
3. Add ATIF to the viewer (viewer.py currently only renders stream-json and ACP JSONL)

**Affected files:** `src/benchflow/sdk.py`, `src/benchflow/_trajectory.py`, `src/benchflow/trajectories/atif.py`, `src/benchflow/cli/main.py`

---

## Issue 6: Skill-eval evals.json per-case environment overrides parsed but not used

**Labels:** bug, skill-eval, P2

**Current state (code):**
- `skill_eval.py:112` — `EvalCase.environment` field parsed from evals.json
- `generate_tasks()` never reads `case.environment` — per-case env vars are silently dropped
- Schema documents the field, giving users false expectations

**Proposed:**
- Either implement: inject `case.environment` into the generated task's agent_env
- Or remove from schema and parsing code (don't document what doesn't work)

**Affected files:** `src/benchflow/skill_eval.py`

---

## Issue 7: No E2B/Modal environment support

**Labels:** enhancement, environments, P2

**Current state (code):**
- `process.py` has only `DockerProcess` and `DaytonaProcess`
- `sdk.py` checks `environment` parameter against these two only
- Harbor supports Docker, Modal, E2B, Runloop — we're missing 3 providers
- Users with Modal/E2B credits can't use benchflow

**Proposed:**
1. Add `E2BProcess` — E2B has a Python SDK with sandbox API
2. Add `ModalProcess` — Modal has container orchestration with GPU support
3. Abstract common interface: `BaseProcess.exec()`, `.upload()`, `.download()`

**Affected files:** `src/benchflow/process.py`, `src/benchflow/sdk.py`

---

## Issue 8: No hosted documentation site

**Labels:** docs, P1

**Current state:**
- All docs are markdown files in `docs/` directory
- README.md is a monolith (quickstart + CLI ref + results + architecture)
- No searchable docs, no versioning, no cross-linking
- Harbor has harborframework.com/docs, Terminal-Bench has tbench.ai/docs

**Proposed:**
- Set up GitHub Pages or Mintlify with docs/ as source
- Split README into landing page that links to docs/
- Add: quickstart, guides (benchmark-your-agent, create-tasks, run-at-scale), reference (SDK, CLI, agents, task-format)

**New files needed:** docs site config, restructured docs/

---

## Issue 9: GEPA integration — trace format unverified + trajectory data missing

**Labels:** enhancement, skill-eval, P2

**Current state (code):**
- `skill_eval.py:export_gepa_traces()` writes trace files with `n_tool_calls` but NOT the actual trajectory events
- GEPA paper (arXiv:2507.19457) describes input format but our trace structure is "best guess"
- If format doesn't match, the entire export is useless
- `CaseResult.rubric_results` is populated from `judge_result.json` but trajectory events are not stored in CaseResult

**Proposed:**
1. Add `trajectory` field to `CaseResult` dataclass
2. Populate from `RunResult.trajectory` after each task run
3. Include full trajectory in GEPA trace JSON
4. Verify format against GEPA's actual input spec (contact GEPA team or read source)

**Affected files:** `src/benchflow/skill_eval.py`

---

## Issue 10: Custom Dockerfile in skill-eval breaks with/without skill comparison

**Labels:** bug, skill-eval, P1

**Current state (code):**
- `skill_eval.py:_default_dockerfile()` includes `COPY skills/ /home/user/.claude/skills/` for with-skill mode
- If user provides `evals/Dockerfile`, it's used verbatim — no skill injection happens
- The with-skill vs baseline comparison becomes meaningless (both runs are identical)

**Proposed:**
- Append skill COPY commands to custom Dockerfile when `with_skill=True`
- Or: document that custom Dockerfiles must include skill paths manually
- Or: use a Dockerfile overlay/multi-stage approach

**Affected files:** `src/benchflow/skill_eval.py`

---

## Issue 11: No `--dry-run` flag for skill-eval

**Labels:** enhancement, skill-eval, P2

**Current state (code):**
- `benchflow skill-eval` immediately starts agent runs (expensive: $0.50-2.00 per run)
- No way to validate evals.json and preview what tasks would be generated without spending money
- Reviewers from both GPT-5.4 and Opus flagged this

**Proposed:**
- Add `--dry-run` flag that generates ephemeral tasks, prints the plan (N cases × M agents × 2 modes = X runs, est. cost $Y), then exits
- Show generated instruction.md content for verification

**Affected files:** `src/benchflow/cli/main.py`, `src/benchflow/skill_eval.py`

---

## Issue 12: evals.json lacks JSON schema / pydantic validation

**Labels:** enhancement, skill-eval, P2

**Current state (code):**
- `skill_eval.py:load_eval_dataset()` does hand-rolled validation (checks for `cases`, `question`, duplicate IDs)
- No type checking on field values (e.g., `timeout_sec: "300"` string silently accepted)
- No `evals.schema.json` file for IDE autocompletion
- NVIDIA users expect schema-validated configs

**Proposed:**
1. Define pydantic models for evals.json (EvalDatasetSchema, EvalCaseSchema)
2. Generate JSON schema from pydantic and publish as `docs/evals.schema.json`
3. Validate on load with clear error messages

**Affected files:** `src/benchflow/skill_eval.py`, new `docs/evals.schema.json`
