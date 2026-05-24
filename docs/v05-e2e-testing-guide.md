# v0.5 End-to-End Testing Guide

How to manually verify every feature shipped in v0.5-integration.

## Prerequisites

```bash
# install benchflow from the v0.5-integration branch
git checkout v0.5-integration
uv sync --extra dev --locked

# required env vars (set your own keys)
export GEMINI_API_KEY=<your-gemini-key>
export DAYTONA_API_KEY=<your-daytona-key>

# optional: tasks directory (SkillsBench)
TASKS=.cache/datasets/benchflow-ai/skillsbench/tasks
```

All commands below assume you are in the repo root.

---

## 1. `--include` / `--exclude` CLI flags (ENG-159, PR #348)

Run a batch eval with only one task included:

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include hello-world-task \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-include
```

**Verify:**
- Console prints `Job: 1 tasks` (not the full task count).
- `summary.json` has `"total": 1`.

Repeat with `--exclude`:

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --exclude hello-world-task \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-exclude
```

**Verify:**
- The excluded task does not appear in the job directory.
- `summary.json` total is (all tasks - 1).

---

## 2. Verifier rc!=0 accepted (ENG-150, PR #349)

Run a task whose verifier exits nonzero after producing a reward:

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include threejs-to-obj \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-eng150
```

**Verify in `result.json`:**
- `"reward": 0.0` (or whatever the agent earns).
- `"error": null` — NOT `"verifier_errored"`.
- `"verifier_error": null`.
- Console log contains: `Verifier exited with rc=1 but produced reward output; accepting reward`.

**Before this fix:** The same task would report `verifier_errored` instead of
`failed`, making it look like infrastructure noise.

---

## 3. Idle timeout diagnostics (ENG-149, PR #350)

Force an idle timeout with a very short timeout:

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include data-to-d3 \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --agent-idle-timeout 5 \
  --jobs-dir /tmp/test-eng149
```

**Verify in `result.json`:**
- `"error_category": "idle_timeout"`.
- `"idle_timeout_info"` is a dict with keys:
  - `reason`, `idle_timeout_sec`, `idle_duration_sec`,
    `wall_clock_elapsed_sec`, `n_tool_calls`, `n_message_chunks`,
    `n_thought_chunks`, `last_activity_at`.

**Before this fix:** `error` was a bare string with no structured info.

---

## 4. Transport error diagnostics (ENG-148, PR #352)

Transport errors (rc=255, SSH drops) are rare in normal runs. To verify the
field exists even in non-error cases:

```bash
# run any successful task
uv run bench eval create \
  --tasks-dir $TASKS \
  --include hello-world-task \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-eng148
```

**Verify in `result.json`:**
- `"transport_error_info": null` — field is present but null (no transport error).

To provoke a real transport error, kill the Daytona sandbox mid-run (advanced):
```bash
# in another terminal, while a long task is running:
daytona sandbox delete <sandbox-id>
```

**Verify in `result.json`:**
- `"transport_error_info"` is a dict with `process_exit_code`,
  `transport_diagnosis`, `sandbox_reachable`, `stderr_snippet`.
- `"error_category"` is one of: `pipe_closed`, `remote_session_killed`,
  `pty_error`.

---

## 5. Sandbox startup retries (ENG-147, PR #353)

Sandbox startup failures are intermittent. To verify the field exists:

```bash
# run any task on Daytona
uv run bench eval create \
  --tasks-dir $TASKS \
  --include hello-world-task \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-eng147
```

**Verify in `result.json`:**
- `"sandbox_startup_info": null` — field present, null on success.

If a startup failure does occur, the field will contain:
`sandbox_id`, `sandbox_state`, `attempts`, `build_timeout_sec`.

The retry logic is also covered by unit tests:
```bash
uv run python -m pytest tests/ -k "sandbox_startup" -v
```

---

## 6. Verifier dep install classification (ENG-151, PR #354)

Run `simpo-code-reproduction` (known to have heavy pip deps):

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include simpo-code-reproduction \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-eng151
```

**Verify in `result.json`:**
- If verifier dep install fails: `"verifier_error_category": "dep_install"`.
- If verifier succeeds: `"verifier_error_category": null`.

---

## 7. Verifier timeout diagnostics (ENG-152, PR #355)

Run a task with a heavy verifier (e.g., quantum-numerical-simulation):

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include quantum-numerical-simulation \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-eng152
```

**Verify in `result.json`:**
- If verifier times out: `"verifier_timeout_info"` has `timeout_budget_sec`,
  `elapsed_sec`, `task_name`.
- If verifier finishes: `"verifier_timeout_info": null`.

---

## 8. CTRF path consistency lint (ENG-153, PR #356)

```bash
uv run bench tasks check $TASKS/hello-world-task
```

**Verify:**
- No CTRF path consistency warnings.
- Exit code 0.

To see a failure, create a task with mismatched CTRF paths and run `bench tasks check`.

---

## 9. Resume + retry dedup (ENG-160, PR #351)

Run an eval, then re-run with the same `--jobs-dir`:

```bash
JOBS=/tmp/test-eng160

# first run
uv run bench eval create \
  --tasks-dir $TASKS \
  --include hello-world-task \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir $JOBS

# second run (resume)
uv run bench eval create \
  --tasks-dir $TASKS \
  --include hello-world-task \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir $JOBS
```

**Verify:**
- Second run logs: `Resuming into existing job directory`.
- No orphan retry directories created (only one task dir per task, not duplicates).
- `summary.json` in both the job subdir and the root `--jobs-dir` (identical content).

---

## 10. Dashboard (ENG-157 + ENG-158, PR #357)

```bash
LINEAR_API_KEY=<your-key> python dashboard/serve.py
# open http://localhost:8777
```

**Verify ENG-157 (stale advisory):**
- Advisories section does NOT show a "thermo-nuclear" advisory banner.

**Verify ENG-158 (file:// guidance):**
- If the dashboard is opened as a `file://` URL instead of via `serve.py`, it
  shows a clear error message with recovery instructions:
  `"This page must be served over HTTP"` and the command to start the server.

---

## 11. Self-gen skill mode (ENG-155)

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include hello-world-task \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --skill-mode self-gen \
  --skill-creator-dir $TASKS/../.agents/skills/skill-creator \
  --jobs-dir /tmp/test-selfgen
```

**Verify:**
- Console shows two scenes: `self-gen-creator` then `self-gen-solver`.
- `result.json` has `"scenes"` array with both scene names.
- `_self_gen/` directory created alongside the job directory with generated skills.

---

## 12. All diagnostic fields present

For **any** completed eval, verify `result.json` contains all six diagnostic
fields (value may be null when not triggered):

```python
import json
r = json.load(open("result.json"))
for field in [
    "error_category",
    "idle_timeout_info",
    "sandbox_startup_info",
    "transport_error_info",
    "verifier_timeout_info",
    "verifier_error_category",
]:
    assert field in r, f"Missing: {field}"
    print(f"  {field}: {r[field]}")
```

---

## 13. check_results.py validation

After running any eval, validate results programmatically:

```bash
python tests/integration/check_results.py /tmp/test-include
```

**Verify:**
- Prints a score table.
- Reports `INVALIDATED` for any tasks with idle timeouts, transport errors, or
  sandbox startup failures (with the structured diagnostic info).
- Exit code 0 for clean runs.

---

## 14. Secret leak audit

After any eval run, check that no API keys leaked into trajectories:

```bash
grep -rn "AIzaSy\|dtn_\|GEMINI_API_KEY\|DAYTONA_API_KEY" /tmp/test-include/
```

**Verify:** No matches.

---

## 15. Full regression suite

```bash
uv run python -m pytest tests/ -x -q
uv run ty check src/
uv run ruff check .
```

**Verify:** 1910+ passed, 0 failed. ruff + ty clean.

---

## Quick Smoke Test (all features in one run)

A single command that exercises most features at once:

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include hello-world-task \
  --include threejs-to-obj \
  --exclude data-to-d3 \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --agent-idle-timeout 600 \
  --jobs-dir /tmp/smoke-test

# then audit
python tests/integration/check_results.py /tmp/smoke-test
grep -rn "AIzaSy\|dtn_" /tmp/smoke-test/
python3 -c "
import json, pathlib
for f in pathlib.Path('/tmp/smoke-test').rglob('result.json'):
    r = json.loads(f.read_text())
    fields = ['error_category','idle_timeout_info','sandbox_startup_info',
              'transport_error_info','verifier_timeout_info','verifier_error_category']
    missing = [k for k in fields if k not in r]
    print(f'{f.parent.name}: reward={r.get(\"reward\")}, missing_diags={missing}')
"
```

**Verify:**
- 2 tasks run (hello-world-task + threejs-to-obj), data-to-d3 excluded.
- All 6 diagnostic fields present in every result.json.
- No secret leaks.
- `threejs-to-obj` shows ENG-150 behavior (rc=1 accepted if verifier produces reward).
