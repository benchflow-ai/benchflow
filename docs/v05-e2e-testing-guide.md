# v0.5 End-to-End Testing Guide

How to manually verify every feature shipped in v0.5-integration.

## Prerequisites

```bash
# install benchflow from the v0.5-integration branch
git checkout v0.5-integration
uv sync --extra dev --extra sandbox-daytona --locked

# required env vars (set your own keys)
export GEMINI_API_KEY=<your-gemini-key>
export DAYTONA_API_KEY=<your-daytona-key>

# tasks directory (SkillsBench)
export TASKS=.cache/datasets/benchflow-ai/skillsbench/tasks
```

All commands below assume you are in the repo root.

The live Daytona scenarios below are smoke checks. When a failure mode is not
triggered, the related diagnostic field should still be present with a `null`
value. Deterministic failure-path checks live in the unit/integration tests
called out in the relevant sections.

> **Note:** SkillsBench does not include a trivial "hello-world" task.
> The examples below use `weighted-gdp-calc` (fast, ~5 tool calls) as the
> default lightweight task. Swap in any task name from `$TASKS/`.

---

## 1. `--include` / `--exclude` CLI flags (ENG-159, PR #348)

Run a batch eval with only one task included:

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include weighted-gdp-calc \
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
  --include weighted-gdp-calc \
  --include shock-analysis-supply \
  --exclude shock-analysis-supply \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-exclude
```

**Verify:**
- Console prints `Job: 1 tasks` (shock-analysis-supply excluded).
- Only `weighted-gdp-calc` appears in the job directory.
- `summary.json` total is 1.

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
- `"rewards": {"reward": 0.0}` (or whatever the agent earns).
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
# run any task
uv run bench eval create \
  --tasks-dir $TASKS \
  --include weighted-gdp-calc \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-eng148
```

**Verify in `result.json`:**
- `"transport_error_info": null` — field is present but null (no transport error).

The normal-path smoke does not prove the transport failure classifications. Use
the fixture coverage below for deterministic checks:

```bash
uv run python -m pytest tests/test_acp.py -k "transport_error" -v
uv run python -m pytest tests/test_integration_check_results.py -k "transport_error" -v
```

To provoke a real transport error, kill the Daytona sandbox mid-run (advanced):
```bash
# in another terminal, while a long task is running:
daytona sandbox delete <sandbox-id>
```

**Verify in `result.json`:**
- Top-level `"error_category"` is classified from the agent error string. A
  closed stdout transport error reports `"pipe_closed"`; other transport-like
  failures may classify as `"acp_error"`, `"infra_failure"`, or `"other"`
  depending on the error text.
- `"transport_error_info"` is a dict with required keys `reason`,
  `raw_message`, and `transport_diagnosis`.
- `transport_diagnosis` is the nested transport-specific diagnosis, such as
  `remote_session_killed`, `process_exited`, `pty_error`, or `unknown`.
- `process_exit_code`, `process_pid`, and `stderr_snippet` are present only
  when the transport error message contains those details.
- Sandbox probe keys such as `sandbox_reachable`, `sandbox_probe_rc`,
  `sandbox_probe_stdout`, and `sandbox_probe_error` are added only if the
  post-failure sandbox probe runs.

---

## 5. Sandbox startup retries (ENG-147, PR #353)

Sandbox startup failures are intermittent. To verify the field exists:

```bash
# run any task on Daytona
uv run bench eval create \
  --tasks-dir $TASKS \
  --include weighted-gdp-calc \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir /tmp/test-eng147
```

**Verify in `result.json`:**
- `"sandbox_startup_info": null` — field present, null on success.

If a startup failure does occur, the field will contain:
`reason`, `sandbox_id`, `sandbox_state`, `attempts`, `build_timeout_sec`, and
`raw_message`.

In the current Daytona creation failure path, `attempts` records the configured
creation retry budget (`3`), not a live observed counter. Skill export/download
retry failures are reported separately as `export_error`; they do not populate
`sandbox_startup_info`.

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
- If verifier dep install fails:
  `"verifier_error_category": "verifier_dep_install"`.
- If verifier succeeds: `"verifier_error_category": null`.

The live task can pass or fail for reasons other than dependency installation.
Use the deterministic fixture coverage for the failure-path category:

```bash
uv run python -m pytest tests/test_acp.py -k "verifier_error_category" -v
uv run python -m pytest tests/test_integration_check_results.py -k "verifier_dep_install" -v
```

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

This live scenario is expensive and may finish without timing out. Use the
deterministic fixture coverage for the failure-path timeout metadata:

```bash
uv run python -m pytest tests/test_acp.py -k "verifier_timeout_info" -v
uv run python -m pytest tests/test_integration_check_results.py -k "verifier_timeout" -v
```

---

## 8. CTRF path consistency lint (ENG-153, PR #356)

```bash
uv run bench tasks check $TASKS/weighted-gdp-calc
```

**Verify:**
- `✓ weighted-gdp-calc — valid`.
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
  --include weighted-gdp-calc \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir $JOBS

# second run (resume)
uv run bench eval create \
  --tasks-dir $TASKS \
  --include weighted-gdp-calc \
  --agent gemini --model gemini-2.5-flash \
  --sandbox daytona \
  --jobs-dir $JOBS
```

**Verify:**
- Second run logs: `Resuming into existing job directory`.
- Retry directories may exist, but resume scans by task name and chooses the
  newest relevant result.
- Second run reports `Job: 1 tasks, 1 done, 0 to run`.
- Orphan retry artifacts must not cause duplicate pending work or duplicate
  summary counts.
- `summary.json` in both the job subdir and the root `--jobs-dir` (identical content).

---

## 10. Dashboard (ENG-157 + ENG-158, PR #357)

```bash
LINEAR_API_KEY=<your-key> python dashboard/serve.py
# open http://localhost:8777
```

**Verify ENG-157 (stale advisory):**
- The thermo-nuclear advisory does NOT appear under `Open follow-ups`.
- It may appear under `Resolved` with `status: "resolved"`.

**Verify ENG-158 (file:// guidance):**
- If the dashboard is opened as a `file://` URL instead of via `serve.py`, it
  shows a clear error message with recovery instructions:
  `"Cannot load data.json over file://"` and the command
  `python dashboard/serve.py`.

---

## 11. Self-gen skill mode (ENG-155)

```bash
uv run bench eval create \
  --tasks-dir $TASKS \
  --include weighted-gdp-calc \
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

For remote sandboxes (`daytona`, `modal`), token and cost usage telemetry is
expected to be unavailable because the host-side usage proxy is unreachable from
the remote agent. This is not an infrastructure failure. Check:

```python
assert r["agent_result"]["usage_source"] == "unavailable"
assert r["agent_result"]["tokens_in"] is None
assert r["agent_result"]["tokens_out"] is None
```

Then inspect `summary.json` for `telemetry_coverage`; Daytona and Modal smoke
runs may report zero coverage by design.

---

## 13. check_results.py validation

After running any eval, validate results programmatically:

Point the script at the timestamped job subdirectory (not the root `--jobs-dir`)
and pass identity args so it knows what to expect:

```bash
# find the job subdir
JOB_DIR=$(ls -d /tmp/test-include/20*/ | head -1)

uv run python tests/integration/check_results.py "$JOB_DIR" \
  agent=gemini model=gemini-2.5-flash environment=daytona concurrency=4
```

**Verify:**
- Prints a score table with pass/fail/error counts.
- Reports `INVALIDATED` for any tasks with idle timeouts, transport errors, or
  sandbox startup failures (with the structured diagnostic info).
- Source provenance warnings are expected for local runs (git remote mismatch).

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
