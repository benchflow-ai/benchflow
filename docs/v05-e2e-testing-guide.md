# v0.5 End-to-End Testing Guide

How to manually verify every feature shipped in v0.5-integration.

## Prerequisites

```bash
# install benchflow from the v0.5-integration branch
git checkout v0.5-integration
# `sandbox-daytona` is required for the `--sandbox daytona` scenarios below.
# (Add `--extra sandbox-modal` similarly if you plan to swap in `--sandbox modal`.)
uv sync --extra dev --extra sandbox-daytona --locked

# required env vars (set your own keys)
export GEMINI_API_KEY=<your-gemini-key>
export DAYTONA_API_KEY=<your-daytona-key>

# tasks directory (SkillsBench)
export TASKS=.cache/datasets/benchflow-ai/skillsbench/tasks
```

All commands below assume you are in the repo root.

> **Note:** SkillsBench does not include a trivial "hello-world" task.
> The examples below use `weighted-gdp-calc` (fast, ~5 tool calls) as the
> default lightweight task. Swap in any task name from `$TASKS/`.

> **Usage telemetry caveat (Daytona / Modal):** Remote sandboxes run the agent
> on a host that cannot reach BenchFlow's host-bound usage proxy. Default
> `--usage-tracking auto` therefore records `agent_result.usage_source ==
> "unavailable"` unless you configure an external tunnel/ingress with
> `--usage-proxy-url` and `--usage-proxy-port`. Official batch runs that need
> token/cost telemetry should use `--usage-tracking required` so the run fails
> before the agent starts if the external endpoint is missing or unhealthy.
> Local sandboxes (e.g. `--sandbox docker`) populate usage telemetry without a
> tunnel.

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

Transport errors (rc=255, SSH drops) are rare in normal runs. There are two
distinct things this section verifies, and they live in two different fields
that are easy to confuse:

- **Top-level `error_category`** — produced by
  `benchflow._utils.scoring.classify_error(...)`. For transport drops the
  only category it can emit is `pipe_closed` (matched from the substring
  `closed stdout` in the error message). Values like `remote_session_killed`
  and `pty_error` are **not** top-level `error_category` values.
- **Nested `transport_error_info.transport_diagnosis`** — emitted by
  `benchflow.acp.transport` / `benchflow.sandbox.process` at the source via
  the structured `TransportClosedDiagnostic` and surfaced through
  `RolloutDiagnostics`. This is where the finer categorization lives.

### Normal-path schema check

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

### Provoking a real transport error

To provoke a real transport error, kill the Daytona sandbox mid-run (advanced):
```bash
# in another terminal, while a long task is running:
daytona sandbox delete <sandbox-id>
```

**Verify in `result.json`** — `transport_error_info` is the serialized
`TransportClosedDiagnostic` (see `benchflow.diagnostics`) with:

- **Always present:**
  - `reason` — currently the constant string `"transport_closed"`.
  - `raw_message` — first 500 chars of the underlying `ConnectionError`.
  - `transport_diagnosis` — one of:
    - `"remote_session_killed"` (message contains
      `"still alive but its stdout/transport closed"`),
    - `"process_exited"` (message contains `"exited with rc="`),
    - `"pty_error"` (message contains `"PTY readline"`),
    - `"unknown"` (none of the above).
- **Conditional (only when the underlying message exposed them):**
  - `process_exit_code` — int or `None`, parsed from `rc=...`.
  - `process_pid` — int, parsed from `pid=...`.
  - `stderr_snippet` — first 500 chars after `stderr: ...`.
- **Enriched by `_probe_sandbox_health(...)` after the transport dies:**
  - `sandbox_reachable` — bool.
  - `sandbox_probe_rc` — int or `None`.
  - `sandbox_probe_stdout` — present when the probe ran but did not echo
    the expected marker.
  - `sandbox_probe_error` — present when the probe itself raised.
  - `sandbox_probe_error_type` — exception class name (preserved alongside
    `sandbox_probe_error` so post-mortem keeps the original type).
  - `sandbox_probe_traceback` — last 2000 chars of `traceback.format_exc()`
    captured when the probe raised.

**Verify the top-level field too:**
- `"error_category"` is `"pipe_closed"` when the underlying error contained
  `closed stdout`; otherwise it may be `"acp_error"`, `"infra_failure"`, or
  `"other"` depending on which marker matched in `classify_error(...)`. It
  is **not** `"remote_session_killed"` or `"pty_error"` — those belong only
  to `transport_error_info.transport_diagnosis`.

> See also §16 for a deterministic failure-path recipe that avoids the live
> `daytona sandbox delete` dance.

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

If a Daytona sandbox creation failure does occur, the field is populated
from `benchflow.sandbox.protocol.SandboxStartupError.diagnostic.to_dict()`
(a `SandboxStartupDiagnostic` defined in `benchflow.diagnostics`) and
contains the full schema:

- `reason` — currently the constant string `"sandbox_startup_failed"`.
- `sandbox_id` — the Daytona sandbox id when one was allocated before the
  failure, otherwise `null`.
- `sandbox_state` — currently the constant string `"error"` for Daytona
  creation failures.
- `attempts` — **note: currently hardcoded to `3`** in
  `src/benchflow/sandbox/daytona.py` (both call sites), not an observed
  retry counter. Treat this as the configured retry budget, not the
  number of attempts actually made before the failure.
- `build_timeout_sec` — `env.task_env_config.build_timeout_sec`.
- `raw_message` — first 500 chars of the underlying error message.

**Coverage limit — export/download timeouts are not captured here.**
The self-gen skill export/download retry path in `daytona.py` raises a
plain `RuntimeError` after exhausting its retries; it does **not** wrap the
failure in `SandboxStartupError`, so `sandbox_startup_info` stays `null`
for that failure mode. Only sandbox _creation_ failures populate this
field today.

The retry logic is also covered by unit tests:
```bash
uv run python -m pytest tests/ -k "sandbox_startup" -v
```

> See also §16 for a deterministic failure-path recipe (fake backend) that
> exercises `sandbox_startup_info` without relying on live Daytona flakiness.

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
- Resume scanning dedupes by task name and chooses the newest result by
  mtime, so retry artifacts on disk do not cause duplicate work. Concrete
  checks:
  - Second run reports `1 done, 0 to run` (the prior result is picked up).
  - `summary.json.total` remains `1` (no duplicate accounting).
  - No duplicate task scheduled on resume.
  - Retry artifact directories named `task__<uuid8>/` **may** still exist
    on disk from earlier attempts — that is expected. The invariant is
    that `_get_completed_tasks(...)` in `src/benchflow/evaluation.py`
    keys results by `task_name` and keeps only the newest by mtime, so
    orphan retry dirs do not pollute resume decisions.
- `summary.json` in both the job subdir and the root `--jobs-dir` (identical content).

---

## 10. Dashboard (ENG-157 + ENG-158, PR #357)

```bash
LINEAR_API_KEY=<your-key> python dashboard/serve.py
# open http://localhost:8777
```

**Verify ENG-157 (stale advisory):**
- The "thermo-nuclear code-quality-review skill not installed" item must
  not appear under **Open follow-ups** — that section is rendered from
  advisories with `status === "open"`. It is expected and fine for the
  item to still appear under **Resolved** (the underlying entry in
  `dashboard/generate.py` has `status: "resolved"` and a `Resolved: ...`
  detail). Reviewers should assert on the rendered section, not on the
  raw presence of the string in `data.json`.

**Verify ENG-158 (file:// guidance):**
- If the dashboard is opened as a `file://` URL instead of via `serve.py`,
  it renders an error card with the heading
  **"Cannot load data.json over file://"**, a short explanation, and the
  recovery command `python dashboard/serve.py` plus the URL
  `http://localhost:8777`. Reviewers should assert on this rendered copy;
  the older string `"This page must be served over HTTP"` is **not**
  emitted by the current dashboard.

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

---

## 16. Deterministic failure-path coverage for diagnostic fields

Sections §3 (idle timeout), §4 (transport), §5 (sandbox startup), §6
(verifier dep install), and §7 (verifier timeout) document the diagnostic
fields, but most of the **live** recipes only exercise the **normal path**
(field present, value `null`) — `transport_error_info`,
`sandbox_startup_info`, and `verifier_timeout_info` in particular are
extremely hard to trigger end-to-end on Daytona without long, flaky runs.

For each failure path there is a deterministic unit/integration test that
shapes and validates the diagnostic. Use this as the canonical
failure-path coverage; treat the live recipes above as smoke tests, not as
proof that every error category has been exercised.

```bash
# Idle timeout (ENG-149) — field shape + classification
uv run python -m pytest tests/test_scoring.py -k "idle_timeout" -v

# Transport error (ENG-148) — TransportClosedDiagnostic categories, classify_error
#   pipe_closed, and the result.json write path
uv run python -m pytest tests/test_scoring.py::TestClassifyError::test_pipe_closed -v
uv run python -m pytest tests/test_acp.py -k "transport_error_info" -v

# Sandbox startup failure (ENG-147) — SandboxStartupError schema + result.json
uv run python -m pytest tests/test_base_install_imports.py -k "sandbox_startup" -v
uv run python -m pytest tests/test_acp.py -k "sandbox_startup_info" -v

# Verifier dep install (ENG-151) — classifier markers
uv run python -m pytest tests/test_acp.py -k "verifier_dep_install" -v

# Verifier timeout (ENG-152) — verifier wait_for path + verifier_timeout_info
uv run python -m pytest tests/test_verify.py::TestSdkVerify::test_verifier_timeout -v
uv run python -m pytest tests/test_integration_check_results.py -k "verifier_timeout" -v
```

**Verify:** All tests pass. These tests assert the field shape (keys,
types, categories) that the §4–§7 live recipes only partially cover.

**Known gaps the test suite does *not* cover today:**

- Daytona _export/download_ retry failures raise plain `RuntimeError`, not
  `SandboxStartupError`, so they do not populate `sandbox_startup_info`
  (see §5). There is no deterministic fixture that surfaces these as a
  structured diagnostic in `result.json` — that is a product gap to track
  separately, not a documentation gap.
- The `transport_diagnosis` values `process_exited` and `unknown` are
  exercised in `tests/test_acp.py` and `tests/test_sandbox_process.py`
  via the source-side `TransportClosedDiagnostic` emission, but there is
  no live recipe for them. Add one via a fake transport fixture if you
  need true end-to-end coverage.

