# Linear-ready issues — benchflow v0.6.0 stress test

**Filed in Linear `eng` team on 2026-06-11:** BF-1→ENG-248 · BF-2→ENG-249 · BF-3→ENG-250 ·
BF-4→ENG-251 · BF-5→ENG-252 · BF-6→ENG-253 · BF-7→ENG-254 · BF-8→ENG-255 · BF-9→ENG-256 ·
BF-10→ENG-257 (deferred/open). All labeled `bug`.

Each fix below is applied on branch `stress/v0.6.0-fork` and dogfooded.

Suggested labels: `bug`, `v0.6.0`, `cli`, `dx`, `examples`.

---

## BF-1 · `bench eval create --concurrency 0` deadlocks forever  — Priority: Urgent (P1)
**Problem:** `--concurrency 0` builds `asyncio.Semaphore(0)` (`src/benchflow/evaluation.py:1007`), which can never be acquired → the run hangs indefinitely. The option had no validation (`src/benchflow/cli/_options.py:20`).
**Evidence:** 20s bounded run → TIMEOUT. Repro: `stress-repros/p1_concurrency_zero_deadlock.sh`.
**Fix (applied):** reject `--concurrency < 1` (and `--build-concurrency < 1`) up front in `cli/main.py` with a clean usage error. Dogfood: `RC=1`, "must be >= 1", no hang.
**Status:** Fixed + dogfooded.

## BF-2 · `--sandbox modal` leaks a raw `ModuleNotFoundError`  — Priority: Urgent (P1)
**Problem:** `modal` is advertised in help, but without the `sandbox-modal` extra the run died with a raw `ModuleNotFoundError` at `src/benchflow/sandbox/setup.py:186` (the missing-extra guard sat *past* the deferred `import modal`), surfacing as a per-task `[ERR]` + misleading `Job complete: 0/1`.
**Evidence:** Repro: `stress-repros/p1_modal_missing_extra_error.sh`.
**Fix (applied):** (a) eager `import modal` inside `_create_benchflow_modal_environment_class()` so the existing `except ModuleNotFoundError` guard fires with the actionable message; (b) upfront CLI preflight so `--sandbox modal` without the extra fails fast (no rollout, no traceback). Dogfood: `RC=1`, no traceback, "Install it with `uv sync --extra sandbox-modal`".
**Status:** Fixed + dogfooded.

## BF-3 · `agent.timeout_sec` accepts negative / zero (verifier rejects them)  — Priority: High (P2)
**Problem:** `AgentConfig.timeout_sec` (`src/benchflow/task/config.py:333`) had no constraint, so `-5`/`0` passed `bench tasks check` at schema *and* structural; `verifier.timeout_sec` correctly enforces `gt=0`.
**Evidence:** Repro: `stress-repros/p2_agent_timeout_negative.sh`.
**Fix (applied):** add `gt=0, allow_inf_nan=False` to `AgentConfig.timeout_sec`. Dogfood: `AgentConfig(timeout_sec=-5)` and `=0` now rejected; `900`/`None` ok.
**Status:** Fixed + dogfooded.

## BF-4 · `--reasoning-effort <anything>` accepted  — Priority: High (P2)
**Problem:** `normalize_reasoning_effort` (`src/benchflow/_utils/config.py`) only type-checked; any string (e.g. `banana`) passed through and launched a build.
**Evidence:** Repro: `stress-repros/p2_reasoning_effort_unvalidated.sh`.
**Fix (applied):** validate against `{none, minimal, low, medium, high, xhigh, max}` (superset of all values used in code/tests); the CLI already wraps the `ValueError` into a clean exit. Dogfood: `banana` → rejected with the valid list; `max`/`MAX`/`xhigh`/`minimal` ok.
**Status:** Fixed + dogfooded.

## BF-5 · `--skill-mode bogus` prints a Rich traceback  — Priority: High (P2)
**Problem:** invalid `--skill-mode` raised a multi-frame traceback (from `skill_policy`) instead of a clean choice error.
**Evidence:** Repro: `stress-repros/p2_skill_mode_traceback.sh`.
**Fix (applied):** validate `--skill-mode ∈ {no-skill, with-skill, self-gen}` at the CLI boundary. Dogfood: `RC=1`, no traceback, "choose no-skill, with-skill, or self-gen".
**Status:** Fixed + dogfooded.

## BF-6 · `--agent codex` with no `--model` dumps a raw `ValueError`  — Priority: High (P2)
**Problem:** `effective_model` (`src/benchflow/evaluation.py:276`) raised an uncaught `ValueError` shown as a traceback once the rollout started.
**Evidence:** Repro: `stress-repros/p2_agent_no_model_traceback.sh`.
**Fix (applied):** pre-validate the agent/model pairing in `cli/main.py` (`try: effective_model(...) except ValueError: clean Exit`). Dogfood: `RC=1`, no traceback, clean message.
**Status:** Fixed + dogfooded.

## BF-7 · Example `judge.py` exits 0 + reward 0.0 when the LLM judge never ran  — Priority: High (P2)
**Problem:** the generated `generated-skill-eval/*/verifier/judge.py` recorded a judge **infrastructure** failure (missing SDK/key/API error) as a legitimate **0.0 score** with exit 0 — indistinguishable from a real agent failure. Reward-integrity foot-gun.
**Evidence:** Repro: `stress-repros/p2_judge_exit0_on_infra_failure.sh`.
**Fix (applied):** `call_llm` raises `JudgeUnavailableError` when no judge ran; `main()` exits non-zero and writes no reward, so the harness classifies it as a verifier-infra error. Fixed in the generator template (`src/benchflow/templates/judge.py.tmpl`) **and** all 3 shipped copies. Dogfood: exit `1`, no reward file written.
**Status:** Fixed + dogfooded.

## BF-8 · `--tasks-dir /nonexistent` raises a raw `FileNotFoundError`  — Priority: Medium (P3)
**Problem:** a missing tasks dir produced a raw traceback instead of a clean "not found".
**Evidence:** Repro: `stress-repros/p3_tasks_dir_nonexistent_traceback.sh`.
**Fix (applied):** existence check at the CLI boundary. Dogfood: `RC=1`, no traceback, "--tasks-dir not found: …".
**Status:** Fixed + dogfooded.

## BF-9 · `3d-scan-calc` oracle can't self-pass in default mode  — Priority: High (P2, examples)
**Problem:** the shipped oracle `solve.sh` did `from mesh_tool import MeshAnalyzer` (a bundled skill), which isn't injected under the default no-skill policy → `ModuleNotFoundError` → no `mass_report.json` → reward 0.0. Failed identically on Docker + Daytona (so **not** a parity bug; the no-skill default is correct and test-guarded). Baking the skill into the Dockerfile would break the no-skill guard for agent runs.
**Evidence:** Repro: `stress-repros/p2_3d_scan_calc_broken_oracle.sh`.
**Fix (applied):** rewrote `solve.sh` to be self-contained, mirroring the verifier's `_get_ground_truth()` STL parse + connected-component volume exactly (oracle stays in lockstep with the reward). Dogfood: Docker oracle run → **reward 1.0** (was 0.0), both verifier tests PASSED, oracle computes Volume/Material ID 42/Mass with no `ModuleNotFoundError`.
**Status:** Fixed + dogfooded.

---

**Regression coverage:** `tests/test_cli_arg_validation.py` (22 tests) locks in BF-1, BF-3, BF-4, BF-5, BF-6, BF-8, and BF-2's preflight. Full fast suite after fixes: **3474 passed, 0 failed, 13 skipped**.

---

## BF-10 · Missing-credential surfaces as an in-rollout traceback  — Priority: Medium (P2) — DEFERRED
**Problem:** an agent whose required key is unset fast-fails (~1s, no build — good) but as a raw Python traceback + per-task `[ERR]` + misleading `Score: 0/1`, rather than a clean preflight message. The underlying message (`ANTHROPIC_API_KEY required for model …`) is already informative.
**Why deferred:** a clean fix means either an upfront credential preflight (risk: false-blocking `--source-env`/plane runs where the plane injects provider keys) or reclassifying the error in the rollout's per-task error path (a `MissingAgentCredentialError(ValueError)` raised at `agents/env.py:738`, logged as a clean one-liner instead of `exc_info`). Both deserve their own focused change + tests rather than a risky bundle in this pass.
**Recommendation:** typed exception + per-task error classification. Repro: `stress-repros/p2_missing_credential_traceback.sh`.
**Status:** Open (deferred, not fixed).

---

### Refuted during stress testing (do NOT file)
- Duplicate `agents.roles` keys → universal PyYAML last-key-wins, not a benchflow bug.
- Unknown `--agent` warn-then-raw-command → intentional, documented escape hatch.
- `google-genai` "missing" / gemini→openai judge fallback → wrong execution env (judge runs in the verifier container where genai is installed; live Gemini-3.1-flash-lite call confirmed working).
