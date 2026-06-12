# Benchflow v0.6.0 — Stress-Test Report

**Branch under test:** `release/v0.6.0` (PR #665), HEAD `b98cf581`, forked to `stress/v0.6.0-fork`
**Date:** 2026-06-11 · **CLI:** `benchflow 0.6.0rc3`
**Method:** 16 parallel deterministic probes (each running real `bench` commands) with an adversarial verification pass on every flagged defect, plus 4 live rollout campaigns (oracle ×Docker, oracle ×Daytona, DeepSeek-v4-flash ×openhands ×Daytona, Docker/Daytona parity isolation). Live model: `deepseek/deepseek-v4-flash`; live judge: `gemini-3.1-flash-lite`; sandboxes: Docker 29.5.3 + Daytona.

## Verdict

The release is **fundamentally healthy**. Core engine, trajectory artifacts, reaper safety, conversion gate, provider routing, and the real agent harness all pass under stress. **3473/3474** fast unit tests pass (the one "failure" is a test-isolation quirk induced by this harness, not a branch defect — see below). No data-integrity or destructive-operation defects were found.

The confirmed issues are **CLI input-validation polish** (raw Python tracebacks where a clean error belongs), **two genuine P1s** (a deadlock and an unactionable missing-extra error), **one reward-integrity foot-gun** in the *example* judge script, and **one broken example oracle**. None block the engine; several would visibly hurt the first-run experience the v0.6 quickstart targets.

| Severity | Confirmed | Area |
|---|---|---|
| **P1** | 2 | `--concurrency 0` deadlock · `--sandbox modal` unactionable error |
| **P2** | 6 | timeout asymmetry · unvalidated `--reasoning-effort` · 3× raw-traceback-instead-of-error · example judge exits 0 on infra failure |
| **P3** | 3 | nonexistent `--tasks-dir` traceback · `cost_usd` null for unpriced model · test-isolation on `GOOGLE_API_KEY` |
| **Example** | 1 | `3d-scan-calc` oracle can't self-pass in default mode |
| Refuted→info | 4 | duplicate YAML keys · unknown-agent fallback · two judge-fallback claims (wrong exec env) |

---

## P1 — confirmed

### P1-1 · `--concurrency 0` deadlocks forever
`bench eval create … --concurrency 0` builds `asyncio.Semaphore(0)` (`src/benchflow/evaluation.py:1007`) which can never be acquired, so the run hangs indefinitely. The `--concurrency` option (`src/benchflow/cli/_options.py:20`) is a plain `int | None` with no validation callback.
**Repro:** `stress-repros/p1_concurrency_zero_deadlock.sh` — first-hand: 20s bounded run → TIMEOUT (deadlock).
**Fix:** validate `concurrency >= 1` at the CLI/`EvalConfig` boundary; reject `0`/negative with a usage error.

### P1-2 · `--sandbox modal` leaks a raw `ModuleNotFoundError`
`modal` is advertised in help (`Sandbox: docker, daytona, or modal`) but with the `sandbox-modal` extra absent the run dies with a raw `ModuleNotFoundError: No module named 'modal'` at `src/benchflow/sandbox/setup.py:186`, surfacing as a per-task `[ERR]`, `errors=1`, and a misleading `Job complete: 0/1`. No "install the `sandbox-modal` extra" hint. The missing-extra guard sits *after* the deferred `from modal import …`.
**Repro:** `stress-repros/p1_modal_missing_extra_error.sh`.
**Fix:** move the extra-availability check ahead of the import (mirror the `sandbox-daytona` pattern) and raise an actionable message.

---

## P2 — confirmed

### P2-1 · `agent.timeout_sec` accepts negative / zero (verifier rejects them)
`AgentConfig.timeout_sec` (`src/benchflow/task/config.py:333`) is `float | None = None` with **no constraint**, so `-5` and `0` pass `bench tasks check` at schema *and* structural. The parallel `VerifierConfig.timeout_sec` (`config.py:244-247`) correctly enforces `gt=0, allow_inf_nan=False`. A task author can ship a nonsensical agent timeout with zero feedback.
**Repro:** `stress-repros/p2_agent_timeout_negative.sh`. **Fix:** add `gt=0, allow_inf_nan=False` to `AgentConfig.timeout_sec`.

### P2-2 · `--reasoning-effort <anything>` is accepted
`normalize_reasoning_effort` (`src/benchflow/_utils/config.py:34-35`) only rejects non-strings, then lowercases and returns — there is no allowed-value set. `--reasoning-effort banana` is accepted and proceeds to launch a sandbox build.
**Repro:** `stress-repros/p2_reasoning_effort_unvalidated.sh`. **Fix:** validate against `{none,low,medium,high,max}` (or the documented set) before launch.

### P2-3 · `--skill-mode bogus` prints a full Rich traceback
Instead of a clean `Invalid value for '--skill-mode'` (as `--level` does), an invalid `--skill-mode` exits 1 with a multi-frame traceback.
**Repro:** `stress-repros/p2_skill_mode_traceback.sh`. **Fix:** make `--skill-mode` an enum/`Choice` option.

### P2-4 · `--agent codex` with no `--model` dumps a raw `ValueError`
Codex has no default model; omitting `--model` raises an uncaught `ValueError` from `effective_model` (`src/benchflow/evaluation.py:276`) shown as a Rich traceback rather than "agent 'codex' requires `--model`".
**Repro:** `stress-repros/p2_agent_no_model_traceback.sh`. **Fix:** catch and re-raise as a clean CLI error.

### P2-5 · Missing-credential surfaces as an in-rollout traceback
Running an agent whose required key is unset (e.g. `claude` with `ANTHROPIC_API_KEY` unset and no `~/.claude/.credentials.json`) fast-fails (~1s, no build started — good) but the failure is a raw Python traceback + per-task `[ERR]` + a misleading `Score: 0/1`, not a clean preflight message.
**Repro:** `stress-repros/p2_missing_credential_traceback.sh`. **Fix:** preflight required-credential check with an actionable "set `$KEY` or log in" message before the rollout loop.

### P2-6 · Example `judge.py` exits 0 + reward 0.0 when the LLM judge never ran
In the shipped `generated-skill-eval/*/verifier/judge.py`, when no judge SDK/key is available the script prints `ERROR: No LLM SDK available` **to stderr but exits 0**, writes `reward.txt=0.0` / `reward.json {"reward":0.0}`. A judge **infrastructure** failure is silently recorded as a legitimate **0.0 score** — indistinguishable from "the agent did the work but it was wrong." This is a reward-integrity foot-gun in the example pack (it does not affect the engine's own verifier path, but these are shipped as canonical authoring examples).
**Repro:** `stress-repros/p2_judge_exit0_on_infra_failure.sh`. **Fix:** exit non-zero (verifier-infra error) when no judge ran, so the harness classifies it as infra-error, not reward 0.0.

---

## P3 — confirmed (low)

- **P3-1 · `--tasks-dir /nonexistent` → raw `FileNotFoundError` traceback** instead of a clean "path not found". Repro: `stress-repros/p3_tasks_dir_nonexistent_traceback.sh`.
- **P3-2 · `cost_usd` is `null` for unpriced models.** The DeepSeek rollout tracked **111,885 tokens** (`usage_source: provider_response`) but `cost_usd: null` / `price_source: null` because there's no price-table entry for `deepseek-v4-flash`. `--usage-tracking required` still **passes** (tokens captured), so this is cosmetic, but `total_cost_usd: 0.0` in the summary can read as "free" rather than "unpriced." Consider surfacing `price_source: unavailable`.
- **P3-3 · Test isolation on `GOOGLE_API_KEY`.** The fast suite has exactly **1** failure *only when a real `GOOGLE_API_KEY` is present in the environment* (an `agents/env.py` resolver test); it passes with the key unset. Not a branch defect, but a CI-portability hazard. Fix: have the test scrub provider keys via the existing `isolate_local_dotenv` pattern.

---

## Broken example — `3d-scan-calc` oracle (P2)

The shipped example `docs/examples/task-md/real-skillsbench/3d-scan-calc` **cannot pass in the default `--agent oracle` invocation** the quickstart/runbook use:

```
Oracle solve.sh exited with rc=1
ModuleNotFoundError: No module named 'mesh_tool'
→ /root/mass_report.json never written → verifier 2 failed → reward 0.0
```

Its `oracle/solve.sh` does `from mesh_tool import MeshAnalyzer` (a module in the task's `environment/skills/mesh-analysis/scripts/`), but the task does **not** declare `environment.skills_dir`, so under the default **no-skill** policy the skill is never injected. The harness behavior is **correct and test-guarded** (`tests/test_agent_setup.py::test_deploy_skills_does_not_autodiscover_bundled_skills`, guarding PR #586); the *example* is the problem. The other two real-skillsbench oracles (`weighted-gdp-calc`, `citation-check`) are self-contained and pass.

**Key parity result:** this fails **identically on Docker and Daytona** — it is **not** a sandbox-parity regression. Combined with `weighted-gdp-calc` and `citation-check` passing identically on both backends, this *positively confirms* the PR's Docker↔Daytona parity claim on the sampled tasks.

**Fix:** add `environment: { skills_dir: environment/skills }` to its `task.md`, or make `solve.sh` self-contained.

---

## Refuted (verified false alarms → reclassified `info`)

1. **Duplicate `agents.roles` keys drop the first role** — true mechanically, but it's universal PyYAML last-key-wins, not a benchflow bug. (Still a *nice-to-have*: pre-parse duplicate-key detection.)
2. **Unknown `--agent` warns then runs as a raw command** — intentional, documented raw-command escape hatch (exit 127, retried), not a silent-pass.
3. **`google-genai` "missing" causes silent reward 0.0** — wrong execution environment: the judge runs in the **verifier container** where `google-genai` is installed (confirmed by the live P13 call), not the host venv.
4. **Gemini `JUDGE_MODEL` silently falls through to OpenAI** — depends on the `genai` import failing, which does not happen in the real verifier container.

---

## Positive coverage (passed under stress)

- **task.md gate:** 7 runnable packages pass schema/structural/publication-grade (RC 0); 4 schema-only fixtures correctly fail higher levels (documented intent); invalid `--level` → clean RC 2. 11/15 malformed task.md fixtures rejected with **field-specific** messages.
- **Authoring round-trips:** `migrate` lossless; `normalize` idempotent (incl. `--write`); `export` to harbor + pier with an **accurate** loss report (`verifier.verifier_md`), **no silent drops**.
- **Provider/model routing:** deepseek → `DEEPSEEK_API_KEY` + base from `DEEPSEEK_BASE_URL`; gemini↔google key aliasing; unknown models degrade to `None` (no crash).
- **Registry/ACP invariants:** 176 passed / 1 intentional skip. **The opencode litellm model-registration gap is fixed on this branch** — the proxy now registers both `<alias>` and `openai/<alias>`, and model selection is correctly ACP-owned.
- **Judge verdict parsing:** 115 passed; `NaN`/`Infinity` actively **rejected** (`_reject_json_constant` hook); injection/empty/prose → clean `ValueError`. Live `gemini-3.1-flash-lite` judge returned a real structured verdict, reward 1.0.
- **Trajectory artifacts:** ATIF-v1.7 conformant; ADP JSONL valid; 112 export tests pass incl. `test_write_redacts_secrets`; **no API-key value leaks** into any rollout artifact.
- **Daytona reaper safety:** dry-run called `client.delete()` **zero** times; a **1773-min-old foreign** sandbox on the shared key (past the 1440m TTL) was correctly **never reaped or even surfaced** — ownership scoping is airtight.
- **Real agent harness:** DeepSeek-v4-flash ×openhands ×Daytona — 12 tool calls, **1.24M tokens** tracked, `telemetry_coverage 1.0`, 0 errors/idle-timeouts, well-formed ATIF agent steps. Scored 0/2 (model capability, not pipeline). openhands correctly skips ACP `set_model` (launch-env owns the model).

## Live-run scoreboard

| Run | Sandbox | Agent/Model | Result | Note |
|---|---|---|---|---|
| oracle smoke | Docker | oracle | 1/1 (gdp 1.0) | ATIF/ADP emitted |
| oracle batch | Daytona | oracle | 2/3 | 3d-scan-calc fails (skill) |
| 3d isolation | Docker | oracle | 0/1 | **same failure as Daytona → parity holds** |
| agent batch | Daytona | openhands / deepseek-v4-flash | 0/2 | real rollout, 1.24M tok, healthy pipeline |

See `stress-repros/` for runnable reproductions of each confirmed defect.
