# BenchFlow user-audit report (SDK artifacts + live runs)

Generated: 2026-05-24  
Branch: `cursor/user-audit-benchflow-0d54`  
Method: parallel subagents executed real `bench` / `bf.run()` flows on Daytona (no Docker on VM).

## Executive summary

The **core eval pipeline works** on Daytona when using a **live Gemini model** (`gemini-2.5-flash-lite` or `gemini-3.1-flash-lite-preview`). Failures cluster around:

1. **Stale model strings** in docs/skills (`gemini-2.0-flash-lite` hard-fails for new API users).
2. **Codex alias + global `DEFAULT_MODEL`** → Anthropic model without `ANTHROPIC_API_KEY`.
3. **SDK artifact inconsistency on error paths** (missing `config.json`, `rewards.jsonl`, empty `verifier/`).
4. **Three reward/event schemas** (`rewards` dict, `reward_events`, `rewards.jsonl` with `value` vs `reward`).
5. **Self-gen export fixed** but **instruction path** (`/instruction.md` vs gemini `/app` workspace) breaks creator reads.
6. **Skill-eval** does not copy `judge_result.json` to host verifier dir.
7. **ProgramBench YAML** crashes with `FileNotFoundError` if `.cache/programbench-benchflow` missing.
8. **Dashboard** does not surface `error_category` / invalid-measurement badges (ENG-149).

---

## Live runs executed

| Flow | Command pattern | Exit | Notes |
|------|-----------------|------|-------|
| Conformance smoke | `bench eval create --tasks-dir tests/conformance/acp_smoke …` | 0/1 | Pass with `gemini-2.5-flash-lite`; fail with deprecated `gemini-2.0-flash-lite` |
| SkillsBench single | `--source-repo benchflow-ai/skillsbench --include 3d-scan-calc` | 0 | Pipeline OK; task reward 0 (agent `write_file` errors) |
| Self-gen | `--skill-mode self-gen` on `acp_smoke` | 0 | Export to `_self_gen/.../generated-skills/` works; reward 0 (instruction path + write_file) |
| Hosted env | `--source-env primeintellect/general-agent …` | 1 | Env starts; 401 on Prime inference (not local `GEMINI_API_KEY`) |
| Skill eval | `bench skills eval` citation-management | 0 | 1/1; `judge_result.json` not on host |
| Tasks init/check | `bench tasks init` / `check` | 0 | Check passes placeholder scaffolds |
| Trace generate | `--from-file` dry-run | 0 | `--from-local` exit 1 if no `~/.claude/projects` |
| Environment create | `bench environment create tests/conformance/acp_smoke` | 0 | Writes `jobs/environment/{name}__{uuid}/` (undocumented) |
| SDK | `bf.run(RolloutConfig(...))` | 0/1 | RolloutResult matches `result.json` on disk |
| ProgramBench YAML | `--config programbench-gemini-flash-lite.yaml` | 1 | Uncaught `FileNotFoundError` for cache dir |

Job roots: `/tmp/bf-audit-jobs/` (smoke1, smoke-main, skillsbench-one, selfgen1, hosted1, sdk-run, etc.).

---

## `result.json` contract (success path)

Top-level keys observed on passing conformance run:

- Identity: `task_name`, `rollout_name`, `agent`, `agent_name`, `model`
- Scores: `rewards` (object with required `reward` in `[0,1]`, optional `rubric[]`)
- Agent metrics: `n_tool_calls`, `n_prompts`, `agent_result.{n_*_tokens, cost_usd, usage_source, price_source}`
- Errors: `error`, `error_category`, `verifier_error`, `verifier_error_category`
- Diagnostics: `idle_timeout_info`, `sandbox_startup_info`, `transport_error_info`, `verifier_timeout_info`
- Trajectory: `partial_trajectory`, `trajectory_source` (`acp` | `partial_acp` | `scraped`)
- Timing: `started_at`, `finished_at`, `timing` (duplicate of `timing.json`)
- Config snapshot: `scenes[]` with `has_prompt` (often **false** when prompt comes from `instruction.md`)
- Provenance: `source` (optional)

**Not persisted** though present on `RolloutResult`: `evolved_skills`, `reward_events` (except memory patch), inline `trajectory` body.

---

## Error-path artifact gaps (production risk)

When agent fails before verify (e.g. ACP model 500):

| Artifact | Success | Agent error |
|----------|---------|-------------|
| `config.json` | yes | **often missing** |
| `rewards.jsonl` | yes | **missing** |
| `verifier/*` | yes | **empty or absent** |
| `timing.json` | 5 keys | **3 keys** (no `agent_execution`, `verifier`) |
| `result.rewards` | object | `null` |

Downstream dashboards and RL exporters must not assume a uniform file set per rollout directory.

---

## `rewards.jsonl` dual schema

**Verifier terminal line** (`_write_rewards_jsonl`):

```json
{"ts":"…","type":"terminal","source":"verifier","value":1.0,"tag":"reward","step_index":null,"meta":{}}
```

**Memory `RewardEvent` lines** (`reward_event_to_jsonl_record`):

```json
{"ts":"…","type":"…","source":"…","value":<float>,"tag":"…","step_index":<int>,"meta":{"space":"…","granularity":"…"}}
```

`result.json` `reward_events` (when patched) uses **`reward`** and **`step`**, not `value` / `step_index`.

---

## Timestamp anti-pattern

- `result.json`: `"2026-05-24 10:20:13.353455"` (space separator)
- `rewards.jsonl`: `"2026-05-24T10:21:11.838276"` (ISO `T`)

---

## CLI / doc bugs found by usage

| Bug | Repro |
|-----|-------|
| `--agent codex` without `--model` uses Claude default | `bench eval create --agent codex` → needs `ANTHROPIC_API_KEY` |
| `CODEX_AUTH_JSON` ignored | Env set; only `~/.codex/auth.json` checked |
| `environment create --tasks-dir` | Use positional `TASK_DIR` only |
| ProgramBench config without cache | `FileNotFoundError` on fresh clone |
| `tasks generate --from-local` exit 1 when empty | No sessions in `~/.claude/projects` |
| `bench skills list` undocumented | Works; truncates descriptions at 60 chars |
| Harbor compat undocumented | `bench compat harbor-registry` works |
| Hosted `environment show` needs `prime` CLI | Eval create does not |

---

## Self-gen handoff (live)

**Fixed:** `jobs/_self_gen/{task}-{uuid}/generated-skills/*/SKILL.md` populated after run (export at cleanup).

**Still broken:**

- Creator prompt references `/instruction.md`; gemini workspace is `/app` only → `read_file` fails.
- Solver `write_file` fails on Daytona (JS error); creator recovers via shell, solver does not.
- `result.json` has no `skill_mode` or sidecar path link.

---

## Suggested GitHub issues (dedupe against Linear ENG-* before filing)

1. **Codex alias inherits `DEFAULT_MODEL` (claude-haiku) when `--model` omitted**
2. **Honor `CODEX_AUTH_JSON` for subscription auth**
3. **Write `config.json` + `rewards.jsonl` on all rollout terminal states**
4. **Normalize timestamps to ISO-8601 across artifacts**
5. **Fix `has_prompt` when prompt loaded from `instruction.md`**
6. **Stop injecting `CLAUDE_CODE_*` env into non-Claude agents**
7. **Replace deprecated `gemini-2.0-flash-lite` in docs/skills/examples**
8. **ProgramBench: validate `tasks_dir` before `iterdir` with actionable error**
9. **Skill-eval: copy `judge_result.json` into verifier output for host rubric/GEPA**
10. **Upload instruction to `/app/instruction.md` for gemini self-gen creator**
11. **Unify `rewards.jsonl` / `reward_events` field names (`reward` vs `value`)**
12. **Dashboard: surface `error_category` + invalid-measurement badge (ENG-149)**

Linear already tracks: ENG-147 (Daytona retry), ENG-149–158 (failure semantics / dashboard), ENG-96 (short flags on generate).

---

## Dashboard

Generated with `BENCHFLOW_DASHBOARD_JOBS_ROOT=/tmp/bf-audit-jobs`. Core outcome/reward/error mirror `result.json`; **`error_category` not promoted to row UI**; OPEN-2 advisory still open (ENG-157).
