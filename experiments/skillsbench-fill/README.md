# SkillsBench max-effort fill harness

Tooling to run a large, **max-reasoning-effort** SkillsBench evaluation across
multiple models × skill-modes × trials on Daytona (rolling, ≤100 parallel),
review every trajectory for health, publish only healthy cells to the HuggingFace
leaderboard, and monitor the whole pipeline live.

Built for the Opus 4.8 / Gemini 3.5 Flash / MiniMax M3 fill →
`benchflow/skillsbench-leaderboard` `refs/pr/5`, but the harness is
model/task-agnostic. Emits an `experiments_ledger.json` status summary for
live monitoring.

## Architecture

One **ledger** keyed by `cell_id = <model>__<mode>__<task>__t<slot>`. Writers are
decoupled (no shared-file locking at 100-wide): each stage drops a per-cell JSON;
an aggregator merges them.

```
queued → running → completed → review_pass → published
                            ↘ review_fail / quarantined → requeue
```

## Scripts

| Script | Role |
|---|---|
| `reconcile.py` | List the HF PR tree, credit healthy cells, emit `state/queue.jsonl` (full grid minus what's done) |
| `runner.py` | Rolling pooled driver — keeps N `run_cell.sh` in flight (Daytona; Docker fallback for heavy tasks). Flags: `--concurrency`, `--docker-concurrency`, `--only-models`, `--only-tasks`, `--skip-heavy` |
| `run_cell.sh` | One cell = one `bench eval create` (one ephemeral sandbox). Per-model effort wiring + with/without skills. Writes `state/<cell>.json` |
| `review_cell.py` | Mechanical health review of completed cells → `review/<cell>.json` (pass/fail/quarantine). Uses the `benchflow-experiment-review` skill's `extract_harness_skills.py` for skill posture |
| `audit_evidence.py` | Per-cell evidence dump for the **subagent** `benchflow-experiment-review` audit (the authoritative gate — only agent-confirmed cells count) |
| `publish.py` | Push a healthy cell's 5 canonical files + group `metadata.yaml` → HF PR ref; dedup by trial id; **scrubs secrets** + aborts if any survive |
| `build_ledger.py` | Merge queue + state + review + published → `experiments_ledger.json` (the live status ledger) |
| `requeue_quarantine.py` | Reset infra-failed (quarantine) cells → re-run (gemini 429 / opus Bedrock transients) |
| `requeue_nonpass.py` | Reset every non-pass cell (fail + quarantine) → re-run (e.g. after a task/skill fix) |
| `trim_skill_fm.py` | Skill hygiene: trim bloated `SKILL.md` frontmatter that the OpenHands SDK silently drops (see skillsbench #914) |
| `apply_audit.py` | Apply subagent-audit verdicts back onto the ledger / pull flagged cells |
| **Autonomy (cron)** | |
| `keep_runners.sh` | Supervisor: relaunch dead runners + requeue quarantine (every 5 min, flock-guarded) |
| `reap_daytona.py` | Reaper: delete Daytona sandboxes older than ~90 min (every 30 min) |
| `wave_once.sh` | One review → publish → ledger wave (every 5 min) |
| `ledger_once.sh` | Rebuild the ledger (every 1 min) |
| `fresh_start.sh` | Clean restart: apply task fixes, kill the run tree, requeue, relaunch runners |

## Per-cell run recipe

`bench eval create --tasks-dir <sb>/tasks --include <task> --concurrency 1 --agent
openhands --sandbox daytona --usage-tracking required --agent-idle-timeout none`.
Trial id = random hex (reconcile by counting distinct healthy cells per
`(model,mode,task)` toward 3). **Do NOT pass `--reasoning-effort`** (it raises
under openhands); effort is env-delivered via `--agent-env`:

| Model | `--model` | effort wiring |
|---|---|---|
| Opus 4.8 | `aws-bedrock/us.anthropic.claude-opus-4-8` | `BENCHFLOW_BEDROCK_THINKING_EFFORT=max` + `AWS_REGION=us-west-2` |
| Gemini 3.5 Flash | `gemini-3.5-flash` | `LLM_REASONING_EFFORT=high` (its ceiling); tokens not captured (native path) |
| MiniMax M3 | `minimax/MiniMax-M3` | `LLM_REASONING_EFFORT=max` + `LLM_CACHING_PROMPT=false` |

- **with-skills:** add `--skills-dir <sb>/tasks/<task>/environment/skills --skill-mode with-skill`. **without:** omit.

## Health gate

Healthy iff: `result.json` parseable, `error==null` (a normal idle/agent timeout is
OK), `partial_trajectory==false`, `reward∈[0,1]`, `timing.total>0`, both `acp`+`llm`
trajectories present; **skill posture** — with ⇒ `task_skills_loading==1`, without ⇒
`==0`; meta (tokens+timing) present except Gemini tokens (native path). Infra
failures (Daytona/Docker/ENOSPC/provider error) ⇒ quarantine+requeue, not fail.
Reward 0/1 does NOT affect health.

The **authoritative** gate is a subagent cluster running `benchflow-experiment-review`
per cell (fed by `audit_evidence.py`); only agent-confirmed cells count.

## Configuration

Scripts read credentials from env files (`~/keys.env` / `~/.env`) at runtime —
**no secrets are committed**. Adjust for your environment:
- Keys: `ANTHROPIC_API_KEY` / `AWS_BEARER_TOKEN_BEDROCK`, `GEMINI_API_KEY`,
  `MINIMAX_API_KEY` / `MINIMAX_BASE_URL`, `DAYTONA_API_KEY`, `HUGGING_FACE_TOKEN`.
- Working dir defaults to `~/sb-fill/` on the run host; the `.env` path in
  `publish.py`/`reconcile.py` and the jobs path in `read_*.py` may need adjusting.
- **Env hygiene:** `unset OPENAI_* LLM_* LITELLM_* BENCHFLOW_PROVIDER_*` before
  running (All-Hands proxy vars otherwise hijack model routing).
