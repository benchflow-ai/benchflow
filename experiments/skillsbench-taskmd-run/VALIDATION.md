# SkillsBench → task.md run-through — validation

Adapts SkillsBench legacy tasks (`task.toml` + `instruction.md` + `tests/` + `solution/`)
to the native `task.md` package, then runs them on **openhands + deepseek-v4-flash**
across the three skill modes (no-skill / with-skill / self-gen).

## Pipeline validated end-to-end — one task, all 3 modes, real rollouts

Task: **`citation-check`**. Each rollout is genuine (`n_tool_calls > 0`,
`total_tokens > 0`, verifier executed, `reward` non-None — i.e. not a resumed/oracle pass):

| mode | sandbox | reward | tool_calls | tokens |
|---|---|---|---|---|
| no-skill   | daytona | 0.0 | 7  | 607,535   |
| with-skill | daytona | 0.0 | 23 | 812,456   |
| self-gen   | docker  | 0.0 | 65 | 4,104,949 |

`reward 0.0` = the model didn't solve the task (expected for flash); what's proven is
that the **adapted task.md runs through** the full harness in every skill mode.

## Audit findings on the adapted packages

Auditing a migrated package (`ada-bathroom-plan-repair`) against the real `TaskConfig`
schema surfaced **one runtime defect** plus cosmetic serializer noise. Fixed/handled in
`adapt_clean.py`:

1. **Runtime verifier bug (real; fixed).** `migrate_task_to_task_md` renames
   `tests/`→`verifier/` but leaves `verifier/test.sh` referencing
   `/tests/test_outputs.py`. The native verifier dir mounts at **`/verifier`**, so
   pytest can't find the file at run time → verifier failure. It is invisible to static
   `check_task` (surfaces only on an oracle/agent run). `adapt_clean.py` rewrites
   `/tests/`→`/verifier/`.
   **Recommended core fix:** apply the same rewrite inside `migrate_task_to_task_md`'s
   dir-promotion step so every migrated legacy task is fixed class-wide.

2. **Empty-default scaffolding (cosmetic; pruned).** `model_dump_toml(by_alias=True)`
   emits `artifacts: []`, `reward: {}`, `oracle: {env: {}}`, every empty `env: {}`,
   `mcp_servers: []`, the deprecated `allow_internet`, and a dormant `verifier.judge`
   block (with a dangling `rubric_path: tests/rubric.toml`) even when
   `verifier.type == test-script`. `adapt_clean.py` drops these — **provably lossless**:
   it asserts the cleaned `task.md` reloads to the *identical* `TaskConfig` as the
   faithful migration. (Frontmatter shrank 78 → 62 lines on the sample task.)

3. **Over-permissive network default (authoring note).** Offline tasks inherit the
   schema default `network_mode: public`. `adapt_clean.py --offline-no-network` hardens
   that to `no-network` (least privilege); left opt-in to keep the migration faithful.

## Files

- `adapt.py` — faithful migration wrapper (`migrate_task_to_task_md` over a task set).
- `adapt_clean.py` — minimal/clean adapter: faithful migrate → lossless prune → verifier-path fix.
- `run_matrix.sh` — openhands/deepseek matrix runner (no-skill / with-skill / self-gen), smoke then full.
- `simple_tasks.txt` — offline, light task subset used for the smoke/run-through.

Run artifacts (`adapted*/`, `jobs/`, `sb-runs/`) are git-ignored.
