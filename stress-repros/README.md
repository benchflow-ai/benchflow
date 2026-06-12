# Stress-test repros — benchflow v0.6.0

Runnable reproductions for the confirmed defects in [`../STRESS-TEST-v0.6.0.md`](../STRESS-TEST-v0.6.0.md).

## Setup

These scripts read API keys from an env file kept **outside** the repo. Create it (do not commit):

```bash
cat > /tmp/bf-stress-env.sh <<'EOF'
export DEEPSEEK_API_KEY=...        # your key
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DAYTONA_API_KEY=...
export GEMINI_API_KEY=...
export GOOGLE_API_KEY=$GEMINI_API_KEY
EOF
chmod 600 /tmp/bf-stress-env.sh
```

Most scripts also need a staged 1-task dir:

```bash
mkdir -p /tmp/bf-stress/oracle_smoke_tasks
cp -R docs/examples/task-md/real-skillsbench/weighted-gdp-calc /tmp/bf-stress/oracle_smoke_tasks/
```

Run any script from the repo root: `bash stress-repros/<name>.sh`. Each prints **EXPECTED vs ACTUAL**.

| Script | Defect | Sev |
|---|---|---|
| `p1_concurrency_zero_deadlock.sh` | `--concurrency 0` hangs forever | P1 |
| `p1_modal_missing_extra_error.sh` | `--sandbox modal` raw `ModuleNotFoundError` | P1 |
| `p2_agent_timeout_negative.sh` | `agent.timeout_sec` accepts `-5`/`0` | P2 |
| `p2_reasoning_effort_unvalidated.sh` | `--reasoning-effort banana` accepted | P2 |
| `p2_skill_mode_traceback.sh` | `--skill-mode bogus` → traceback | P2 |
| `p2_agent_no_model_traceback.sh` | `--agent codex` no `--model` → traceback | P2 |
| `p2_judge_exit0_on_infra_failure.sh` | example `judge.py` exits 0 on infra failure | P2 |
| `p3_tasks_dir_nonexistent_traceback.sh` | bad `--tasks-dir` → traceback | P3 |
| `p2_3d_scan_calc_broken_oracle.sh` | shipped oracle can't self-pass (skill not injected) | P2 |
