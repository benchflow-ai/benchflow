# SkillsBench → task.md run-through (openhands + DeepSeek v4)

Adapt simple SkillsBench tasks into native `task.md`, then run them on
`openhands` + DeepSeek v4 across the three skill modes (no-skill / with-skill /
self-gen), smoke first then full.

> **Why this is a local harness:** it could not run in the Claude-on-web
> container — that environment's network policy returns `403 Host not in
> allowlist` for `api.deepseek.com` and `app.daytona.io`, and has no docker
> daemon. The deterministic adaptation + `bench tasks check` rubrics *do* pass
> there; only the live agent matrix needs this branch run locally.

## 0 · Prereqs

```bash
uv sync --extra dev --extra sandbox-daytona      # daytona extra REQUIRED for --sandbox daytona
git clone --depth 1 https://github.com/benchflow-ai/skillsbench ../skillsbench

export DAYTONA_API_KEY=...        # and DAYTONA_API_URL if non-default
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com   # provider does NOT default this
```

Confirm the exact model id (the harness defaults to `deepseek/deepseek-v4`):

```bash
curl -s https://api.deepseek.com/models -H "Authorization: Bearer $DEEPSEEK_API_KEY" | jq '.data[].id'
# export MODEL=deepseek/deepseek-v4-flash   # or whatever id is listed
```

## 1 · Adapt (deterministic — no model, no sandbox)

```bash
python adapt.py --skillsbench ../skillsbench --out ./adapted --tasks-file simple_tasks.txt
# each: task.toml+instruction.md+solution/+tests/ -> task.md + oracle/ + verifier/ + environment/
# validated structurally; this same pipeline passes schema/structural/publication-grade
```

## 2 · Smoke (one task, prove a real rollout)

```bash
./run_matrix.sh smoke
# inspect jobs/sb-smoke-*/*/result.json — trust ONLY if:
#   n_tool_calls > 0  AND  total_tokens > 0  AND  reward is non-None
```

## 3 · Full matrix (3 skill modes)

```bash
./run_matrix.sh full
#   no-skill   = baseline
#   with-skill = skill mounted
#   self-gen   = agent generates the skill first (--self-gen-no-internet)
# compare per-task mean reward across the three modes (skill lift)
```

Overrides: `AGENT`, `MODEL`, `SANDBOX` (default `daytona`; use `docker` if you
have a local daemon), `TASKS_DIR`, `CONC`.

## Landmines (from the handoff runbook §5)

- **Resume trap:** `bench eval create` resumes a matching `--jobs-dir` and reuses
  stale results. The harness uses a fresh timestamped dir per batch — keep it.
- **Dead key = opaque error:** a revoked DeepSeek key surfaces only as
  `ACP error -32603` on the first model call. The harness verifies the key first.
- **`openhands`, not `opencode`** — opencode's litellm block never registers the
  proxy model, so `set_model` fails `model not found`.
- **Daytona 10 GB/sandbox** — keep tasks light (these are); heavy tasks overflow
  at bootstrap and hang silently at "Sandbox user agent ready".
- A run that "passed" instantly is probably resumed/oracle results — verify
  tokens/tools > 0.
