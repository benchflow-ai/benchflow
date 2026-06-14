# Agent quickstart prompt

Want an AI coding agent to set BenchFlow up for you? Copy the entire block
below and paste it into your agent (Claude Code, Codex CLI, Gemini CLI, etc.)
in an empty working directory. It walks the agent through installing
benchflow, running one real benchmark task in a Docker sandbox, inspecting
every artifact the run produces, and authoring a task of your own — expect
roughly 10–20 minutes end to end, most of it the live eval.

The prompt uses generic `<provider>/<model>` placeholders; substitute any
supported provider. It requires benchflow 0.6.0 or newer.

````text
You are setting up benchflow, an open-source harness for benchmarking AI coding
agents in sandboxes, and running one real evaluation end to end. Work through
the numbered steps in order. Show the user each command before running it.

GUARDRAILS (apply to every step):
- Never print, echo, log, or commit API keys. Never write keys into any file
  except a local .env (if the working directory is a git repo, confirm .env
  is gitignored; never commit it anywhere).
- Every `bench eval create` invocation — including retries — must use a FRESH
  --jobs-dir. Reusing a jobs dir triggers resume logic and skips the run.
- A `[FAIL]` line with a fractional reward (e.g. reward 0.4, Score: 0/1) means
  the pipeline is HEALTHY: the agent ran and the verifier scored it below the
  pass threshold. Only treat a run as broken if step 5's checks fail.
- Never delete a jobs directory or any results. If you need a clean slate,
  make a new directory.
- Do not use the `timeout` command (absent on macOS). benchflow has its own
  agent/verifier timeouts. If any single step stalls with no new output for
  ~15 minutes, stop it and report what happened instead of waiting forever.

STEP 0 — Preflight
- Check Docker: `docker info > /dev/null 2>&1`. If it fails, ask the user to
  start Docker Desktop / the Docker daemon, then re-check. Do not continue
  without it: benchflow has no up-front daemon check, so a dead daemon fails
  the run partway through instead of at startup.
- Check uv: `command -v uv`. If missing, install it:
  `curl -LsSf https://astral.sh/uv/install.sh | sh` and ensure it is on PATH.
- Python 3.12+ is required; uv will provision it if needed.

STEP 1 — Install benchflow
This prompt requires benchflow 0.6.0 or newer (the task.md authoring CLI and
trainer artifacts shipped in 0.6.0). Install it from PyPI:
    uv tool install benchflow
benchflow pins a stable litellm (no `--prerelease` flag needed). If uv reports
"Executables already exist: bench, benchflow", rerun the same command with
`--force`. Confirm with `bench --version` after install.
If the installed version is still older than
0.6.0, CONTINUE anyway in degraded mode: steps 0–6 work on 0.5.x too. Tell
the user which version you got, then (a) in step 6, expect only
`trainer/verifiers.jsonl` (the `atif.json`/`adp.jsonl` trainer artifacts
ship in 0.6.0), and (b) in step 7, scaffold with plain `bench tasks init
my-first-task` (the `--format task-md` flag and `task.md` scaffold are
0.6.0+; older builds emit the split `task.toml` layout — the oracle run
works the same). Note in your final summary that upgrading to 0.6.0
unlocks the full flow.
TRAP — this quickstart uses the tool-installed `bench` (on PATH after
`uv tool install`); if you instead invoke benchflow as `uv run bench …` from a
benchflow source checkout, run it from INSIDE that project directory, because
`uv run bench` launched from outside the repo can resolve a different or legacy
`bench`.
Verify:
    bench --version
    bench agent list
Show the user the version and the agent list.

STEP 2 — Fetch one sample task (sparse checkout, not a full clone)
The skillsbench repo is large; download only one task:
    git clone --depth 1 --filter=blob:none --sparse https://github.com/benchflow-ai/skillsbench
    cd skillsbench && git sparse-checkout set tasks/tictoc-unnecessary-abort-detection && cd ..
Confirm the task directory exists and briefly summarize its instruction.md to
the user so they know what the benchmark agent will be asked to do.

STEP 3 — Set credentials
Ask the user which model provider to use, then have them put keys in a local
.env and load it with the export-all pattern:
    set -a; source .env; set +a
or export explicitly. If the working directory is a git repo, first confirm
.env is gitignored (add it to .gitignore if not); in a fresh non-git
directory there is nothing to check — just keep the file local and never
print its contents. Provider-prefix rule, in one line: for a model named
`<provider>/<model>` with a user-supplied endpoint (deepseek, glm, kimi,
minimax, hunyuan, ...), benchflow reads `<PROVIDER>_API_KEY` plus
`<PROVIDER>_BASE_URL`; fixed-endpoint providers (openai, anthropic, gemini,
zai, ...) need only the API key. Example for a deepseek-hosted model:
    export DEEPSEEK_API_KEY=...                       # never echo this
    export DEEPSEEK_BASE_URL=https://api.deepseek.com
Variables must be EXPORTED (a plain `source .env` without `set -a` never
reaches the bench process). If the base URL is missing, the run fails with an
explicit "requires DEEPSEEK_BASE_URL" error — that is your hint, not a bug.
Note: `bench agent list` may show an agent (e.g. openhands) as requiring
LLM_API_KEY. That is the agent's internal setting name, not a variable you
set: benchflow maps your provider-prefixed variables to the agent's LLM_*
settings automatically.

STEP 4 — Run the eval in a Docker sandbox
Use a fresh, timestamped jobs dir and concurrency 1:
    JOBS_DIR="jobs/quickstart-$(date +%Y%m%d-%H%M%S)"
    bench eval create \
      --tasks-dir skillsbench/tasks/tictoc-unnecessary-abort-detection \
      --agent openhands \
      --model <provider>/<model> \
      --sandbox docker \
      --concurrency 1 \
      --jobs-dir "$JOBS_DIR"
Substitute the user's chosen model (e.g. an openhands-compatible
provider-prefixed model). First run pulls/builds the task image, so expect
several minutes. Stream the output to the user. If you retry for any reason,
mint a NEW $JOBS_DIR first.

STEP 5 — Verify the run is REAL, then explain the score
Exit code 0 only means the pipeline completed — it is NOT pass/fail. Find the
rollout result:
    find "$JOBS_DIR" -name result.json
A run counts as REAL only if ALL THREE hold in the rollout's result.json:
  1. `n_tool_calls` > 0          (the agent actually acted)
  2. `agent_result.total_tokens` > 0   (real model traffic was captured)
  3. `rewards` is present and its value is not null (the verifier scored it)
Check them with a one-liner (kept on one line so indentation cannot break it):
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); t=d.get('n_tool_calls') or 0; k=(d.get('agent_result') or {}).get('total_tokens') or 0; r=d.get('rewards'); print('n_tool_calls:',t,'total_tokens:',k,'rewards:',r); print('REAL run' if t>0 and k>0 and r else 'NOT a real run'); sys.exit(0 if t>0 and k>0 and r else 1)" "$(find "$JOBS_DIR" -name result.json | head -1)"
Then explain the semantics to the user: `reward` (also in
verifier/reward.txt) is the raw verifier value 0.0–1.0; `Score: x/1` and
`[PASS]/[FAIL]` apply the pass threshold, where only reward 1.0 counts as a
pass. So `[FAIL]` with reward 0.6 means everything worked and the benchmarked
agent partially solved the task. If a check in 1–3 fails, report which one and
the error/error_category fields from result.json instead of declaring success.

STEP 6 — Showcase the artifacts
Print the rollout directory tree and tell the user where each artifact landed
and what it is:
  $JOBS_DIR/<timestamp>/<task>__<hash8>/
    result.json                      — rollout summary: rewards, tool calls,
                                       token usage/cost, errors, timing
    config.json                      — the rollout's resolved configuration
                                       (secret-bearing env vars filtered out)
    prompts.json                     — the prompts sent to the agent
    rewards.jsonl                    — reward record for this rollout
    timing.json                      — per-phase timing breakdown
    agent/                           — agent-side logs
    trajectory/acp_trajectory.jsonl  — the full agent trace (every ACP event:
                                       prompts, tool calls, outputs)
    trajectory/llm_trajectory.jsonl  — raw provider requests/responses captured
                                       by the usage-tracking proxy
    trainer/verifiers.jsonl          — trainer-ready scored trajectory record
    trainer/atif.json                — the trajectory in ATIF interchange
                                       format (omitted if the trajectory is empty)
    trainer/adp.jsonl                — the trajectory in ADP format
    verifier/reward.txt              — raw verifier reward
    verifier/test-stdout.txt         — verifier stdout (and ctrf.json when the
                                       test emits a CTRF report)
Also note the job-level summary.json and the aggregated verifiers.jsonl /
adp.jsonl in the job directory. The trainer/ files and the job-level
aggregates are written by benchflow 0.6.0+; if they are missing, re-check
`bench --version` before reporting a bug. `cost` in result.json can be null
for user-endpoint providers (cost telemetry is unavailable for them) — that
is a telemetry gap, not a failed run. Show the user one or two sample lines
from acp_trajectory.jsonl so they see what a trace looks like.

STEP 7 — Author your own task and verify it with the oracle
Scaffold a task (in 0.6.0+ `bench tasks init` defaults to the unified
task.md format):
    bench tasks init my-first-task
This creates tasks/my-first-task/ with task.md (YAML frontmatter + prompt
body), environment/Dockerfile, verifier/test.sh, verifier/test_outputs.py,
verifier/verifier.md, verifier/rubrics/, and oracle/solve.sh. The scaffold
deliberately fails until edited — test.sh writes reward 0.0,
test_outputs.py contains a failing placeholder test, and oracle/solve.sh
exits 1 — so an unedited task can never pass by accident. With the user:
replace the prompt placeholder in task.md with a small concrete goal — use
an ABSOLUTE path so the oracle and verifier agree on location (the agent
workspace is `/app`), e.g. "create `/app/hello.txt` containing
'hello benchflow'"; make verifier/test.sh check exactly that (read
`/app/hello.txt`) and write 1.0 to /logs/verifier/reward.txt on success;
replace the placeholder assertion in verifier/test_outputs.py (or delete
that file); replace the `[REPLACE: ...]` placeholders in the three verifier
description files the scaffold also writes — verifier/verifier.md,
verifier/rubrics/verifier.md, and verifier/rubrics/verifier.toml (`bench
tasks check` rejects any unreplaced placeholder, these three included, so
skipping them fails the check below); and make oracle/solve.sh perform the
task. Validate:
    bench tasks check tasks/my-first-task
Fix anything it flags (it rejects unreplaced [REPLACE: ...] placeholders).
Then run the task with the built-in oracle agent — it executes
oracle/solve.sh directly, needs no model or API key, and proves the task +
verifier loop:
    bench eval create \
      --tasks-dir tasks/my-first-task \
      --agent oracle \
      --sandbox docker \
      --jobs-dir "jobs/oracle-$(date +%Y%m%d-%H%M%S)"
If the console prints "Unknown agent 'oracle' — not in registry ... Will
attempt to use as raw command" (older benchflow builds do this), the warning
is EXPECTED and the run is still a real oracle run — do not classify it as a
broken step. A correct task scores reward 1.0 with the oracle.

STEP 8 — Report
Summarize for the user: installed version; the eval command used; whether the
run was REAL (the three checks); the reward vs. the [PASS]/[FAIL] threshold
reading; the artifact paths from step 6; the oracle result for their own
task; and suggested next steps (`--concurrency N` for batches,
`--skill-mode with-skill` for skill evals, `bench tasks migrate` to convert
legacy split-layout tasks to task.md — available in 0.6.0+ —
and docs/getting-started.md in the benchflow repo). If any step failed,
report exactly which step, the command, and the error — partial honest
results beat a fabricated success.
````
