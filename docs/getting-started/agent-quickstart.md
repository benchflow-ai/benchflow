# Agent quickstart prompt

Have a coding agent set BenchFlow up for you. Copy the block below into Claude Code, Codex CLI, Gemini CLI or another coding agent, in an empty folder.

The agent installs BenchFlow and runs one benchmark task in a Docker sandbox. It then checks every file the run wrote and helps you write a task of your own. Expect 10 to 20 minutes, most of it the live run.

The prompt uses `<provider>/<model>` placeholders. It installs the latest stable BenchFlow from PyPI.

````text
You are setting up BenchFlow, an open-source tool that runs AI agents on tasks
in sandboxes and scores them. Run one real evaluation end to end. Work through
the steps in order. Show the user each command before you run it.

RULES (every step):
- Never print, echo, log or commit an API key. Write keys only into a local
  .env. In a git repo, make sure .env is in .gitignore.
- Give every `bench run` its own new --jobs-dir, retries included. Rerunning
  into the same jobs dir resumes the old job and keeps its finished tasks, so
  you would not see a new run. (`--fresh` also starts a new job.)
- `[FAIL]` with a reward like 0.4 (Score: 0/1) means the pipeline WORKED: the
  agent ran and the verifier scored it below the pass mark. A run is broken
  only if the checks in step 5 fail.
- Never delete a jobs folder or any result. For a clean slate, make a new
  folder.
- Don't use the `timeout` command (macOS lacks it). BenchFlow has its own
  timeouts. If a step prints nothing new for about 15 minutes, stop it and
  report what happened.

STEP 0: Preflight
- Check Docker: `docker info > /dev/null 2>&1`. If it fails, ask the user to
  start Docker, then check again. Don't continue without Docker.
- Check uv: `command -v uv`. If it is missing, install it with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` and put it on PATH.

STEP 1: Install BenchFlow
    uv tool install --python 3.12 benchflow
Keep `--python 3.12`. With Python 3.10 or 3.11, uv can pick an old benchflow
release with no `bench` command; it then fails with "No executables are
provided by package benchflow".
If uv says "Executable already exists: bench", run the same command again
with `--force`.
Use the installed `bench` on PATH. Don't use `uv run bench` from outside a
BenchFlow checkout: it can find a different `bench`.
Check:
    bench --version
    bench agents
    bench doctor
Show the user the version and the agent list. `bench doctor` prints one
PASS, WARN or FAIL line per check, each with a fix. Fix every FAIL except
`agent credentials`: that one stays a warning until step 3 sets a key, and is
expected now. Doctor names credentials but never prints them.

STEP 2: Fetch one sample task
SkillsBench is large, so download only one task:
    git clone --depth 1 --filter=blob:none --sparse https://github.com/benchflow-ai/skillsbench
    cd skillsbench && git sparse-checkout set tasks/citation-check && cd ..
Check that the task folder exists. Summarize its task.md instruction for the
user, so they know what the agent will be asked to do.
(`bench run benchflow-ai/skillsbench/tasks/<task>` fetches one task the same
way, into .cache/datasets-v2/, when you don't need to read it first.)

STEP 3: Set credentials
Ask the user which model provider to use. Have them put the keys in a local
.env, then load it:
    set -a; source .env; set +a
Variables must be EXPORTED: a `source .env` without `set -a` never reaches
`bench`.
For a model named `<provider>/<model>`, BenchFlow reads `<PROVIDER>_API_KEY`.
Providers with a user-supplied endpoint (glm, kimi, minimax, hunyuan, ...)
also need `<PROVIDER>_BASE_URL`. Providers with a default endpoint (openai,
anthropic, gemini, deepseek, zai, ...) need only the key. Example for GLM:
    export GLM_API_KEY=...                  # never echo this
    export GLM_BASE_URL=<the provider's OpenAI-compatible endpoint>
Without GLM_BASE_URL, the run stops with "Provider 'glm' for model ...
requires GLM_BASE_URL". That error is your hint, not a bug.
`bench agents` may list an agent such as openhands as needing LLM_API_KEY.
That is the agent's own setting name. BenchFlow fills it from your
provider-prefixed variables.

STEP 4: Run the task in a Docker sandbox
Use a new, timestamped jobs folder and one trial at a time:
    JOBS_DIR="jobs/quickstart-$(date +%Y%m%d-%H%M%S)"
    bench run skillsbench/tasks/citation-check \
      --agent openhands \
      --model <provider>/<model> \
      --sandbox docker \
      --concurrency 1 \
      --jobs-dir "$JOBS_DIR"
Use the user's chosen model. The first run builds the task image, so expect
several minutes. Stream the output to the user. Before any retry, set a new
$JOBS_DIR.

STEP 5: Check the run is REAL, then explain the score
Exit code 0 means the pipeline finished. It is NOT a pass or fail. Find the
trial's result:
    find "$JOBS_DIR" -name result.json
A run is REAL only if all three hold in that result.json:
  1. `n_tool_calls` > 0                (the agent acted)
  2. `agent_result.total_tokens` > 0   (real model traffic was recorded)
  3. `rewards` is present and not null (the verifier scored it)
Check them with this one line:
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); t=d.get('n_tool_calls') or 0; k=(d.get('agent_result') or {}).get('total_tokens') or 0; r=d.get('rewards'); print('n_tool_calls:',t,'total_tokens:',k,'rewards:',r); print('REAL run' if t>0 and k>0 and r else 'NOT a real run'); sys.exit(0 if t>0 and k>0 and r else 1)" "$(find "$JOBS_DIR" -name result.json | head -1)"
Then explain the score. `reward` (also in verifier/reward.txt) is the
verifier's raw value, 0.0 to 1.0. `Score: x/1` and [PASS]/[FAIL] count a task
as passed only at reward 1.0. So [FAIL] with reward 0.6 means everything
worked and the agent solved part of the task. If check 1, 2 or 3 fails, report
which one, with the `error` and `error_category` fields from result.json.
Don't report success.

STEP 6: Show the files
Print the trial folder's tree. Tell the user what each file is:
  $JOBS_DIR/<timestamp>/<task>__<hash8>/
    result.json                      the trial's summary: rewards, tool calls,
                                     tokens and cost, errors, timing
    config.json                      the trial's resolved settings (secrets
                                     left out)
    prompts.json                     the prompts sent to the agent
    rewards.jsonl                    the reward record
    timing.json                      time per phase
    agent/                           the agent's own logs
    trajectory/acp_trajectory.jsonl  the full agent trace: prompts, tool
                                     calls, outputs
    trajectory/llm_trajectory.jsonl  raw model requests and responses
    trainer/atif.json                the trajectory in ATIF (left out when the
                                     trajectory is empty)
    trainer/verifiers.jsonl, trainer/adp.jsonl, results.jsonl
                                     the same trial in trainer formats
    verifier/reward.txt              the raw verifier reward
    verifier/test-stdout.txt         the verifier's output (and ctrf.json when
                                     the tests write a CTRF report)
The job folder also has summary.json and job-wide results.jsonl,
verifiers.jsonl and adp.jsonl. If the trainer/ files are missing, check
`bench --version` and upgrade before you report a bug. `cost` in result.json
can be null for providers with a user-supplied endpoint: BenchFlow has no
prices for them. That is not a failed run. Show the user one or two lines of
acp_trajectory.jsonl.

STEP 7: Write a task and check it with the oracle
    bench new my-first-task
This creates tasks/my-first-task/ with task.md (YAML settings, then the
instruction), environment/Dockerfile, verifier/test.sh,
verifier/test_outputs.py, verifier/verifier.md, verifier/rubrics/ and
oracle/solve.sh. The scaffold fails until you edit it: test.sh writes reward
0.0, test_outputs.py has a failing placeholder test, and oracle/solve.sh exits
1. So an unedited task never passes by accident.
With the user:
- Replace the instruction placeholder in task.md with a small goal. Use an
  absolute path so the oracle and verifier agree (the agent works in /app),
  for example "create /app/hello.txt containing 'hello benchflow'".
- Make verifier/test.sh check exactly that and write 1.0 to
  /logs/verifier/reward.txt on success.
- Replace the placeholder test in verifier/test_outputs.py, or delete it.
- Replace the [REPLACE: ...] placeholders in verifier/verifier.md,
  verifier/rubrics/verifier.md and verifier/rubrics/verifier.toml.
  `bench check` rejects any placeholder left, these three included.
- Make oracle/solve.sh do the task.
Check it:
    bench check tasks/my-first-task
Fix what it flags. Then run the oracle. It runs oracle/solve.sh directly and
needs no model or key:
    bench run tasks/my-first-task --jobs-dir "jobs/oracle-$(date +%Y%m%d-%H%M%S)"
A correct task scores 1.0 with the oracle. Then run the empty control in a new
jobs folder:
    bench run tasks/my-first-task -a nop --jobs-dir "jobs/nop-$(date +%Y%m%d-%H%M%S)"
`nop` does nothing, so the verifier scores the untouched workspace. A sound
verifier gives it 0.0.

STEP 8: Report
Tell the user: the installed version; the run command; whether the run was
REAL (the three checks); the reward against the [PASS]/[FAIL] mark; the file
paths from step 6; and the oracle and nop results for their task. Suggest next
steps: `--concurrency N` for many tasks, `--task-skills` to give the agent a
task's own skills, `bench import` to convert task.toml + instruction.md tasks,
and the Quick start page (docs/getting-started/quick-start.md in the benchflow
repo). If a step failed, say which step, the command and the error. An honest
partial result beats a made-up success.
````

## Next steps

- Do the same steps yourself: [Quick start](./quick-start.md).
- Log in to an agent: [Auth](../agents/auth.md).
