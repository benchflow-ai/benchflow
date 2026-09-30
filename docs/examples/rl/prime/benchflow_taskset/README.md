# benchflow-taskset

BenchFlow tasks as a [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) v1 taskset, with the env that runs them, for evaluation and for RL training with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

Each episode starts the task's BenchFlow sandbox (on Daytona by default), gives the model two tools, `run_bash` and `submit`, and scores the episode with the task's own BenchFlow verifier. It is the tool shape of BenchFlow's TRL integration.

## How it runs

BenchFlow and Verifiers v1 cannot be installed in one Python environment: BenchFlow's `litellm[proxy]` requires `mcp<2`, and Verifiers requires `mcp==2.0.0`. So this package, installed next to Verifiers, never imports BenchFlow. Each episode starts `bridge.py` with the Python of a separate BenchFlow venv (`$BENCHFLOW_PYTHON`), and that process owns the sandbox through BenchFlow's `TaskRuntime`.

- `BenchFlowEnv` holds the sandbox for the whole episode: it starts it, serves `run_bash` and `submit` over MCP from its own process, runs the agent with them, and closes the sandbox whatever happens. The task's reward runs the verifier while the sandbox is still alive.
- The agent's seat is the `null` harness (a chat loop whose only tools are the MCP ones) in the `subprocess` runtime. Harnesses that execute commands themselves are refused, since those commands would run on this machine instead of in the sandbox.
- `submit(answer)` writes a non-empty answer to `/workdir/answer.txt` (`--env.taskset.task.submit-path`) and ends the episode.

## Rewards and drops

A failure counts as infrastructure only when the policy could not have caused it: the sandbox never started, the bridge process failed, the model endpoint failed, or the verifier crashed on a sandbox the policy never touched. Such an episode is marked failed, so prime-rl leaves it out of the batch and counts it (`errored_rollouts`). Every other failure scores 0, timeouts included: the per-command limit, `max_turns`, and the wall-clock `agent_budget_sec` all end the episode normally, and the verifier scores what the policy left. The decision itself comes from `benchflow.integrations.rewards` in the BenchFlow venv. Every episode appends one line to `<jobs_dir>/outcomes.jsonl` with its reward or drop reason.

## Use

```bash
# A BenchFlow venv (with the Daytona extra), and this package next to Verifiers
export BENCHFLOW_PYTHON=/path/to/benchflow/.venv/bin/python
uv pip install --no-deps -e .                 # into the venv that has verifiers (and prime-rl)
export DAYTONA_API_KEY=... BENCHFLOW_DAYTONA_OWNER=my-run

uv run vf-eval benchflow-taskset \
  --env.taskset.tasks-dir /path/to/tasks \
  --model Qwen/Qwen3-4B-Instruct-2507 --client.base-url http://localhost:8000/v1
```

Knobs live under `--env.taskset.*` (which tasks) and `--env.taskset.task.*` (how they run): `sandbox`, `sandbox_user`, `bash_timeout_sec` (60), `max_output_chars` (4096), `submit_path`, `agent_budget_sec` (900), `max_sandboxes` (16, across every env-server worker on the machine), `jobs_dir`, `outcomes_path`. The seat's `max_turns` defaults to 30 (`--env.agent.max-turns`).
