# Before you run an eval

The same model gets a different score with another agent, effort level, time limit, retry rule or machine. Before a run whose number you'll publish or compare, decide each of these settings on purpose, look at the plan BenchFlow prints, and write the settings down. A run you only want to try out needs none of this.

Runs here use MiMo (`-m xiaomi/mimo-v2.6-flash`): export its key and base URL first ([Quick start](../start/quick-start.md#on-another-provider)), or use `-m sonnet` with a Claude login.

## The settings that change a score

Each row has a case, published or from our own runs, where the setting moved a score, swapped two models' order, or never reached the model.

| Setting | What happened | Set it with |
|---|---|---|
| Reasoning effort | On Snorkel's OSWorld 2.0 leaderboard, Claude Opus 5 scored 44.33% at max effort and 36.89% at high (binary accuracy at 500 steps; [the board](https://snorkel.ai/leaderboard/os-world-2-0/), read 2026-11-04). | `-e` |
| Agent (harness) | In PaperBench, Claude 3.5 Sonnet beat o1 21.0% to 13.2% with one agent loop and lost 16.1% to 24.4% with another ([Tables 4 and 5](https://arxiv.org/abs/2504.01848)). | `-a`, `--harness` |
| Turn, time and cost limits | In the SWE-bench Pro paper, with 50 turns and a $2 cost cap, GPT-5 (high) led Claude Sonnet 4, 25.9% to 17.6%; in its main results, with no cost cap, Sonnet 4 led, 42.7% to 41.8% ([Tables 5 and 1](https://arxiv.org/abs/2509.16941)). | the task's `agent.timeout_sec`, `--timeout-multiplier`, `--max-cost-usd` |
| Output-token cap | BenchFlow starts every agent with `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000` in its environment, and Claude Code reads it as its limit on one response's output: a longer response ends the agent's turn, and the trial is scored on what it left ([Known pitfalls](../audit/pitfalls.md#settings-that-dont-reach-the-model)). | `--agent-env CLAUDE_CODE_MAX_OUTPUT_TOKENS=<n>` |
| Sandbox user | Agents run as a non-root `agent` user, which owns the working directory and its own home folder, nothing else: an agent that must install system packages or write outside that folder gets `Permission denied` and can score 0 for it ([Task format](../tasks/task-format.md#the-environment)). | `--sandbox-user` (`agent` by default; `none`, `null` or `root` runs the agent as root, `sandbox_user: null` in a run config), and the task's `WORKDIR` |
| Machine and sandbox | Terminal-Bench 2.0 scores moved 6 points between the most and least generous resource settings, and infrastructure errors fell from 5.8% to 0.5% ([Anthropic](https://www.anthropic.com/engineering/infrastructure-noise)). | `--sandbox`, the task's `sandbox:` resources |
| Task version | Terminal-Bench 2.1 changed 28 of 2.0's 89 tasks; most agent-model pairs scored higher, one by 12.1 points ([tbench.ai](https://www.tbench.ai/news/terminal-bench-2-1)). | the task set's version or commit |
| Retries | Retrying failed requests until they succeeded overstated MathArena scores by up to 5% ([MathArena](https://matharena.ai/retry/)). | `--retry-attempts` |
| The path to the model | Through OpenHands, BenchFlow 0.9.1 sent DeepSeek no effort at all: `-e max` was refused, `max` in the agent's own variable was rejected, and `high` was accepted but never sent ([Reproduce a published score](../audit/reproduce.md#match-them-in-a-run)). | the agent, and the provider prefix in `-m`; check the requests in `trajectory/llm_trajectory.jsonl` |

Tries per task matter too: one try per task gives a noisy number. [Compare runs](../audit/compare-runs.md) shows how many tries a difference needs.

## See the plan before it runs

`--dry-run` prints what a run would do and runs nothing:

```bash
bench run skillsbench/tasks \
  -i econ-detrending-correlation \
  -i crystallographic-wyckoff-position-analysis \
  -i offer-letter-generator \
  -a claude -m xiaomi/mimo-v2.6-flash --trials 3 -j 3 --dry-run
```

`skillsbench/` is a clone of SkillsBench at commit f860ecf0 ([how](./run-a-benchmark.md#run-three-tasks-three-times-each)).

```text
{
  "agent": "claude-agent-acp",
  "model": "xiaomi/mimo-v2.6-flash",
  "environment": "docker",
  "concurrency": 3,
  ...
  "verifier_sandbox": "task",
  "dataset": null,
  "jobs_dir": "jobs",
  "tasks": [
    "crystallographic-wyckoff-position-analysis",
    "econ-detrending-correlation",
    "offer-letter-generator"
  ],
  "rollouts": 3
}
```

In 0.9.2, `rollouts` is the number of trials in one pass over the tasks, one per task: 3 here, so `--trials 3` runs nine. Check the task list and the agent and model here; it costs nothing.

## What to write down

0.9.2 doesn't write these down for you. Keep them with the result:

| Setting | Where to read it |
|---|---|
| BenchFlow version | `bench --version`; every trial's `config.json` (`benchflow_version`) |
| Agent and its version | `-a`; each trial's `agent/install-stdout.txt` names the versions it installed, and `bench doctor` lists them under "Agent versions" before a run |
| Model id and provider | `-m`; `config.json` (`model`) |
| Effort | `-e`; `config.json` (`reasoning_effort`), and what each request sent in `trajectory/llm_trajectory.jsonl` |
| Sampling (temperature, top_p) | the agent's defaults: 0.9.2 has no flag for them, and Claude Code, Codex and OpenCode take none from BenchFlow (only the `deepagents` agent reads `BENCHFLOW_MODEL_TEMPERATURE` and `BENCHFLOW_MODEL_TOP_P`, passed with `--agent-env`); each request's values are in `trajectory/llm_trajectory.jsonl` |
| Tries per task, and how they're combined | `--trials`; see [Compare runs](../audit/compare-runs.md#choose-the-metric) |
| Concurrency and sandbox | `-j`, `--sandbox`; `summary.json` (`concurrency`, `environment`) |
| Time limits | each task's `agent.timeout_sec`, `--timeout-multiplier`, the idle limit (`summary.json`, `agent_idle_timeout_sec`) |
| Retries | `--retry-attempts` (default 2) |
| What the agent is asked, and how often | `--prompt`, `--extra-instruction`, `--loop-strategy` (`verify-retry` and `self-review` prompt the agent again), `--skill-mode` and `--config-override`; all are under `bench run --help-all`, and a run that uses any of them isn't a plain score |
| Network policy | each task's settings; see [Network and secrets](../tasks/network-and-secrets.md) |
| Task set and its version | the folder and its Git commit, or `-d name@version` |
| Controls | the oracle and nop jobs for the same task set ([Check a benchmark](../audit/check-a-benchmark.md)) |

Effort and sampling are in `llm_trajectory.jsonl` only for calls that went through BenchFlow's model proxy. A subscription login bypasses it, so a run on one records only what the agent reports, and its cost is the agent's own estimate (Claude Code's session log) ([Usage and telemetry](../scale/telemetry.md)); run a result you'll publish on an API key.

[Report a result](../audit/report.md) turns this into a write-up.

## The run card (0.10)

> Needs BenchFlow `0.10.0rc2`, a release candidate on PyPI that you install by its version ([Install](../start/install.md#install-the-010-prerelease)).

0.10.0rc2 prints a run card before every run, and with `--dry-run` it prints the card and stops. The same command as above:

```text
Run card (a dry run: nothing ran, and no job folder was made):
  Tasks:     3 from .../skillsbench/tasks, selected with -i crystallographic-wyckoff-position-analysis -i econ-detrending-correlation -i offer-letter-generator
  Agent:     claude-agent-acp, harness acp, installs @agentclientprotocol/claude-agent-acp@0.81.2 + @anthropic-ai/claude-code@2.1.316
  Model:     xiaomi/mimo-v2.6-flash; effort: the agent's default
  Sampling:  the agent's own; BenchFlow sets no temperature or top_p for agent trials and caps Claude Code's output at 128000 tokens (CLAUDE_CODE_MAX_OUTPUT_TOKENS), BenchFlow's default; each proxied request's sampling is in the trial's trajectory/llm_trajectory.jsonl
  Context:   Claude Code assumes 200000 tokens (Claude Code's assumed window for a model id it does not recognize); BenchFlow does not know the window the model serves
  Settings:  CLAUDE_CODE_MAX_OUTPUT_TOKENS 128000 (BenchFlow's default)
  Vendor:    no recommended setting on file for xiaomi/mimo-v2.6-flash: read its model card before you compare
  Trials:    9 in all, 3 per task (--trials 3 runs the task set 3 times; each repeat is a job of its own); 3 at a time on docker
  Retries:   up to 2 per trial on a retryable failure, which can come after the agent acted (an idle timeout, an ACP error, a verifier that never started); --retry-attempts 0 for none
  Limits:    agent 15 min per task; idle 10 min (BenchFlow's default); verifier 4 min to 15 min
  Network:   public (3 tasks)
  Grading:   each task's own verifier setting; the final check hardened (the agent stopped, its start-up hooks removed); no integrity audit (--integrity audit records and checks what the agent did)
  Controls:  local jobs: oracle none found: bench run .../skillsbench/tasks --agent oracle --jobs-dir oracle; nop none found: bench run .../skillsbench/tasks --agent nop --jobs-dir nop
  Time:      about 90 min if each trial runs once to its limits: 3 repeats one after another, each 3 trials 3 at a time (setup, image builds, retries and rubric reviews not counted)
  Budget:    none (--max-tokens caps a job's tokens; --max-cost-usd its cost, for a model BenchFlow has a price for; each repeat is a job of its own)
  Per trial: no token or turn budget (--token-budget and --turn-budget cap each trial's tokens and model calls, subagents' included)
```

Read it line by line. Here it says the effort and sampling are the agent's defaults under BenchFlow's output cap, that Claude Code assumes a window the model may not serve, that no recommended setting is on file for this model, that a retry can repeat the agent's work, that no oracle or nop run exists for these tasks yet (and the commands that make them), how long the run can take, and that nothing caps its cost. The card is also written to the job as `run-card.json`. A task set can declare the setup its runs must match, in `run-reference.toml`. With a strict reference, the card refuses a run that differs from it unless you pass `--accept-deviations` with your reason; see [Publish a benchmark](../tasks/publish.md#declare-the-setup-runs-should-match-010).

The `Per trial:` line under `Budget:` is each trial's [token and turn budget](../scale/cost.md#per-trial-budgets-010) (`--token-budget`, `--turn-budget`, or the tasks' own), counting every model call through BenchFlow's model proxy, subagents' and side calls included, or `no token or turn budget`. `run-card.json` keeps it as `trial_budget`. A run whose agent calls its model around the proxy cannot keep a budget, and the card refuses it.

After the run, the card adds what happened: `run-card.json`'s `after_run` block, and the same lines in the run's log. Besides the truncation rate and the settings the calls carried, it gives the score by the [scoring rule](../reference/scoring-rule.md) with its 95% interval over tasks; beside it, the trials that ended on a model refusal or a content filter (`Refusals:`, `0 of ...` when none did) and those a trial budget stopped (`Budgets:`, when any was), each with its rate; the trials with no score by who ended them; the rule's warnings (fewer than 3 trials per task, an error, timeout, rate-limit, model-refusal or budget-stop rate above 5%); the seeds (the task sample's, the rollout seeds, and the sampling seed asked for and carried); and the setting the trials ran, or `mixed settings` with one line per setting. With a trial budget, its `Per trial:` line says how many trials it stopped, by which measure, the median and the most tokens and turns a trial used, and the subagents' share of the tokens where the agents' own logs say.

### The run checks

Each trial's ending and model calls are checked against the card as the trial finishes, and `summary.json`'s `run_checks` keeps the figures. A check that fails the run makes `bench run` exit 1 with `Run check failed: ...`, then a `next:` line saying what to do (`run_checks.next_steps`). To run the tasks with another setting, start a new job, or run this one again with `--restart-changed`: a plain resume of a refused job starts nothing.

- **Truncation.** A trial is cut when the agent's last call ended at the output cap, its last turn ended `max_tokens`, or the agent's error says the cap was exceeded (Claude Code's "exceeded the 64000 output token maximum"). Such an ending is the model's under the run's cap: it is never retried, and the trial is scored as the agent left the workspace, the error kept on the record. The rate is over every attempt with a known ending, an attempt the retry policy ran again for another cause included. At 10% the run warns. At 50%, once five attempts with a known ending have finished, the job starts no more trials and the run fails, whatever the rate is by the time the running trials end. A job of fewer than five is too few to refuse on its own, though it still warns: the repeats of `--trials N` and the workers of a sharded run are judged together, and the loop stops the later repeats only when the repeats so far refuse the run.
- **Settings.** The model, effort, output cap, temperature, top_p and sampling seed the run asked for are compared with what the agent's main calls carried ([Model calls](../reference/model-calls.md)). A trial in which more than half the calls that carried a setting carried another value, or whose largest output cap is not the one asked for, stops the job right after it and fails the run. Fewer odd calls fail the run only when they are more than 1% of the job's calls that carried the setting (over every repeat or worker), judged once its trials have finished, so they never stop a job. These only warn: a setting no main call carried (the provider's default applied), a sampling seed the calls did not carry as asked, BenchFlow's own default output cap that the agent lowered, a model id that names a deployment or an alias (`azure/`, `azure_ai/`, `azure-foundry-openai/`, `azure-foundry-anthropic/`, `litellm/`, `litellm_proxy/`), whose served model is recorded instead, and an effort the model proxy's route sets on each request (Bedrock sets `high` for Claude 4.8), which is the effort that ran: the calls' own are then not compared.
