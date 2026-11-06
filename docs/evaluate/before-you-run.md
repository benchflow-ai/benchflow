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

In 0.9.1, `rollouts` is the number of trials in one pass over the tasks, one per task: 3 here, so `--trials 3` runs nine. Check the task list and the agent and model here; it costs nothing.

## What to write down

0.9.1 doesn't write these down for you. Keep them with the result:

| Setting | Where to read it |
|---|---|
| BenchFlow version | `bench --version`; every trial's `config.json` (`benchflow_version`) |
| Agent and its version | `-a`; each trial's `agent/install-stdout.txt` names the versions it installed, and `bench doctor` lists them under "Agent versions" before a run |
| Model id and provider | `-m`; `config.json` (`model`) |
| Effort | `-e`; `config.json` (`reasoning_effort`), and what each request sent in `trajectory/llm_trajectory.jsonl` |
| Sampling (temperature, top_p) | the agent's defaults: 0.9.1 has no flag for them, and Claude Code, Codex and OpenCode take none from BenchFlow (only the `deepagents` agent reads `BENCHFLOW_MODEL_TEMPERATURE` and `BENCHFLOW_MODEL_TOP_P`, passed with `--agent-env`); each request's values are in `trajectory/llm_trajectory.jsonl` |
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

> Needs BenchFlow `0.10.0rc1`, a release candidate that isn't published yet, so you can't install it from PyPI ([what you can do](../start/install.md#install-the-010-prerelease)).

0.10.0rc1 prints a run card before every run, and with `--dry-run` it prints the card and stops. The same command as above, with one wrong line: 0.10.0rc1's card says BenchFlow sets no output cap, but it starts Claude Code with `CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000`, as the table above says, and later 0.10 builds print that cap on the card:

```text
Run card (a dry run: nothing ran, and no job folder was made):
  Tasks:     3 from .../skillsbench/tasks, selected with -i crystallographic-wyckoff-position-analysis -i econ-detrending-correlation -i offer-letter-generator
  Agent:     claude-agent-acp, harness acp, installs @agentclientprotocol/claude-agent-acp@0.81.2 + @anthropic-ai/claude-code@2.1.280
  Model:     xiaomi/mimo-v2.6-flash; effort: the agent's default
  Sampling:  the agent's own; BenchFlow sets no temperature, top_p or output cap for agent trials, and each proxied request's sampling is in the trial's trajectory/llm_trajectory.jsonl
  Vendor:    no recommended setting on file for xiaomi/mimo-v2.6-flash: read its model card before you compare
  Trials:    9 in all, 3 per task (--trials 3 runs the task set 3 times; each repeat is a job of its own); 3 at a time on docker
  Limits:    agent 15 min per task; idle 10 min; verifier 4 min to 15 min
  Network:   public (3 tasks)
  Grading:   each task's own verifier setting; the final check hardened (the agent stopped, its start-up hooks removed); no integrity audit (--integrity audit records and checks what the agent did)
  Controls:  oracle none found: bench run .../skillsbench/tasks --agent oracle --jobs-dir oracle; nop none found: bench run .../skillsbench/tasks --agent nop --jobs-dir nop
  Time:      about 90 min if each trial runs once to its limits: 3 repeats one after another, each 3 trials 3 at a time (setup, image builds, retries and rubric reviews not counted)
  Budget:    none (--max-tokens caps a job's tokens; --max-cost-usd its cost, for a model BenchFlow has a price for; each repeat is a job of its own)
```

Read it line by line. Here it says the effort and sampling are the agent's defaults, that no recommended setting is on file for this model, that no oracle or nop run exists for these tasks yet (and the commands that make them), how long the run can take, and that nothing caps its cost. The card is also written to the job as `run-card.json`. A task set can declare the setup its runs must match, in `run-reference.toml`. With a strict reference, the card refuses a run that differs from it unless you pass `--accept-deviations` with your reason; see [Publish a benchmark](../tasks/publish.md#declare-the-setup-runs-should-match-010).

After the run, the card adds what happened: `run-card.json`'s `after_run` block, and the same lines in the run's log. Besides the truncation rate and the settings the calls carried, later 0.10 builds give the score by the [scoring rule](../reference/scoring-rule.md) with its 95% interval over tasks; the trials with no score by who ended them; the rule's warnings (fewer than 3 trials per task, an error, timeout or rate-limit rate above 5%); the seeds (the task sample's, the rollout seeds, and the sampling seed asked for and carried); and the setting the trials ran, or `mixed settings` with one line per setting.
