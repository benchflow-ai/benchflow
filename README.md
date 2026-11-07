<div align="center">
  <h1>BenchFlow</h1>
  <p>Run AI agents on benchmark tasks in sandboxes, and score them.</p>
  <a href="https://pypi.org/project/benchflow/" target="_blank">
    <img src="https://img.shields.io/badge/PyPI-benchflow-3775A9?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI package">
  </a>
  <a href="https://github.com/benchflow-ai/benchflow" target="_blank">
    <img src="https://img.shields.io/github/stars/benchflow-ai/benchflow?style=for-the-badge&logo=github&logoColor=white&label=Stars&color=181717" alt="GitHub stars">
  </a>
  <a href="https://discord.gg/mZ9Rc8q8W3" target="_blank">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FmZ9Rc8q8W3%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=Discord&suffix=%20members&logo=discord&logoColor=white&color=5865F2&style=for-the-badge" alt="Discord members">
  </a>
</div>

BenchFlow runs AI agents on tasks in sandboxes and scores them. Point it at a benchmark's tasks and pick an agent: Claude Code, Codex, Gemini CLI, OpenHands and more, over [ACP](https://agentclientprotocol.com). Each run records its reward, cost and full trajectory.

Tasks run on Docker (local or remote), Daytona, Modal or AgentCore. The same runtime serves evaluation and RL training.

These docs describe BenchFlow 0.10. `bench --version` prints the version you have; [What's new in 0.9](./docs/versions/upgrade-from-0.7.md) lists what changed since 0.7, and the [changelog](./CHANGELOG.md) what 0.10 adds.

## Quick start

You need [Docker](https://docs.docker.com/get-started/get-docker/) running and [uv](https://docs.astral.sh/uv/getting-started/installation/). Steps 1 and 2 need no model and no API key.

**1. Install BenchFlow and check this machine.**

```bash
uv tool install --python 3.12 benchflow
bench doctor    # needs 0.8 or newer; checks Docker, agent logins and network, with a fix for each problem
```

The examples use `bench`; the installed `benchflow` command is the same CLI.

Before you log in to an agent, `bench doctor` reports that no agent can run. That is expected: step 2 needs no login.

**2. Run a task with the oracle.** The oracle runs the task's own reference solution. It tests the sandbox, the task and its verifier with no model:

```bash
bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation \
  --agent oracle --sandbox docker --jobs-dir jobs/oracle
```

BenchFlow downloads only that task from [SkillsBench](https://github.com/benchflow-ai/skillsbench), builds its image, runs its solution and verifier, and prints `Score: 1/1`. A cold-cache first run took 6 minutes 12 seconds, including the verifier's package downloads; allow longer on a slow connection.

**3. Run a model.** With a Claude subscription, no API key is needed. `claude setup-token` comes with [Claude Code](https://code.claude.com/docs/en/quickstart) and prints a login token that lasts a year:

```bash
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN='<token>'
bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation \
  --agent claude --model claude-haiku-4-5-20251001 --sandbox docker --jobs-dir jobs/claude
```

> **Billing:** an exported `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` wins over the token and is billed. Unset both to use your subscription.

With an API key, export it and name a provider-prefixed model. For a DeepSeek key:

```bash
export DEEPSEEK_API_KEY='<key>'
bench run benchflow-ai/skillsbench/tasks/econ-detrending-correlation \
  --agent openhands --model deepseek/deepseek-flash --sandbox docker --jobs-dir jobs/model
```

The agent may pass or fail. Either way, the run worked. Version 0.9.0 of `bench doctor` warns that `deepseek/` also needs `DEEPSEEK_BASE_URL`; it does not, so the run goes ahead without it. [Quick start: run a model](./docs/start/quick-start.md) has the agent and model to use for OpenAI, Anthropic, Gemini and other keys, and for a ChatGPT plan.

**4. Read the results.**

```bash
bench results jobs/              # one row per job, named by its folder, with its score
bench eval metrics jobs/claude   # pass rate, solve rate, pass@k, tool calls and time
bench view jobs/claude           # trajectory, verifier output and score, in your browser
```

`jobs/claude` is a jobs directory; each timestamped child is one job. Copy the exact `Artifacts:` path printed by `bench run` to inspect or resume that job with `bench results <job>` or `bench eval resume <job>`. If you skipped the model run, use `jobs/oracle` instead.

Each run writes its rewards, token usage and trajectory under `--jobs-dir`. A trial's reward is in `verifier/reward.txt` and in `result.json` under `rewards.reward`; its transcript is in `trajectory/`. Run the same command again to resume the latest matching job: finished tasks are kept, not rerun. Add `--fresh` to start over.

## Next steps

- [Write and test your own task](./docs/tasks/write-a-task.md): `bench new my-task`, then the oracle and a do-nothing agent.
- [Run several tasks at once on Daytona](./docs/scale/daytona.md), a cloud sandbox.
- [Export runs for training](./docs/train/export.md): rewards and rollouts as JSONL.
- Run a whole benchmark: [Quick start](./docs/start/quick-start.md).
- Check every agent login at once: `bench eval smoke` runs a bundled hello-world task once per logged-in agent.
- [Run from Python](./docs/start/quick-start.md): [`quickstart.py`](./docs/examples/python-sdk/quickstart.py) runs a task with the oracle, with no key.
- Learn the words: [Core concepts](./docs/start/how-it-works.md).
- Upgrading from 0.7? Read [What's new in 0.9](./docs/versions/upgrade-from-0.7.md).

## Documentation

To have a coding agent do the quick start for you, paste the [agent quickstart prompt](./docs/start/quick-start.md) into Claude Code, Codex CLI or Gemini CLI.

### Start and run

| If you want to… | Read |
|------------------|------|
| Run your first task, then a model | [Quick start](./docs/start/quick-start.md) |
| Install, upgrade or uninstall | [Install](./docs/start/install.md) |
| Keep the Rust `bench` beside this one | [Which bench runs](./docs/start/install.md) |
| Log in to an agent, or use an API key | [Auth](./docs/evaluate/agents.md#log-in-or-set-a-key) |
| Know what a task, dataset, agent, sandbox, verifier, trial and job are | [Core concepts](./docs/start/how-it-works.md) |
| Run any benchmark, including ones with their own task format | [Run any benchmark](./docs/evaluate/bring-what-you-have.md) |
| Find out why a run failed, whose problem it is, and what to do | [When a run fails](docs/evaluate/failed-trials.md) |
| Resume an interrupted job | [Resume a job](./docs/scale/resume.md) |
| Run an agent from the public agents repo (goose, qwen-code, prime-agent, …) | [Running external agents](docs/evaluate/your-own-agent.md#agents-from-the-agents-repository) |
| Run Claude Code or Codex through their own CLIs instead of their ACP adapters | [Native harness](docs/evaluate/agents.md#run-claude-code-or-codex-through-their-own-cli) |
| Run a hosted PrimeIntellect / Verifiers environment | [CLI reference](./docs/reference/commands.md) |
| Run tasks on a Docker host you control | [Remote Docker](docs/scale/sandboxes.md#remote-docker) |
| Keep BenchFlow loaded between CLI calls, and prepare the next sandbox | [Local runtime](./docs/scale/sandboxes.md) |
| Cap a job's spend, sandbox time, tokens or rollouts | [Budgets](./docs/scale/cost.md#budgets) |

### Several agents, several rounds, branching

| If you want to… | Read |
|------------------|------|
| Run several agents in one task: coder and reviewer, simulated user | [Scenes and simulated users](./docs/tasks/multi-turn.md) |
| Give one agent hints over several rounds | [Progressive disclosure](docs/tasks/multi-turn.md#a-scripted-user) |
| Fork a run at a checkpoint into scored children | [Branching](docs/train/environment.md) |
| Evaluate a skill rather than a workspace | [Skill eval](docs/evaluate/skills.md) |
| Run robots and simulators | [Embodied rollouts](docs/tasks/task-format.md#other-task-formats) |

### Scores you can trust

| If you want to… | Read |
|------------------|------|
| Understand the security model | [Sandbox hardening](docs/tasks/reward-hacking.md) |
| Score in a sandbox that holds only the agent's final workspace and declared artifacts | [Separate verifier sandboxes](docs/tasks/verifiers.md#run-the-verifier-in-its-own-sandbox) |
| Check what the agent did to the things its reward depends on (BenchShield) | [Reward integrity](./docs/tasks/reward-hacking.md#audit-how-each-reward-was-earned) |
| Re-score stored runs after fixing a verifier, without running the agent again | [Regrade](docs/scale/resume.md#regrade-saved-work) |

### Tasks

| If you want to… | Read |
|------------------|------|
| Write a new task | [Task authoring](docs/tasks/task-format.md) |
| Write a task in the native `task.md` format | [task.md authoring](docs/tasks/task-format.md) |
| Run task.md draft 2 packages | [task.md draft 2 packages](docs/tasks/task-format.md) |
| Run a benchmark's own task folders with no conversion | [Task formats](./docs/tasks/task-format.md) |

### Results and training

| If you want to… | Read |
|------------------|------|
| Read, compare and export runs in Python or a notebook | [Analysing runs](./docs/evaluate/read-results.md) |
| Turn runs into rows a trainer reads (JSONL with rewards and messages) | [Export runs for training](./docs/train/export.md) |
| Stream rollouts to a trainer while a job runs | [Streaming rollouts](./docs/train/trainer.md#stream-rollouts-while-a-job-runs) |
| Capture token ids and logprobs for training | [Token capture](./docs/train/trainer.md#token-ids-and-logprobs) |
| Share a coding-agent session or trial | [Share a trajectory](./docs/audit/report.md#share-trajectories) |

### Reference

| If you want to… | Read |
|------------------|------|
| Every command and flag, generated from the CLI | [Command reference](./docs/reference/commands.md) |
| Worked CLI examples: run configs, resuming, reading the score | [CLI guide](./docs/reference/commands.md) |
| Every public Python name, generated from the package | [Python reference](./docs/reference/python.md) |
| Worked Python examples | [Python API guide](./docs/reference/python.md) |
| The JSON documents BenchFlow writes, generated from their schemas | [Schema reference](./docs/reference/files.md) |
| Stable and preview releases | [Install: stable and preview releases](./docs/start/install.md) |

Runnable scripts and notebooks are in [`docs/examples/`](./docs/examples/).

## Run a benchmark from a Git repository

A run config names the repository and the folder of tasks in it, then the agent and model:

```yaml
# benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
source:
  repo: benchflow-ai/benchmarks
  path: datasets/harvey-lab/tasks

agent: gemini
model: gemini/gemini-3.5-flash-lite
environment: docker
concurrency: 4
```

```bash
# From the config, in a clone of this repository (benchmarks/ is not in the installed package)
bench run --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml

# The same source, inline
bench run benchflow-ai/benchmarks/datasets/harvey-lab/tasks \
    --agent gemini --model gemini/gemini-3.5-flash-lite --sandbox docker --concurrency 4
```

BenchFlow caches repositories in `.cache/datasets-v2/`, at the root of the Git repository you run from, or in the current folder outside one. BenchFlow writes a `.gitignore` there, so the cache never shows in `git status`. With a folder in the target, it downloads only that folder. A target with no folder, or a folder with no BenchFlow task in it, downloads the whole repository.

## Write a task

A task is a folder: a `task.md` (YAML settings on top, the instruction below), an `environment/` folder and a `verifier/` folder. Four commands cover a task's life:

```bash
bench new my-task                          # scaffold tasks/my-task
bench check tasks/my-task                  # validate it (default: --level structural)
bench import legacy-task/ --remove-legacy  # convert a task.toml + instruction.md task
bench export tasks/my-task out/            # write the split layout, with a loss report
```

`bench new` leaves placeholders for the task author; `bench check` reports them until you fill in the instruction, oracle and verifier. The import command assumes an existing `legacy-task/` folder in the split format.

BenchFlow also runs older split tasks (`task.toml`, `instruction.md`, `solution/`, `tests/`), [task.md draft 2](docs/tasks/task-format.md) packages, and any [task format](./docs/tasks/task-format.md) a benchmark registers. See [task.md authoring](docs/tasks/task-format.md) and the [task standard](docs/tasks/task-format.md).

## Contributing

For a source build, prepare the pinned `bfx` binaries first; see [Build from a checkout](./docs/start/install.md). Without them, Docker uses the fallback root steps.

Open pull requests against `main`. CI runs ruff (lint and format), `ty`, the unit tests, a Docker integration tier and `pip-audit`. Run the same checks before you push:

```bash
uv sync --extra dev --extra sandbox-daytona --extra sandbox-modal
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run ty check
uv run python -m pytest tests/ -k 'not cloud_metadata'
```

A checkout has no `bfx` binaries (BenchFlow's helper for root's steps on Docker; they are not in git). An editable install runs without them, as with `BENCHFLOW_BFX_DELIVERY=off`. To run Docker trials with `bfx`, or before you build a wheel, build them on Linux with `rustup` installed: `BFX_REQUIRE_PIN=1 scripts/build_bfx.sh` (or `make bfx`). See [Try a pre-release](./docs/start/install.md).

Merges to `main` publish an internal preview once CI passes. A release tag publishes the public release. See [Stable and preview releases](./docs/start/install.md); maintainers, see [Publishing a release](./docs/contributing/release.md).

## License

Apache-2.0.
