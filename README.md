<div align="center">
  <h1>BenchFlow</h1>
  <p>The universal environment framework — a benchmark is just a frozen environment.</p>
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

## What

BenchFlow runs AI agents on tasks in sandboxes and scores them, with one hardened runtime for evaluation and training. **A benchmark is just a frozen environment**: point BenchFlow at a benchmark's tasks, drive them with any [ACP](https://agentclientprotocol.com) agent (Claude Code, Codex, Gemini CLI, OpenHands and more), and read rewards, costs and full trajectories from every run. The same tasks run single-agent, multi-agent or multi-round, on Docker (local or remote), Daytona or Modal.

## Quick start

You need [Docker](https://docs.docker.com/get-started/get-docker/) running and [uv](https://docs.astral.sh/uv/getting-started/installation/). Steps 1 and 2 need no model and no API key.

**1. Install BenchFlow and check the machine.**

```bash
uv tool install --python 3.12 --upgrade benchflow
bench doctor    # Docker, agent logins, network: one PASS/WARN/FAIL line each, with a fix
```

Until you log in to an agent, `bench doctor` reports that no agent can run. The oracle in step 2 needs no login.

**2. Run a benchmark task with the oracle.** The oracle agent runs the task's own reference solution, so this checks the sandbox, the task and its verifier without a model:

```bash
bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks/citation-check \
  --agent oracle --sandbox docker --jobs-dir jobs/oracle
```

BenchFlow downloads only that task from the [SkillsBench](https://github.com/benchflow-ai/skillsbench) repository (a few megabytes), builds its image (a couple of minutes the first time) and ends with `Score: 1/1`.

**3. Run an agent on your Claude subscription.** No API key is needed. `claude setup-token` comes with [Claude Code](https://code.claude.com/docs/en/quickstart) and prints a login token that lasts a year:

```bash
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN='<token>'
bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks/citation-check \
  --agent claude --model claude-haiku-4-5-20251001 --sandbox docker --jobs-dir jobs/claude
```

The agent may pass or fail; either way the evaluation completed. An exported `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` wins over the token and is billed, so unset them to use the subscription. `bench eval smoke` runs a bundled hello-world task once with every agent you are logged in to, which checks each login in about a minute. For a ChatGPT subscription, run `codex login` and use `--agent codex --model <model>` with a model your plan offers; see [Auth](./docs/getting-started.md#auth-oauth-long-lived-token-or-api-key) for API keys and other agents.

**4. Read the results.**

```bash
bench eval list jobs/            # one row per experiment: claude 0/1, oracle 1/1, ...
bench eval metrics jobs/claude   # pass rate, solve rate, pass@k, tool calls and time
bench eval view jobs/claude      # trajectory, verifier output and score, in your browser
```

Each run writes its rewards, token usage and full trajectory under its `--jobs-dir`. Running the same command again resumes that job: finished tasks are kept, not rerun. Add `--fresh` to start a new run.

Next, [write your own task](./docs/task-authoring.md) with `bench tasks init`, or run a whole benchmark (`--source-path tasks --concurrency 8`). Prefer Python? See [Run from Python](./docs/getting-started.md#run-from-python); [`quickstart.py`](./docs/examples/python-sdk/quickstart.py) runs a task with the oracle in under a minute, with no key. [Getting started](./docs/getting-started.md) walks the whole path. Existing users: see [What's new in 0.8](./docs/whats-new-0.8.md). Here for the trajectory prize? See [Contribute trajectory captures](./docs/traj-upload.md).

## New in 0.8

- **Reward integrity.** `--integrity audit` records what the agent did to the things its reward depends on, checks that record against the task's contract, and writes a verdict next to the reward: `Checked`, `VectorExposed`, `AgentViolation` or `Rejected`. The verdict never changes the reward. `--integrity strict` also runs the verifier in a sandbox the agent was never in. This is BenchShield ([arXiv 2609.11028](https://arxiv.org/abs/2609.11028)); see [Reward integrity](./docs/integrity.md).
- **Native harness.** `--harness native` runs Claude Code and Codex through their own CLIs in headless JSON mode instead of their ACP adapters, with the same agent entry, credentials, skills, sandbox and trajectory. See [Native harness](./docs/native-harness.md).
- **Separate verifier sandboxes.** A task can score in a sandbox that receives only the agent's final workspace and the task's declared artifacts, so nothing else the agent left behind can reach the verifier. See [Separate verifier sandboxes](./docs/separate-verifier.md).
- **Regrade and branching.** `bench eval regrade` re-scores a run made with `--freeze-workspace` after a verifier fix, without running the agent again, and `bench eval branch` forks a run at a checkpoint into verifier-scored children. See [Regrade](./docs/regrade.md) and [Branching](./docs/branching.md).
- **Task formats.** task.md draft 2 packages run natively, and a benchmark can register its own format so its task folders run with no export step. See [task.md draft 2 packages](./docs/task-authoring-taskmd-v2.md) and [Task formats](./docs/task-formats.md).
- **Robots and simulators.** `benchflow.embodied` provides the embodiment spec, the `robo` protocol, seeded rollouts and training export. See [Embodied rollouts](./docs/embodied.md).
- **Training.** `bench train stream` hands each finished rollout to a trainer while the job is still running, and gateway token capture records token ids and logprobs from vLLM and SGLang servers. See [Streaming rollouts](./docs/reference/rollout-stream.md) and [Token capture](./docs/reference/token-capture.md).
- **Caps and CI.** Budgets (`--max-cost-usd`, `--max-tokens`, `--max-rollouts`) stop a job at a cap; `--fail-under` and `--summary-out` gate CI; `bench eval resume` finishes an interrupted job; and `--sandbox remote-docker` runs tasks on a Docker host you control. See [Budgets](./docs/reference/budget.md) and [Remote Docker](./docs/remote-docker.md).
- **Errors that say what to do.** `bench doctor` checks the machine before the first run, and the end-of-run summary says why trials have no score, whose problem it is and what to do next. See [When a run fails](./docs/when-a-run-fails.md).
- **Reading results.** `bf.load_job`, `bench eval inspect` and `bench eval compare`, pass@k and pass^k, and solve rates with a 95% interval. See [Analysing runs](./docs/analysing-runs.md).

Upgrading from 0.7? [What's new in 0.8](./docs/whats-new-0.8.md) lists what to do differently, and the [CHANGELOG](./CHANGELOG.md) has the details.

## Install

Install or upgrade to the latest stable release from PyPI with `uv`:

```bash
uv tool install --python 3.12 --upgrade benchflow
```

- Confirm with `bench --version`.
- BenchFlow CLI releases require Python 3.12 or newer. Keep `--python 3.12` in the install command so `uv` does not resolve an older Python-compatible package that lacks the CLI entrypoints.
- If you see `Executables already exist: bench, benchflow`, re-run with `uv tool install --python 3.12 --upgrade --force benchflow` to replace stale entrypoints from an older install.
- For Daytona, Modal, or AgentCore extras, install the relevant optional package, for example `uv tool install --python 3.12 --upgrade 'benchflow[sandbox-daytona]'`.

Internal users wanting the newest preview from `main` install the [internal preview channel](./docs/release.md) (`uv tool install --python 3.12 --prerelease allow --upgrade benchflow`).

**Requirements & auth.** Install [uv](https://docs.astral.sh/uv/); the `--python 3.12` flag lets it provision a compatible interpreter for the tool install. Set `DAYTONA_API_KEY` for Daytona or configure Modal auth for Modal; export an agent API key (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, …) or use subscription auth (`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`, or `codex login`). Provider-prefixed models may need provider-specific credentials; Azure Foundry uses `AZURE_API_KEY` + `AZURE_API_ENDPOINT`.

## Documentation

Start with [Getting started](./docs/getting-started.md), then [Concepts](./docs/concepts.md) for the mental model. Prefer to have an AI coding agent run the whole quickstart for you? Paste the [agent quickstart prompt](./docs/agent-quickstart.md) into Claude Code, Codex CLI, or Gemini CLI. Then by goal:

### Start and run

| If you want to… | Read |
|------------------|------|
| Run an eval on an existing task | [Getting started](./docs/getting-started.md) |
| Upgrade from 0.7: what changed and what to do differently | [What's new in 0.8](./docs/whats-new-0.8.md) |
| Have an AI agent install + run the quickstart end to end | [Agent quickstart prompt](./docs/agent-quickstart.md) |
| Understand Rollout / Scene / Role / Verifier | [Concepts](./docs/concepts.md) |
| Understand how BenchFlow runs *any* benchmark (the three-layer model) | [Run any benchmark](./docs/running-any-benchmark.md) |
| Find out why a run failed, whose problem it is, and what to do | [When a run fails](./docs/when-a-run-fails.md) |
| Run an agent from the public agents repo (goose, qwen-code, prime-agent, …) | [Running external agents](./docs/external-agents.md) |
| Run Claude Code or Codex through their own CLIs instead of their ACP adapters | [Native harness](./docs/native-harness.md) |
| Run a hosted PrimeIntellect / Verifiers environment | [CLI reference](./docs/reference/cli.md) |
| Run tasks on a Docker host you control | [Remote Docker](./docs/remote-docker.md) |
| Cap a job's spend, sandbox time, tokens or rollouts | [Budgets](./docs/reference/budget.md) |

### Multi-agent, multi-round and branching

| If you want to… | Read |
|------------------|------|
| Multi-agent: coder + reviewer, simulated user, BYOS, stateful envs | [Use cases](./docs/use-cases.md) |
| Multi-round single-agent (progressive disclosure, oracle access) | [Progressive disclosure](./docs/progressive-disclosure.md) |
| Fork a run at a checkpoint into children (compare prompts, parallel or nested children, retry from a checkpoint, branch-tree training data) | [Branching guide](./docs/branching.md) |
| Skill evaluation (when the artifact is a skill, not a workspace) | [Skill eval](./docs/skill-eval.md) |
| Robots and simulators: embodiment spec, `robo` protocol, episode server, seeded rollouts, training export | [Embodied rollouts](./docs/embodied.md) |

### Scores you can trust

| If you want to… | Read |
|------------------|------|
| Understand the security model | [Sandbox hardening](./docs/sandbox-hardening.md) |
| Score in a sandbox that holds only the agent's final workspace and declared artifacts | [Separate verifier sandboxes](./docs/separate-verifier.md) |
| Check what the agent did to the things its reward depends on (BenchShield) | [Reward integrity](./docs/integrity.md) |
| Re-score stored runs after fixing a verifier, without running the agent again | [Regrade](./docs/regrade.md) |

### Tasks

| If you want to… | Read |
|------------------|------|
| Author a new task | [Task authoring](./docs/task-authoring.md) |
| Author a task in the native `task.md` format | [Native task.md authoring](./docs/task-authoring-task-md.md) |
| Run task.md draft 2 packages (a design draft) | [task.md draft 2 packages](./docs/task-authoring-taskmd-v2.md) |
| Run a benchmark's own task folders natively (a task format plugin) | [Task formats](./docs/task-formats.md) |

### Results and training

| If you want to… | Read |
|------------------|------|
| Read, compare and export runs in Python or a notebook | [Analysing runs](./docs/analysing-runs.md) |
| Stream rollouts to a trainer while a job runs | [Streaming rollouts](./docs/reference/rollout-stream.md) |
| Capture token ids and logprobs for training | [Token capture](./docs/reference/token-capture.md) |
| Contribute a trajectory capture (the eval prize) | [Trajectory upload](./docs/traj-upload.md) |

### Reference

| If you want to… | Read |
|------------------|------|
| CLI flags + commands | [CLI reference](./docs/reference/cli.md) |
| Python API surface | [Python API reference](./docs/reference/python-api.md) |
| Use public vs internal preview SDK releases | [Release channels](./docs/release.md) |

Notebooks and runnable example scripts live under [`docs/examples/`](./docs/examples/) so examples stay versioned with the docs that explain them.

> **`bench agent` vs `bench eval adopt`.** `bench agent list` / `bench agent show` inspect **registered AI agents** (the solver programs like Claude Code or Gemini CLI). Onboarding a third-party benchmark into `benchmarks/<name>/` is a separate workflow — `bench eval adopt <source>` scaffolds and drives the conversion, and `bench eval adopt <name> --verify` parity-gates it. (The legacy `bench agent create|run|verify` commands still work as deprecated aliases.) See the [CLI reference](./docs/reference/cli.md#bench-eval-adopt) for details.

## Benchmark task sources

Benchmark datasets live in external Git repos and are referenced with two fields:

```yaml
# benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml
source:
  repo: benchflow-ai/benchmarks    # GitHub org/repo
  path: datasets/harvey-lab/tasks  # optional subpath within repo
  ref: main                         # optional branch/tag
agent: gemini
model: gemini/gemini-3.1-flash-lite-preview
```

Run any benchmark via the CLI:

```bash
# From a YAML config (shipped with the repo)
bench eval run --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml

# Inline — mirrors the YAML source fields
bench eval run \
    --source-repo benchflow-ai/skillsbench --source-path tasks \
    --agent gemini --model gemini-3.1-flash-lite-preview --sandbox daytona --concurrency 64
```

Repos are cached under `.cache/datasets/` (in the enclosing git repository's root, or the current directory outside one). With a source path, only that folder is downloaded (a sparse clone), and each later path joins the same cache; a source without a path, or a folder that holds no BenchFlow task, gets the whole repository.

Hosted environments are another source type. Instead of a repo, pass `--source-env` with the environment's pinned source version to run an external PrimeIntellect / Verifiers environment on its own native harness — BenchFlow preserves the hosted identity (`env_uid`, `hub_url`) and still writes the shared rollout output contract. See the [CLI reference](./docs/reference/cli.md) for the full hosted-environment command shape.

Downstream projects should depend on the public PyPI release by default. For internal validation before the next public release, install or lock the internal preview channel with prereleases enabled; see [Release channels](./docs/release.md).

## Authoring tasks

A task is one `task.md` (YAML frontmatter for config + a markdown prompt body) plus `environment/` and `verifier/` sidecars. The `bench tasks` commands cover the authoring lifecycle:

```bash
bench tasks init my-task                 # scaffold a task.md package under tasks/
bench tasks check tasks/my-task          # validate (default --level structural)
bench tasks migrate legacy-task/ --remove-legacy  # convert old split packages to task.md
bench tasks export tasks/my-task out/             # write a compatibility export + loss report
```

See [Native task.md authoring](./docs/task-authoring-task-md.md) and the [task standard](./docs/task-standard.md). BenchFlow also runs older split packages (`task.toml`, `instruction.md`, `solution/`, `tests/`) until you migrate them, [task.md draft 2](./docs/task-authoring-taskmd-v2.md) packages, and any [task format](./docs/task-formats.md) a benchmark registers.

## Featured

- **Progressive disclosure on SWE-bench Pro** — the `BaseUser` abstraction drives a multi-round rollout: terse round-0 prompt → failing-test hints → full spec. 5/5 oracle on Daytona, runnable demo at [`docs/examples/swebench_pro_progressive_disclosure.ipynb`](./docs/examples/swebench_pro_progressive_disclosure.ipynb). See [Progressive disclosure](./docs/progressive-disclosure.md).
- **Hill-climbing a skill with a held-out test split** — an optimizer agent edits a skills folder once per round, and an edit is kept only if the train score gains and the held-out test split also improves. The optimizer runs as a sandboxed rollout that only ever holds the train split. Built from BenchFlow's public primitives after [Automating eval design and hillclimbing with Claude](https://claude.dev/blog/automating-eval-design-and-hillclimbing/), with a SkillsBench recipe. See [`docs/examples/hillclimb/`](./docs/examples/hillclimb/).

## Audience

- **Eval researchers / paper writers** → [Getting started](./docs/getting-started.md) → [Concepts](./docs/concepts.md) → [Use cases](./docs/use-cases.md)
- **Task authors** → [Task authoring](./docs/task-authoring.md) → [Sandbox hardening](./docs/sandbox-hardening.md) → [Reward integrity](./docs/integrity.md)
- **Agent builders integrating with benchflow** → [Concepts](./docs/concepts.md) → [Python API reference](./docs/reference/python-api.md) → [`benchflow.agents.registry`](./src/benchflow/agents/registry.py)
- **External benchmark adapters** → [Task authoring](./docs/task-authoring.md) → [Task formats](./docs/task-formats.md) → [Progressive disclosure](./docs/progressive-disclosure.md#comparison-with-multi-agent-simulated-user)
- **RL and post-training** → [Streaming rollouts](./docs/reference/rollout-stream.md) → [Token capture](./docs/reference/token-capture.md) → [Training signal](./docs/reference/training-signal.md)

## Contributing

PRs welcome. Open them against `main`. On every pull request, CI runs ruff (lint and format), the `ty` type checker, the unit tests, a Docker integration tier and `pip-audit`. Run the same checks before you push:

```bash
uv sync --extra dev --extra sandbox-daytona --extra sandbox-modal
uv run ruff check src tests tools
uv run ruff format --check src tests tools
uv run ty check
uv run python -m pytest tests/
```

Release channels are documented in [Release channels](./docs/release.md). In short: merges to `main` publish an internal preview after CI passes, while a matching release tag publishes the public release.

## License

Apache-2.0.
