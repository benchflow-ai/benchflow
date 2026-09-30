<div align="center">
  <h1>BenchFlow</h1>
  <p>The universal environment framework — a benchmark is just a frozen environment.</p>
  <a href="https://pypi.org/project/benchflow/" target="_blank">
    <img src="https://img.shields.io/badge/PyPI-benchflow-3775A9?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI package">
  </a>
  <a href="https://discord.gg/mZ9Rc8q8W3" target="_blank">
    <img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
  </a>
</div>

## What

BenchFlow runs AI agents on tasks in sandboxes and scores them, with one hardened runtime for evaluation and training. **A benchmark is just a frozen environment**: point BenchFlow at a benchmark's tasks, drive them with any [ACP](https://agentclientprotocol.com) agent (Claude Code, Codex, Gemini CLI, OpenHands and more), and read rewards, costs and full trajectories from every run. The same tasks run single-agent, multi-agent or multi-round, on Docker, Daytona or Modal.

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
export CLAUDE_CODE_OAUTH_TOKEN=<token>
bench eval run --source-repo benchflow-ai/skillsbench --source-path tasks/citation-check \
  --agent claude --model claude-haiku-4-5-20251001 --sandbox docker --jobs-dir jobs/claude
```

The agent may pass or fail; either way the evaluation completed. `bench eval smoke` runs a bundled hello-world task once with every agent you are logged in to, which checks each login in about a minute. For a ChatGPT subscription, run `codex login` and use `--agent codex`; see [Auth](./docs/getting-started.md#auth-oauth-long-lived-token-or-api-key) for API keys and other agents.

**4. Read the results.**

```bash
bench eval list jobs/            # one row per experiment: claude 0/1, oracle 1/1, ...
bench eval metrics jobs/claude   # pass rate, solve rate, pass@k, tool calls and time
bench eval view jobs/claude      # trajectory, verifier output and score, in your browser
```

Each run writes its rewards, token usage and full trajectory under its `--jobs-dir`. Running the same command again resumes that job: finished tasks are kept, not rerun. Add `--fresh` to start a new run.

Next, [write your own task](./docs/task-authoring.md) with `bench tasks init`, or run a whole benchmark (`--source-path tasks --concurrency 8`). [Getting started](./docs/getting-started.md) walks the whole path. Existing users: see [What's new in 0.8](./docs/whats-new-0.8.md). Here for the trajectory prize? See [Contribute trajectory captures](./docs/traj-upload.md).

## Install

Install or upgrade to the latest stable release from PyPI with `uv`:

```bash
uv tool install --python 3.12 --upgrade benchflow
```

- Confirm with `bench --version`.
- BenchFlow CLI releases require Python 3.12 or newer. Keep `--python 3.12`
  in the install command so `uv` does not resolve an older Python-compatible
  package that lacks the CLI entrypoints.
- If you see `Executables already exist: bench, benchflow`, re-run with `uv tool install --python 3.12 --upgrade --force benchflow` to replace stale entrypoints from an older install.
- For Daytona, Modal, or AgentCore extras, install the relevant optional package, for example `uv tool install --python 3.12 --upgrade 'benchflow[sandbox-daytona]'`.

Internal users wanting the newest preview from `main` install the [internal preview channel](./docs/release.md) (`uv tool install --python 3.12 --prerelease allow --upgrade benchflow`).

**Requirements & auth.** Install [uv](https://docs.astral.sh/uv/); the `--python 3.12` flag lets it provision a compatible interpreter for the tool install. Set `DAYTONA_API_KEY` for Daytona or configure Modal auth for Modal; export an agent API key (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, …) or use subscription auth (`CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`, or `codex login`). Provider-prefixed models may need provider-specific credentials; Azure Foundry uses `AZURE_API_KEY` + `AZURE_API_ENDPOINT`.

## Documentation

Start with [Getting started](./docs/getting-started.md), then [Concepts](./docs/concepts.md) for the mental model. Prefer to have an AI coding agent run the whole quickstart for you? Paste the [agent quickstart prompt](./docs/agent-quickstart.md) into Claude Code, Codex CLI, or Gemini CLI. Then by goal:

| If you want to… | Read |
|------------------|------|
| Run an eval on an existing task | [Getting started](./docs/getting-started.md) |
| Upgrade from 0.7: what changed and what to do differently | [What's new in 0.8](./docs/whats-new-0.8.md) |
| Understand how BenchFlow runs *any* benchmark (the three-layer model) | [Run any benchmark](./docs/running-any-benchmark.md) |
| Have an AI agent install + run the quickstart end to end | [Agent quickstart prompt](./docs/agent-quickstart.md) |
| Run an agent from the public agents repo (goose, qwen-code, prime-agent, …) | [Running external agents](./docs/external-agents.md) |
| Understand Rollout / Scene / Role / Verifier | [Concepts](./docs/concepts.md) |
| Author a new task | [Task authoring](./docs/task-authoring.md) |
| Author a task in the native `task.md` format | [Native task.md authoring](./docs/task-authoring-task-md.md) |
| Run a hosted PrimeIntellect / Verifiers environment | [CLI reference](./docs/reference/cli.md) |
| Multi-agent: coder + reviewer, simulated user, BYOS, stateful envs | [Use cases](./docs/use-cases.md) |
| Multi-round single-agent (progressive disclosure, oracle access) | [Progressive disclosure](./docs/progressive-disclosure.md) |
| Fork a run at a checkpoint into children (compare prompts, parallel or nested children, retry from a checkpoint, branch-tree training data) | [Branching guide](./docs/branching.md) |
| Skill evaluation (when the artifact is a skill, not a workspace) | [Skill eval](./docs/skill-eval.md) |
| Read or score a physical robot trial (`trial-record.json`) | [Physical robot trials](./docs/robotics.md) |
| Contribute a trajectory capture (the eval prize) | [Trajectory upload](./docs/traj-upload.md) |
| Understand the security model | [Sandbox hardening](./docs/sandbox-hardening.md) |
| Use public vs internal preview SDK releases | [Release channels](./docs/release.md) |
| CLI flags + commands | [CLI reference](./docs/reference/cli.md) |
| Python API surface | [Python API reference](./docs/reference/python-api.md) |

Notebooks and runnable example scripts live under [`docs/examples/`](./docs/examples/) so examples stay versioned with the docs that explain them.

> **`bench agent` vs `bench eval adopt`.** `bench agent list` / `bench agent show`
> inspect **registered AI agents** (the solver programs like Claude Code or
> Gemini CLI). Onboarding a third-party benchmark into `benchmarks/<name>/` is a
> separate workflow — `bench eval adopt <source>` scaffolds and drives the
> conversion, and `bench eval adopt <name> --verify` parity-gates it. (The legacy
> `bench agent create|run|verify` commands still work as deprecated aliases.)
> See the [CLI reference](./docs/reference/cli.md#bench-eval-adopt) for details.

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

Repos are cached under `.cache/datasets/` (in the enclosing git repository's root, or the current directory outside one). With a source path, only that path is downloaded (a sparse clone that fetches no other file), and each later path joins the same cache; a source without a path, or a folder that holds no BenchFlow task, gets the whole repository.

Hosted environments are another source type. Instead of a repo, pass
`--source-env` with the environment's pinned source version to run an external
PrimeIntellect / Verifiers environment on its own native harness — BenchFlow
preserves the hosted identity (`env_uid`, `hub_url`) and still writes the shared
rollout output contract. See the [CLI reference](./docs/reference/cli.md) for
the full hosted-environment command shape.

Downstream projects should depend on the public PyPI release by default. For
internal validation before the next public release, install or lock the internal
preview channel with prereleases enabled; see [Release channels](./docs/release.md).

## Authoring tasks

A task is one `task.md` (YAML frontmatter for config + a markdown prompt body)
plus `environment/` and `verifier/` sidecars. The `bench tasks` commands cover
the authoring lifecycle:

```bash
bench tasks init my-task                 # scaffold a task.md package under tasks/
bench tasks check tasks/my-task          # validate (default --level structural)
bench tasks migrate legacy-task/ --remove-legacy  # convert old split packages to task.md
bench tasks export tasks/my-task out/             # write a compatibility export + loss report
```

See [Native task.md authoring](./docs/task-authoring-task-md.md) and the
[task standard](./docs/task-standard.md).

## Featured

- **Progressive disclosure on SWE-bench Pro** — the `BaseUser` abstraction drives a multi-round rollout: terse round-0 prompt → failing-test hints → full spec. 5/5 oracle on Daytona, runnable demo at [`docs/examples/swebench_pro_progressive_disclosure.ipynb`](./docs/examples/swebench_pro_progressive_disclosure.ipynb). See [Progressive disclosure](./docs/progressive-disclosure.md).

## Audience

- **Eval researchers / paper writers** → [Getting started](./docs/getting-started.md) → [Concepts](./docs/concepts.md) → [Use cases](./docs/use-cases.md)
- **Task authors** → [Task authoring](./docs/task-authoring.md) → [Sandbox hardening](./docs/sandbox-hardening.md)
- **Agent builders integrating with benchflow** → [Concepts](./docs/concepts.md) → [Python API reference](./docs/reference/python-api.md) → [`benchflow.agents.registry`](./src/benchflow/agents/registry.py)
- **External benchmark adapters** → [Task authoring](./docs/task-authoring.md) → [Progressive disclosure](./docs/progressive-disclosure.md#comparison-with-multi-agent-simulated-user)

## Contributing

PRs welcome. Open against `main`. CI runs ruff + tests on every PR; please run `ruff check .` and `pytest tests/` locally first.

Release channels are documented in [Release channels](./docs/release.md). In
short: merges to `main` publish an internal preview after CI passes, while a
matching release tag publishes the public release.

## License

Apache-2.0.
