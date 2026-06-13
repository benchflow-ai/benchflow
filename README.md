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

BenchFlow is a universal environment framework: it runs AI agents against task environments and scores them through one hardened contract. **A benchmark is just a frozen environment** — point BenchFlow at any of them, drive it with *any* ACP agent, and run single-agent, multi-agent, or multi-round patterns over the same Scene-based lifecycle.

- **Run any benchmark** — three-layer routing runs supported frameworks natively, translates unknown formats and proves equivalence with a parity gate, or runs a bespoke harness as-is; every layer emits one scored-trajectory contract. See [Run any benchmark](./docs/running-any-benchmark.md)
- **Any ACP agent** — Gemini CLI, Claude Code, Codex, OpenCode, OpenHands, Pi, or your own
- **Single + multi + progressive** — single-agent / multi-agent (coder + reviewer, simulated user) / multi-round with a Python `BaseUser` callback
- **Loop strategies** — wrap any agent in a `--loop-strategy` (`verify-retry`, `self-review`); every rollout captures a per-iteration reward + token trajectory, so you can plot capability against cost (can a cheap model + loops match an expensive one at equal token spend?)
- **`task.md` tasks** — one file (YAML frontmatter + prompt body) replaces the split `task.toml` + `instruction.md` layout; author with `bench tasks init` / `check` / `migrate` / `export`
- **Hosted environments** — run external PrimeIntellect / Verifiers environments through `--source-env`, without converting them to BenchFlow tasks
- **Sandboxes** — Docker locally, Daytona for parallel cloud runs (orphaned sandboxes auto-reaped at eval start), Modal for serverless/GPU-backed task environments
- **Hardened verifier** — defaults block BenchJack/Meerkat-style reward-hacking; tasks opt out per-feature
- **Training-ready output** — every scored rollout emits ATIF (`trainer/atif.json`) and ADP (`trainer/adp.jsonl`) trajectory records next to the Verifiers/ORS (OpenReward) reward record

## Quickstart

```bash
# Install the current 0.6 release candidate (see Install for why the wheel URL)
uv tool install --prerelease allow \
  'benchflow @ https://github.com/benchflow-ai/benchflow/releases/download/0.6.0-rc.6/benchflow-0.6.0rc6-py3-none-any.whl'

# Run a benchmark: any task source, any ACP agent, any sandbox
export GEMINI_API_KEY=...            # or claude login / codex --login for subscription auth
bench eval create \
    --source-repo benchflow-ai/skillsbench --source-path tasks \
    --agent gemini --model gemini-3.1-flash-lite-preview \
    --sandbox daytona --concurrency 64
```

Each run writes a per-task `result.json` (rewards + trajectory + token usage) and a job `summary.json` (pass-rate, cost, and — for looped runs — a pass@iteration convergence curve). New here? Start with [Getting started](./docs/getting-started.md), or paste the [agent quickstart prompt](./docs/agent-quickstart.md) into Claude Code / Codex / Gemini CLI and let it drive the whole thing.

## Install

`0.6.0` is in **release-candidate** testing and is **not on PyPI yet** (the newest PyPI build is still `0.5.x`). Until it ships, install the latest `0.6.0-rc.*` wheel from the [GitHub releases page](https://github.com/benchflow-ai/benchflow/releases) — the Quickstart pins `0.6.0-rc.6`; if a newer `rc.*` exists, swap the tag and filename. Confirm with `bench --version`.

- `--prerelease allow` is required for BenchFlow's pinned LiteLLM release-candidate dependency.
- If you see `Executables already exist: bench, benchflow`, re-run with `--force` to replace stale entrypoints from an older install.

**Once `0.6.0` ships to PyPI**, the plain commands resolve (until then they pick up only `0.5.x`):

```bash
pip install --upgrade benchflow                                  # once 0.6.0 is on PyPI
uv tool install --prerelease allow --upgrade 'benchflow==0.6.0'  # once 0.6.0 is on PyPI
```

Internal users wanting the newest preview from `main` install the [internal preview channel](./docs/release.md) (`uv tool install --prerelease allow --upgrade benchflow`).

**Requirements & auth.** Python 3.12+ and [uv](https://docs.astral.sh/uv/). Set `DAYTONA_API_KEY` for Daytona or configure Modal auth for Modal; export an agent API key (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, …) or use subscription auth (`claude login` / `codex --login`). Provider-prefixed models may need provider-specific credentials; Azure Foundry uses `AZURE_API_KEY` + `AZURE_API_ENDPOINT`.

## Documentation

Start with [Getting started](./docs/getting-started.md), then [Concepts](./docs/concepts.md) for the mental model. Prefer to have an AI coding agent run the whole quickstart for you? Paste the [agent quickstart prompt](./docs/agent-quickstart.md) into Claude Code, Codex CLI, or Gemini CLI. Then by goal:

| If you want to… | Read |
|------------------|------|
| Run an eval on an existing task | [Getting started](./docs/getting-started.md) |
| Understand how BenchFlow runs *any* benchmark (the three-layer model) | [Run any benchmark](./docs/running-any-benchmark.md) |
| Have an AI agent install + run the quickstart end to end | [Agent quickstart prompt](./docs/agent-quickstart.md) |
| Understand Rollout / Scene / Role / Verifier | [Concepts](./docs/concepts.md) |
| Author a new task | [Task authoring](./docs/task-authoring.md) |
| Author a task in the native `task.md` format | [Native task.md authoring](./docs/task-authoring-task-md.md) |
| Adopt an upstream benchmark into BenchFlow | [Benchmark adoption](./docs/benchmark-adoption.md) |
| Run a hosted PrimeIntellect / Verifiers environment | [CLI reference](./docs/reference/cli.md) |
| Multi-agent: coder + reviewer, simulated user, BYOS, stateful envs | [Use cases](./docs/use-cases.md) |
| Multi-round single-agent (progressive disclosure, oracle access) | [Progressive disclosure](./docs/progressive-disclosure.md) |
| Skill evaluation (when the artifact is a skill, not a workspace) | [Skill eval](./docs/skill-eval.md) |
| Understand the security model | [Sandbox hardening](./docs/sandbox-hardening.md) |
| Use public vs internal preview SDK releases | [Release channels](./docs/release.md) |
| CLI flags + commands | [CLI reference](./docs/reference/cli.md) |
| Python API surface | [Python API reference](./docs/reference/python-api.md) |

Notebooks and runnable example scripts live under [`docs/examples/`](./docs/examples/) so examples stay versioned with the docs that explain them.

> **`bench agent` vs `bench adopt`.** `bench agent list` / `bench agent show`
> inspect **registered AI agents** (the solver programs like Claude Code or
> Gemini CLI). Onboarding a third-party benchmark into `benchmarks/<name>/` is a
> separate workflow — `bench adopt init` → `convert` → `verify`. (The legacy
> `bench agent create|run|verify` still work as deprecated aliases through 0.6.)
> See the [CLI reference](./docs/reference/cli.md#bench-agent) for details.

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
bench eval create --config benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml

# Inline — mirrors the YAML source fields
bench eval create \
    --source-repo benchflow-ai/skillsbench --source-path tasks \
    --agent gemini --model gemini-3.1-flash-lite-preview --sandbox daytona --concurrency 64
```

Repos are cloned and cached locally under `.cache/datasets/` on first use.

Hosted environments are another source type. Instead of a repo, pass
`--source-env` to run an external PrimeIntellect / Verifiers environment on its
own native harness — BenchFlow preserves the hosted identity (`env_uid`,
`hub_url`) and still writes the shared rollout output contract:

```bash
bench eval create \
    --source-env primeintellect/general-agent \
    --source-env-version 0.1.1 \
    --model google/gemini-2.5-flash-lite
```

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
bench tasks migrate legacy-task/         # convert task.toml + instruction.md → task.md
bench tasks export tasks/my-task out/    # write a Harbor/Pier split layout + loss report
```

See [Native task.md authoring](./docs/task-authoring-task-md.md) and the
[task standard](./docs/task-standard.md).

## Featured

- **Progressive disclosure on SWE-bench Pro** — the `BaseUser` abstraction drives a multi-round rollout: terse round-0 prompt → failing-test hints → full spec. 5/5 oracle on Daytona, runnable demo at [`docs/examples/swebench_pro_progressive_disclosure.ipynb`](./docs/examples/swebench_pro_progressive_disclosure.ipynb). See [Progressive disclosure](./docs/progressive-disclosure.md).

## Research artifacts

Two runnable labs validate the security story (historical, 0.2.x-era — archived under [`docs/labs/`](./docs/labs/)):

- [`docs/labs/benchjack-sandbox-hardening/`](./docs/labs/benchjack-sandbox-hardening/) — end-to-end demo that 0.2.1+ blocks three [BenchJack](https://rdi.berkeley.edu/blog/trustworthy-benchmarks-cont/) exploits that flip 0.2.0's reward from 0.0 to 1.0.
- [`docs/labs/reward-hack-matrix/`](./docs/labs/reward-hack-matrix/) — full reward-hack sweep across real benchmarks comparing 0.2.0 vs 0.2.2.

## Audience

- **Eval researchers / paper writers** → [Getting started](./docs/getting-started.md) → [Concepts](./docs/concepts.md) → [Use cases](./docs/use-cases.md)
- **Task authors** → [Task authoring](./docs/task-authoring.md) → [Sandbox hardening](./docs/sandbox-hardening.md)
- **Agent builders integrating with benchflow** → [Concepts](./docs/concepts.md) → [Python API reference](./docs/reference/python-api.md) → [`benchflow.agents.registry`](./src/benchflow/agents/registry.py)
- **External benchmark adapters** → [Task authoring](./docs/task-authoring.md) → [Progressive disclosure](./docs/progressive-disclosure.md#comparison-with-multi-agent-simulated-user)

## Contributing

PRs welcome. Open against `main`. CI runs ruff + tests on every PR; please run `ruff check .` and `pytest tests/` locally first.

Release channels are documented in [Release channels](./docs/release.md). In
short: merges to `main` publish an internal preview after CI passes, while a
matching `v<version>` tag publishes the public release.

## License

Apache-2.0.
