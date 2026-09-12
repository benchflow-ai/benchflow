# Running external agents

BenchFlow's built-in registry covers a handful of agents (`bench agent list`).
Everything else — goose, qwen-code, prime-agent, the omnigent harnesses, … —
lives in the public **[benchflow-ai/agents](https://github.com/benchflow-ai/agents)**
repo and loads into BenchFlow through one of three paths. For most users the
first one is all there is to know.

## 1. Zero-config remote autoload (the default)

Using an agent name BenchFlow doesn't recognize triggers a one-shot fetch of
the declarative manifests from `benchflow-ai/agents@main`. Nothing to install
or configure. End to end, from an empty directory to a scored rollout
(verified as written: skillsbench `edit-pdf`, reward 1.0, ~23 min, a few
cents on `deepseek-v4-flash` — rates approximate):

```bash
# Install. `uv tool install --python 3.12 'benchflow[sandbox-daytona]'` (the
# README's global-CLI idiom) and a plain venv + pip both work; pick one.
pip install 'benchflow[sandbox-daytona]'

export DEEPSEEK_API_KEY=sk-...          # credentials for your --model provider
export DAYTONA_API_KEY=dtn_...          # or use --sandbox docker

# Get a task (sparse checkout of one SkillsBench task — same recipe as
# getting-started; any task.md/Harbor-format task directory works):
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/benchflow-ai/skillsbench
cd skillsbench && git sparse-checkout set tasks/edit-pdf

bench eval run --tasks-dir tasks/edit-pdf --agent prime-agent \
  --model deepseek/deepseek-v4-flash --sandbox daytona
```

While the agent works, a terminal (TTY) shows the live Rich dashboard —
progress bar, pass/fail counts, and a per-task activity column that tracks
tool calls/tokens and labels the non-agent stretches (`creating sandbox…`,
`installing agent…`, `verifying…`). Plain output (CI, pipes) keeps the
throttled progress heartbeat instead: single-concurrency runs print a line
about every 45 seconds (`… 6.2min, 12 tool calls (last: …)`), and
multi-concurrency jobs gate it off by default. `--quiet` silences both. The
full event stream lands in `trajectory/acp_trajectory.jsonl` in the rollout
dir; the run ends with the `✓ Score` line either way.

The first rollout runs the agent's `install_cmd` inside the sandbox (a few
minutes for agents that bootstrap toolchains); artifacts land in the jobs dir
exactly as for built-in agents, including the gateway's raw LLM trace.

Details worth knowing:

- `DEEPSEEK_BASE_URL` is optional: unset, `deepseek/*` models default to the
  public OpenAI-compatible endpoint (`https://api.deepseek.com/v1`); set it
  only to route to a different OpenAI-compatible deployment.
- The fetch happens **at most once per process**, only on a resolution miss,
  and only fills gaps — it never shadows a built-in or already-registered
  agent name.
- Availability of a name depends on it being merged to the agents repo's
  `main`. What exists is listed in that repo's `acp/` directory and, for the
  ACP-registry tier, its generated `acp-registry/AGENTS.md`.

## 2. Pin the source: `BENCHFLOW_AGENTS_SOURCE`

Override where the autoload fetches from — a branch/ref, another repo, a local
directory, or off entirely:

```bash
export BENCHFLOW_AGENTS_SOURCE="benchflow-ai/agents@my-branch"   # owner/repo[@ref]
export BENCHFLOW_AGENTS_SOURCE="benchflow-ai/agents@0123456789abcdef0123456789abcdef01234567" # reproducible
export BENCHFLOW_AGENTS_SOURCE="/path/to/agents-checkout"        # local dir
export BENCHFLOW_AGENTS_SOURCE="off"                             # disable autoload
```

Accepted off-values: `off`, `0`, `none`, `disabled`, `false`. Pinning a ref is
the standard way to try an agent from an open PR — e.g. verified live on
BenchFlow 0.6.6: `BENCHFLOW_AGENTS_SOURCE="benchflow-ai/agents@add-prime-agent"`
resolved and ran the `prime-agent` manifest with zero local setup.

`bench agent list` reads this catalog best-effort and reports one consolidated
warning when it is incomplete. `bench agent show NAME` uses runtime resolution.
Unknown bare IDs fail closed. Raw commands require explicit syntax: whitespace
(`agent --flag`) or a path prefix (`/`, `./`, `../`, or `~/`). Remote refs are
fetched/refreshed by the normal source resolver; there is no offline fallback.

## 3. Local checkout override: `BENCHFLOW_AGENTS_DIR`

For agents-repo development, point at a checkout. This selects the same lazy
catalog path used by `BENCHFLOW_AGENTS_SOURCE`:

```bash
export BENCHFLOW_AGENTS_DIR=/path/to/agents-checkout
```

When both variables are set, a nonblank `BENCHFLOW_AGENTS_DIR` wins. Its local
override mode may update an unchanged built-in's manifest-owned fields, but
never replaces a plugin/runtime-modified entry. Loading remains one-shot after
first runtime resolution. `bench agent list` may preview the selected catalog
without activating or caching it; after activation, listing reuses the applied
result.

## 4. Plugin packages (entry points)

Agents that need host-side Python (session-factory adapters like the omnigent
harnesses, or packaged code agents like `mini-swe-acp`) ship as pip packages
that register through the `benchflow.agents` entry-point group — installing
the package is all it takes:

```bash
pip install "mini-swe-acp @ git+https://github.com/benchflow-ai/agents#subdirectory=acp/mini-swe-acp"
bench eval run --tasks-dir ./tasks --agent mini-swe --model openai/gpt-4o-mini
```

These load at `benchflow` import time. A plugin that fails to import never
blocks the run — the failure is recorded and surfaced in the "Unknown agent"
error message if its name is later requested.

## Precedence

1. Built-in registry (core `AGENTS`).
2. Entry-point plugin packages, loaded at import. Plugins may replace built-ins.
3. One lazily selected manifest catalog. Nonblank `BENCHFLOW_AGENTS_DIR` wins
   over `BENCHFLOW_AGENTS_SOURCE` and uses local-override policy; otherwise
   remote/default loading uses gap-fill policy and never replaces existing
   entries.

Manifest capabilities are deliberately bounded: a `manifest.toml` is data-only
(install/launch commands, env mapping, model-routing hints — the
[agents-repo contract](https://github.com/benchflow-ai/agents/tree/main/contract)).
Anything needing host-side logic (credential files, session factories, native
MCP config) must come in as a plugin package or a core agent instead.
