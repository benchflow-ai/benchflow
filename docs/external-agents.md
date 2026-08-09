# Running external agents

BenchFlow's built-in registry covers a handful of agents (`bench agent list`).
Everything else — goose, qwen-code, prime-agent, the omnigent harnesses, … —
lives in the public **[benchflow-ai/agents](https://github.com/benchflow-ai/agents)**
repo and loads into BenchFlow through one of four paths. For most users the
first one is all there is to know.

## 1. Zero-config remote autoload (the default)

Using an agent name BenchFlow doesn't recognize triggers a one-shot fetch of
the declarative manifests from `benchflow-ai/agents@main`. Nothing to install
or configure. End to end, from an empty directory to a scored rollout
(verified as written: skillsbench `edit-pdf`, reward 1.0, ~23 min, ~$0.02 on
`deepseek-v4-flash`):

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

Expect a quiet stretch while the agent works (its events stream to
`trajectory/acp_trajectory.jsonl` in the rollout dir, not the console) — the
console picks back up at the verifier and the `✓ Score` line.

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
export BENCHFLOW_AGENTS_SOURCE="/path/to/agents-checkout"        # local dir
export BENCHFLOW_AGENTS_SOURCE="off"                             # disable autoload
```

Accepted off-values: `off`, `0`, `none`, `disabled`, `false`. Pinning a ref is
the standard way to try an agent from an open PR — e.g. verified live on
BenchFlow 0.6.6: `BENCHFLOW_AGENTS_SOURCE="benchflow-ai/agents@add-prime-agent"`
resolved and ran the `prime-agent` manifest with zero local setup.

## 3. Local checkout at import: `BENCHFLOW_AGENTS_DIR`

For agents-repo development: point at a checkout and every
`<dir>/manifest.toml` under it merges into the registry when `benchflow`
imports (not lazily on miss):

```bash
export BENCHFLOW_AGENTS_DIR=/path/to/agents-checkout
```

Unlike the miss-driven autoload, this path loads even for names that would
never miss, and it is the loop used while editing a manifest. It is additive
and compatible-merge only: colliding with an existing agent's aliases is a
hard error rather than a silent shadow. Unset, the import is byte-for-byte
identical to core — the mechanism is strictly opt-in.

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

1. Built-in registry (core `AGENTS`) — never shadowed by anything below.
2. `BENCHFLOW_AGENTS_DIR` manifests — merged at import.
3. Entry-point plugin packages — loaded at import, after the manifest merge.
4. Remote autoload (`BENCHFLOW_AGENTS_SOURCE`, default `benchflow-ai/agents@main`)
   — consulted last, once, only for names still unknown at resolution time.

Manifest capabilities are deliberately bounded: a `manifest.toml` is data-only
(install/launch commands, env mapping, model-routing hints — the
[agents-repo contract](https://github.com/benchflow-ai/agents/tree/main/contract)).
Anything needing host-side logic (credential files, session factories, native
MCP config) must come in as a plugin package or a core agent instead.
