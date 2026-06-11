# Run any benchmark

BenchFlow's job is to take *any* benchmark and produce a scored trajectory you
can read, compare, and train on. It sits downstream of every environment
framework: whatever shape a benchmark arrives in, BenchFlow routes it to one of
three execution layers and ends at a single output contract.

The routing is the whole idea. You do not pick the layer by hand — the benchmark's
format picks it for you:

| If the benchmark is… | Layer | What BenchFlow does |
|----------------------|-------|---------------------|
| In a framework BenchFlow already speaks (Harbor / Terminal-Bench, PrimeIntellect / Verifiers, Inspect AI, ORS / OpenReward) | **1 — native** | Runs it in its supported form; correctness is inherited from the original format |
| In a variant of a known format, or a format BenchFlow has never seen | **2 — translated** | Translates it to the native `task.md` format, then *proves* equivalence with a parity gate |
| A one-off harness with its own runner and scoring, no reusable adapter | **3 — as-is** | Runs the benchmark under its own harness and interfaces with its output only |

All three layers terminate at the same scored-trajectory contract (see
[The seam](#the-seam-one-scored-trajectory-contract) below). One ingestion,
every benchmark.

---

## Layer 1 — supported framework → run natively

When a benchmark already speaks a framework BenchFlow supports, BenchFlow runs it
in that form and converts the output to results. There is no translation step and
nothing to prove: correctness is inherited from the original format.

The adapters live in [`src/benchflow/adapters/`](../src/benchflow/adapters/) and
are pure format translators — none of them require the external framework's SDK.

**Inbound (foreign task dir → native runtime).**
[`detect_adapter()`](../src/benchflow/adapters/inbound.py) sniffs a task
directory by its signature file and returns the matching adapter:

- A `task.toml` → [`HarborAdapter`](../src/benchflow/adapters/harbor.py). Harbor
  is the upstream framework BenchFlow's own `TaskConfig` was internalized from, so
  a Harbor task directory is *already* in native shape; the adapter is a thin
  normalizer. This is also what keeps Terminal-Bench tasks backward-compatible.
- A `task.yaml` → [`TerminalBenchAdapter`](../src/benchflow/adapters/terminal_bench.py),
  a pure `task.yaml` → native `TaskConfig` translation that lifts the embedded
  instruction out, flattens the metadata, and remaps the foreign file layout.

Each adapter returns an `InboundTask`; the benchmark then runs on BenchFlow's
native runtime exactly like a first-party task.

**Hosted environments (run on their own native surface).** External
PrimeIntellect / Verifiers environments are not BenchFlow task directories and do
not use BenchFlow's Docker/Daytona sandbox runner. BenchFlow runs them through
their native Verifiers execution surface and preserves their hosted identity
(`env_uid`, `hub_url`), while still writing the shared output contract — see
[`src/benchflow/hosted_env.py`](../src/benchflow/hosted_env.py):

```bash
bench eval create \
    --source-env primeintellect/general-agent \
    --source-env-version 0.1.1 \
    --model google/gemini-2.5-flash-lite
```

**Outbound format seams (native results → other frameworks).** Results also
round-trip *out* into the frameworks teams already use:
[`InspectAdapter`](../src/benchflow/adapters/inspect_ai.py) maps a BenchFlow
`Scene` + `Rubric` into an Inspect AI task, and
[`ORSAdapter`](../src/benchflow/adapters/ors.py) maps `VerifyResult` /
`RewardEvent` into the ORS (OpenReward) reward-record shape.

---

## Layer 2 — unknown or variant format → translate, then prove parity

When a benchmark's format is **not** one BenchFlow speaks natively — a variant of
a known layout, or an entirely new one — BenchFlow translates it to the native
`task.md` format and then **proves** the translation produces the same results.
This is the load-bearing "same results" guarantee, and it is the difference
between a conversion you can trust and one you merely hope is faithful.

### Translate

- `bench tasks migrate <dir>` converts a legacy split layout (`task.toml` +
  `instruction.md`) into the unified `task.md` format
  ([`cli/main.py`](../src/benchflow/cli/main.py) → `migrate_task_to_task_md`).
- For a foreign benchmark with no reusable adapter, the benchmark-adoption router
  in [`src/benchflow/agent_router.py`](../src/benchflow/agent_router.py) drives
  the work: `bench agent create <name>` scaffolds `benchmarks/<name>/` to the
  reference layout and the contract in
  [`benchmarks/CONVERT.md`](../benchmarks/CONVERT.md), and `bench agent run
  <source>` drives that conversion workflow with an agent toward a
  `benchmarks/<name>/` pull request.

### Prove — the parity gate

`bench agent verify <name>` closes the adopt → verify loop. It is a **parity-only**
gate (`build_verify_report` in
[`agent_router.py`](../src/benchflow/agent_router.py)) over two layers:

1. **Deterministic conversion parity (the floor).** Every compared criterion's
   *converted* verdict must match the *original's* verdict on identical inputs —
   a side-by-side, per-criterion comparison (`extract_criterion_comparisons` →
   `ConversionParity`).
2. **Reward-distribution parity (the statistical layer).** Every
   legacy-vs-converted reward delta must sit within tolerance
   (`DEFAULT_REWARD_TOLERANCE = 0.02`, overridable with `--tolerance`) —
   `RewardDistributionParity`.

The gate emits one of three verdicts: `parity-confirmed`, `parity-divergent`, or
`insufficient-evidence` (a layer with no data does not block; no data at all
returns `insufficient-evidence`). Two principles keep it honest:

- **It never improves the source.** A faithful conversion reproduces the
  original's behavior on identical inputs — including any reward-hackability the
  original has. Parity never sanitizes or "fixes" the benchmark.
- **Divergences are triaged, not buried.** On a non-confirmed verdict the gate
  renders a draft issue body (`render_divergence_issue`) for a human to review
  and file — it never auto-files anything.

### The artifacts

Each adopted benchmark records its evidence in
`benchmarks/<name>/parity_experiment.json`, and `bench agent verify` scores that
file. The repository ships recorded parity experiments under
[`benchmarks/*/parity_experiment.json`](../benchmarks/) spanning deterministic
side-by-side conversion parity (per-criterion verdict agreement), structural
parity (every generated task carries the required files and valid metadata), and
agent-scale reward-distribution parity (legacy-vs-converted reward deltas across
trials).

Read the recorded experiments honestly: they report per-criterion agreement and
aggregate reward deltas **within tolerance**, with the residual disagreements
triaged to causes such as model non-determinism rather than conversion defects.
The guarantee is "parity within tolerance, divergences triaged, no conversion
defect found" — not a fixed headline percentage.

---

## Layer 3 — bespoke benchmark → run as-is, interface with output only

Some benchmarks have no reusable adapter and no portable task format: a one-off
harness with its own runner, its own scoring, and sometimes its own agent loop.
Translating it would be a rewrite. Instead, BenchFlow runs the benchmark **under
its own harness** and interfaces only with its output, mapping that onto the
shared contract.

Three real seams support this shape:

- **The ACP shim pattern.** A benchmark's native agent loop is wrapped as an ACP
  server, so BenchFlow drives it over stdio and reads the trajectory it emits —
  the original loop runs unchanged. The agent registry
  ([`src/benchflow/agents/registry.py`](../src/benchflow/agents/registry.py))
  registers harness agents this way — a `*-harness` ACP-shim entry whose
  `launch_cmd` runs the original harness and whose `protocol` is `acp`.
- **Native-harness / hosted runs.**
  [`hosted_env.py`](../src/benchflow/hosted_env.py) runs an environment on its own
  execution surface, preserves the raw native evidence (e.g. under a `hosted_env/`
  subdir for forensics), and still writes the shared contract — the same
  "run as-is, ingest the output" shape, tagged with `source.type="hosted_env"`
  lineage.
- **Trace import.** `bench tasks generate --from-file / --from-hf / --from-local`
  ([`cli/trace_import.py`](../src/benchflow/cli/trace_import.py)) ingests external
  agent traces — JSONL trace files, HuggingFace datasets, or local Claude Code
  sessions — into BenchFlow records, with no harness conversion at all.

---

## The seam: one scored-trajectory contract

Every layer terminates at the *same* output contract, written per rollout under
`<jobs-dir>/<job>/<task>__<hash8>/` (the full layout is in
[Getting started → Where results land](./getting-started.md#where-results-land)):

| File | What it carries |
|------|-----------------|
| `result.json` | Rollout summary — rewards, errors, token usage/cost |
| `rewards.jsonl` | The reward record for the rollout (ORS / OpenReward shape) |
| `trajectory/acp_trajectory.jsonl` | Full agent trace as ACP events |
| `trajectory/llm_trajectory.jsonl` | Raw provider requests/responses (when captured) |
| `trainer/verifiers.jsonl` | Trainer-ready scored trajectory (Verifiers record) |
| `trainer/atif.json` | ATIF trajectory record |
| `trainer/adp.jsonl` | ADP trajectory record |
| `verifier/` | Raw verifier output (`reward.txt`, `ctrf.json`, stdout) |

Hosted runs share this artifact contract too (see the `hosted_env.py` module
docstring), with `source.type="hosted_env"` / `trajectory_source="hosted_env"`
marking the lineage.

Because the contract is the same regardless of which layer produced it, one
ingestion path serves every benchmark: dashboards, release checks, and trainers
read the same files whether the run came from a native adapter, a parity-proven
translation, or a bespoke harness.

---

## Where to go next

- [Getting started](./getting-started.md) — install and run your first eval
- [Concepts](./concepts.md) — Rollout / Scene / Role / Verifier
- [Native `task.md` authoring](./task-authoring-task-md.md) — the translation target for Layer 2
- [Benchmark conversion guide](../benchmarks/CONVERT.md) — the Layer 2 conversion + parity contract
- [Architecture](./architecture.md) — adapters and trainers as the edges of the system
