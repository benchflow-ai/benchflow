# Run any benchmark

BenchFlow's job is to take *any* benchmark and produce a scored trajectory you
can read, compare, and train on. It sits downstream of every environment
framework: whatever shape a benchmark arrives in, BenchFlow routes it to one of
three execution layers and ends at a single output contract.

The routing is the whole idea. You do not pick the layer by hand — the benchmark's
format picks it for you:

| If the benchmark is… | Layer | What BenchFlow does |
|----------------------|-------|---------------------|
| In a framework BenchFlow speaks *inbound* — Harbor (task-dir adapter) or PrimeIntellect / Verifiers (hosted) | **1 — native** | Runs it in its supported form; correctness is inherited from the original format |
| In a variant of a known format, or a format BenchFlow has never seen | **2 — translated** | Translates it to the native `task.md` format, then *proves* equivalence with a parity gate |
| A one-off harness with its own runner and scoring, no reusable adapter | **3 — as-is** | Runs the benchmark under its own harness and interfaces with its output only |

All three layers terminate at the same scored-trajectory contract (see
[The seam](#the-seam-one-scored-trajectory-contract) below). One ingestion,
every benchmark.

Inspect AI and ORS / OpenReward are deliberately **not** in the table above:
BenchFlow has no inbound run path for them. They are *outbound* export targets —
BenchFlow writes its own native results into those formats — covered under
[Layer 1's outbound format seams](#layer-1--supported-framework--run-natively).

---

## Layer 1 — supported framework → run natively

When a benchmark already speaks a format BenchFlow reads, BenchFlow runs it in that form and converts the output to results. For Harbor tasks this is a claim about the file format only; each task still has to pass BenchFlow's runtime checks, and results are not guaranteed to match a Harbor run (different sandboxes, agent adapters and verifier hardening).

**Harbor task directories.** `bench eval run` loads a Harbor `task.toml` + `instruction.md` directory directly with the task loader; no adapter runs. BenchFlow reads `[task]`, `[metadata]`, `[agent]`, `[verifier]`, `[environment]` (as its `sandbox`) and `[solution.env]`; genuinely unknown keys outside the decision-bearing tables are ignored with a warning (`task.config.ignored_keys`, and a `bench tasks check` warning), but a probable typo of a known key or an unknown key inside a table that decides isolation, network, timeouts, resources or grading (`[environment]`/`sandbox`, `[agent]`, `[verifier]`, `[steps]`) is refused with a did-you-mean rather than silently defaulted (set `BENCHFLOW_TASK_TOML_ALLOW_UNKNOWN_KEYS=1` to override); keys it parses but cannot honour are refused before launch: multi-step `steps`, root `artifacts` off Docker and Daytona, a separate verifier environment (`verifier.environment_mode = "separate"`, `[verifier.environment]`) off Docker and Daytona ([Separate verifier sandboxes](./separate-verifier.md)), `[[verifier.collect]]`, verifier network allowlists (agent and environment allowlists run on `docker` and `daytona`, see [sandbox hardening](./sandbox-hardening.md#network-policy-allowlist-egress)), Windows, TPUs and GPUs (see [Running benchmarks](./running-benchmarks.md#running-foreign-benchmarks-harbor-task-directories) for the full list). [`HarborAdapter`](../src/benchflow/adapters/harbor.py) / [`detect_adapter()`](../src/benchflow/adapters/inbound.py) are a Python-only reader that returns an `InboundTask` and keeps unknown keys in `compatibility.config_extra`; they are not on the run path.

**Hosted environments (run on their own native surface).** External
PrimeIntellect / Verifiers environments are not BenchFlow task directories and do
not use BenchFlow's Docker/Daytona sandbox runner. BenchFlow runs them through
their native Verifiers execution surface and preserves their hosted identity
(`env_uid`, `hub_url`), while still writing the shared output contract — see
[`src/benchflow/hosted_env.py`](../src/benchflow/hosted_env.py):

```bash
bench eval run \
    --source-env primeintellect/general-agent \
    --source-env-version 0.1.1 \
    --model google/gemini-2.5-flash-lite
```

BenchFlow installs the environment package (from Prime's package index, `https://hub.primeintellect.ai/primeintellect/simple/`) into a fresh venv, reads which `vf-eval` the installed verifiers ships, and speaks it:

- **Legacy CLI** (verifiers 0.3.1 and older, the Hub's v0 environments): `--env-args`, `--num-examples`, `--sampling-args`, `--save-results`; the reward is the `reward: avg` line of its summary.
- **v1 CLI** (the verifiers 0.3.2 dev line, the Hub's v1 tasksets): the taskset id, `-n`/`-r`/`-c`, `--sampling.*`, `--env.taskset.<key>` for each `--source-env-arg`, and always `--no-push` (a v1 run uploads to the Prime platform by default; BenchFlow runs stay local). The reward is the mean over the rollouts of `traces.jsonl`, each the sum of its weighted rewards; episodes that errored are left out and counted, and if every episode errored the run is an error, never a 0 reward. Token counts come from the traces.

A v1 taskset cannot run on the legacy CLI (it has no `load_environment`), and v0 environments were removed from v1, so each environment runs on the verifiers its package requires; `--source-env-verifiers-version` pins that release for reproducibility. Tested against verifiers 0.3.2.dev releases with a stub model endpoint; a run of a real Hub environment needs access to Prime's package index and a model key (`PRIME_API_KEY` for Prime inference, or `--source-env-base-url` / `--source-env-api-key-var` for another OpenAI-compatible endpoint).

**Outbound format seams (native results → other frameworks).** Results also
round-trip *out* into the frameworks teams already use:
[`InspectAdapter`](../src/benchflow/adapters/inspect_ai.py) maps a BenchFlow
`Scene` + `Rubric` into an Inspect AI task, and
[`ORSAdapter`](../src/benchflow/adapters/ors.py) maps `VerifyResult` /
`RewardEvent` into the ORS (OpenReward) reward-record shape.

---

## Layer 2 — unknown or variant format → translate, then validate or prove parity

When a benchmark's format is **not** one BenchFlow speaks natively — a variant of
a known layout, or an entirely new one — BenchFlow translates it to the native
`task.md` format. There are **two distinct translation flows**, and they check the
result in **different** ways. They do not chain together; pick the one that
matches what you are translating:

| You are translating… | Command path | How the result is checked |
|----------------------|--------------|---------------------------|
| A task you already control, in the legacy split layout (`task.toml` + `instruction.md`) | `bench tasks migrate` → `bench tasks check` | Structural validation of the generated `task.md` (config equivalence is enforced at conversion time) |
| A foreign benchmark with no reusable adapter | `bench eval adopt <source>` → `bench eval adopt <name> --verify` | The parity gate **proves** the converted benchmark reproduces the original's results |

The two paths are not interchangeable: `bench eval adopt <name> --verify` runs only
against a benchmark *adopted* with `bench eval adopt <source>`. It reads
`benchmarks/<name>/parity_experiment.json` and errors `benchmark not adopted …
run bench eval adopt first` on anything else — including a migrated `task.md`.
A migrated `task.md` is validated with `bench tasks check`, never with `--verify`.

### (a) Migrate a task you control → validate with `bench tasks check`

`bench tasks migrate <dir>` converts a legacy split layout (`task.toml` +
`instruction.md`) into the unified `task.md` format
([`cli/main.py`](../src/benchflow/cli/main.py) → `migrate_task_to_task_md`). The
conversion checks config equivalence before writing, so it cannot silently drop
supported task configuration. Validate the generated package with:

```bash
bench tasks check <dir>
```

That is the structural gate (`check_task` in
[`_utils/task_authoring/`](../src/benchflow/_utils/task_authoring/__init__.py)): it
rejects unreplaced `[REPLACE: …]` placeholders, missing required files, and a
verifier package with no runnable entrypoint. This flow records no
`parity_experiment.json` and runs no parity gate — it is a faithful in-place
format conversion of a task you already own.

### (b) Adopt a foreign benchmark → `bench eval adopt <source>`, then prove with `bench eval adopt <name> --verify`

For a foreign benchmark with no reusable adapter, the benchmark-adoption router
in [`src/benchflow/agent_router.py`](../src/benchflow/agent_router.py) drives the
work as a single multi-mode command:

- `bench eval adopt <source>` scaffolds `benchmarks/<name>/` to the reference
  layout (if missing), then drives the conversion workflow with an agent toward a
  `benchmarks/<name>/` pull request. The conversion guide is embedded in the
  command. (`bench eval adopt <name> --scaffold-only` writes just the package.)

Only a benchmark adopted this way carries the `benchmarks/<name>/` directory the
parity gate below requires.

### Prove — the parity gate

`bench eval adopt <name> --verify` closes the adopt → verify loop. It is a **parity-only**
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
`benchmarks/<name>/parity_experiment.json`. `bench eval adopt <name> --verify` reads
and scores that file when it is a JSON object in the shape the scaffold writes: it
pulls per-criterion verdict pairs and legacy-vs-converted reward samples from the
object and emits a verdict. A file that records neither yields no comparisons, so
the gate returns `insufficient-evidence`.

The repository ships several recorded experiments under
[`benchmarks/*/parity_experiment.json`](../benchmarks/), and they are **not**
uniform — do not assume `verify` scores all of them. `bench eval adopt
programbench --verify` reads recorded reward-distribution samples and reports
`parity-confirmed` (max abs reward delta within the default `0.02` tolerance).
Other shipped experiments record structural- and eval-parity notes the gate does
not read as criteria or reward samples, so `verify` returns
`insufficient-evidence` for them; and one predates this object contract and
stores a top-level list of experiment runs, which the gate cannot score. Use
`verify` as the gate for experiments recorded in the object shape, and read the
JSON files directly for the rest.

Read the recorded experiments honestly: where rewards are recorded they report
aggregate deltas **within tolerance**, with residual disagreements triaged to
causes such as model non-determinism rather than conversion defects. The
guarantee is "parity within tolerance, divergences triaged, no conversion defect
found" — not a fixed headline percentage.

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
| `results.jsonl` | Verifiers/Prime-RL shaped rollout row |
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
ingestion path serves every benchmark: release checks, trainers, and downstream
reporting tools read the same files whether the run came from a native adapter, a parity-proven
translation, or a bespoke harness.

---

## Where to go next

- [Getting started](./getting-started.md) — install and run your first eval
- [Concepts](./concepts.md) — Rollout / Scene / Role / Verifier
- [Native `task.md` authoring](./task-authoring-task-md.md) — the translation target for Layer 2
- [Architecture](./architecture.md) — adapters and trainers as the edges of the system
