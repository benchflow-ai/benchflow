# Benchmark adoption

Adopt an upstream benchmark into a BenchFlow benchmark with `bench adopt`.

## What the router is

`bench adopt` is the benchmark-adoption router. It *routes* an external
benchmark into `benchmarks/<name>/` — scaffold, codex-driven conversion, and a
parity gate — so the result is a first-class BenchFlow benchmark. It sits
upstream of evaluation: the router *adopts*, while `bench eval create` *runs*
the resulting tasks. Once `bench adopt verify <name>` reports
`parity-confirmed`, you point `bench eval create` at the converted tasks and run
them like any other benchmark.

(These commands were `bench agent create|run|verify` before 0.6; the old names
still work as deprecated aliases through 0.6 and are removed in 0.7.)

Three subcommands form the adopt → verify loop:

```
$ bench adopt --help
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ init      Scaffold benchmarks/<name>/ for a new benchmark adoption.          │
│ convert   Drive the CONVERT.md workflow by launching the host codex CLI.     │
│ verify    Run the parity gate for an adopted benchmark; emit a verdict.      │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The reference for what a finished adoption looks like is
[`benchmarks/programbench/`](../benchmarks/programbench/); the conversion
contract is [`benchmarks/CONVERT.md`](../benchmarks/CONVERT.md). The router
embeds both into the conversion workflow for you.

## `bench adopt init <name>`

`init` writes a deterministic scaffold under `benchmarks/<name>/`, matching
the reference layout and the CONVERT.md contract. Use `--benchmarks-dir` to
target a directory other than the repo's `benchmarks/`:

```
$ bench adopt init webarena-lite --benchmarks-dir /tmp/router-docs/benchmarks
Scaffolded /tmp/router-docs/benchmarks/webarena-lite
  README.md
  __init__.py
  benchflow.py
  benchmark.yaml
  main.py
  parity_experiment.json
  parity_test.py
  run_webarena_lite.py
  webarena-lite.yaml
```

That produces this tree:

```
webarena-lite/
├── __init__.py
├── benchflow.py            # converter: source instances → BenchFlow task dirs
├── main.py                 # converter CLI delegator
├── parity_test.py          # structural / eval / side-by-side parity checks
├── parity_experiment.json  # recorded parity results (read by verify)
├── benchmark.yaml          # standard benchmark descriptor
├── run_webarena_lite.py    # runner: convert, then evaluate via BenchFlow
├── webarena-lite.yaml      # BenchFlow job config (how to run)
└── README.md               # generated workflow notes
```

What each file is for:

- **`benchflow.py`** — the converter. Its documented `convert()` /
  `convert_all()` entry points are `NotImplementedError` stubs that point at
  CONVERT.md step 2; you fill them in to map each source instance to a
  BenchFlow task directory (`task.toml`, `instruction.md`,
  `environment/Dockerfile`, `tests/test.sh`).
- **`parity_test.py`** — the parity harness, with `--mode full | eval-parity |
  side-by-side` (CONVERT.md steps 3–5). Side-by-side parity records the
  per-criterion `original_verdict` / `adapted_verdict` pairs that `verify`
  scores.
- **`parity_experiment.json`** — the recorded parity results `verify` reads. The
  scaffold writes a `status: "template"` placeholder with empty
  `conversion_parity.tasks` and `reward_distribution_parity.samples`; you
  populate it from a real parity run.
- **`benchmark.yaml`** — the standard descriptor (name, conversion method,
  verification method, parity tallies). Fields start as `TODO`/`0`.

`main.py`, `run_webarena_lite.py`, and `webarena-lite.yaml` are the converter
CLI delegator, the convert-then-evaluate runner, and the BenchFlow job config
respectively.

### Fail-closed behavior

`init` refuses to overwrite an existing benchmark — re-running it is an error,
not a silent clobber:

```
$ bench adopt init webarena-lite --benchmarks-dir /tmp/router-docs/benchmarks
benchmark already exists: /tmp/router-docs/benchmarks/webarena-lite (refusing to
overwrite)
```

Names must be lowercase slugs (leading letter, single internal hyphens). The
slug is also the security floor — it keeps `init`/`verify` from being steered
outside `benchmarks/`. An uppercase or underscored name is rejected:

```
$ bench adopt init WebArena_Lite --benchmarks-dir /tmp/router-docs/benchmarks
invalid benchmark name 'WebArena_Lite': use a lowercase slug like 'my-bench'
(letters/digits, single internal hyphens, leading letter)
```

Both fail-closed cases exit non-zero.

## `bench adopt convert <source> [--name]`

`convert` drives the conversion. It assembles an adoption prompt — the source, the
target `benchmarks/<name>/` path, the adoption skills (CONVERT.md, the
programbench worked example, the parity harness), and the full embedded
CONVERT.md guide — then launches the host `codex` CLI to do the conversion
toward a pull request. If you omit `--name`, the slug is derived from the source
basename (so `.../webarena` becomes `webarena`).

Use `--dry-run` to print the exact command the router would launch without
running it:

```
$ bench adopt convert https://github.com/web-arena-x/webarena --name webarena-lite --dry-run
codex exec --cd /path/to/benchflow --skip-git-repo-check --sandbox workspace-write '# Benchmark adoption: webarena-lite

Adopt the source benchmark below into a BenchFlow benchmark by
following the conversion guide. Produce the converter, parity tests,
metadata, and task directories, then open a pull request.

Source benchmark: https://github.com/web-arena-x/webarena
Target directory: benchmarks/webarena-lite/

## Adoption skills
- conversion-guide: benchmarks/CONVERT.md
- reference-benchmark: benchmarks/programbench/ (worked example)
- parity-harness: parity_test.py + parity_experiment.json (verify gate)

## Conversion guide (benchmarks/CONVERT.md)

# Benchmark Conversion Guide
...
## Definition of done
- benchmarks/webarena-lite/ has benchflow.py, parity_test.py,
  parity_experiment.json, benchmark.yaml, run_webarena_lite.py,
  README.md
- `bench adopt verify webarena-lite` reports parity-confirmed'
```

The full prompt embeds CONVERT.md verbatim (elided above). The `codex exec`
argv is constructed deterministically: it runs in the repo root
(`--cd <repo>`), with `--skip-git-repo-check` and
`--sandbox workspace-write`. Pass `--model` to set the codex driver model and
`--codex-bin` to point at a different codex binary.

A live run (drop `--dry-run`) requires codex credentials and fails closed
without them — set `OPENAI_API_KEY` (or `CODEX_API_KEY`), or run `codex login`
to create `~/.codex/auth.json`. Without credentials `convert` errors before
assembling any context:

```
codex needs credentials to launch: set OPENAI_API_KEY (or CODEX_API_KEY), or run
`codex login` to create ~/.codex/auth.json
```

The codex run is the manual-validation step — it iterates on the converter and
parity tests until `bench adopt verify` confirms parity.

## `bench adopt verify <name>`

`verify` is the gate that closes the loop. It reads the adopted benchmark's
`parity_experiment.json` and emits a confidence verdict. The gate is *parity
only*: a faithful conversion must reproduce the original's behavior on identical
inputs — including any reward-hackability the original has. It never "improves"
or sanitizes the source.

It scores two layers:

- **Conversion parity (deterministic floor)** — every compared criterion's
  converted verdict must match the original's verdict on identical inputs.
- **Reward-distribution parity (statistical layer)** — every
  legacy-vs-converted reward delta must sit within `--tolerance` (default
  `0.02`).

A layer with no recorded data does not block the verdict. The three verdicts:

| Verdict | Meaning |
| --- | --- |
| `parity-confirmed` | Every recorded layer agrees; high-confidence the conversion is faithful. |
| `parity-divergent` | A criterion disagrees or a reward delta exceeds tolerance. |
| `insufficient-evidence` | No recorded comparisons at all — run `parity_test.py` and record results first. |

A freshly scaffolded benchmark has no recorded parity, so it is
`insufficient-evidence` and exits non-zero:

```
$ bench adopt verify webarena-lite --benchmarks-dir /tmp/router-docs/benchmarks
Verdict: insufficient-evidence
  conversion: 0/0 criteria agree (rate 0.0000)
Insufficient evidence: no recorded parity comparisons. Run parity_test.py and
record results before trusting the conversion.
...
```

### A parity-confirmed run

Populate `parity_experiment.json` from a parity run. `verify` reads
per-criterion verdicts under `conversion_parity.tasks` and reward samples under
`reward_distribution_parity.samples`:

```json
{
  "experiment": "side-by-side-parity",
  "benchmark": "webarena-lite",
  "status": "recorded",
  "judge_model": "gemini-3.1-flash-lite",
  "conversion_parity": {
    "tasks": [
      {
        "task_id": "shopping-001",
        "n_criteria": 2,
        "criteria_results": [
          {"criterion_id": "C-001", "original_verdict": "pass", "adapted_verdict": "pass", "agreement": true},
          {"criterion_id": "C-002", "original_verdict": "fail", "adapted_verdict": "fail", "agreement": true}
        ]
      },
      {
        "task_id": "reddit-002",
        "n_criteria": 1,
        "criteria_results": [
          {"criterion_id": "C-001", "original_verdict": "pass", "adapted_verdict": "pass", "agreement": true}
        ]
      }
    ]
  },
  "reward_distribution_parity": {
    "samples": [
      {"task_id": "shopping-001", "legacy_reward": 0.50, "converted_reward": 0.50},
      {"task_id": "reddit-002", "legacy_reward": 1.00, "converted_reward": 1.00}
    ]
  }
}
```

With every criterion agreeing and every reward delta at zero, the verdict is
`parity-confirmed` and `verify` exits zero:

```
$ bench adopt verify webarena-lite --benchmarks-dir /tmp/router-docs/benchmarks
Verdict: parity-confirmed
  conversion: 3/3 criteria agree (rate 1.0000)
  reward: max abs delta 0.0000 (tolerance 0.0200)
High-confidence: the converted evaluation reproduces the original's verdicts on
every compared criterion and stays within reward tolerance.
```

### A parity-divergent run

Flip one criterion so the converted verdict no longer matches the original
(here `C-002`'s `adapted_verdict` goes from `fail` to `pass`). The deterministic
floor trips, the verdict becomes `parity-divergent`, and `verify` prints a draft
GitHub issue body for the support path:

```
$ bench adopt verify webarena-lite --benchmarks-dir /tmp/router-docs/benchmarks
Verdict: parity-divergent
  conversion: 2/3 criteria agree (rate 0.6667)
  reward: max abs delta 0.0000 (tolerance 0.0200)
Divergence found: the conversion does not yet reproduce the original's behavior
— iterate, then open an issue for support.
## Benchmark adoption parity: webarena-lite

**Verdict:** parity-divergent

Divergence found: the conversion does not yet reproduce the original's behavior
— iterate, then open an issue for support.

### Conversion parity (deterministic floor)
- criteria compared: 3
- agreed: 2
- agreement rate: 0.6667
  - shopping-001/C-002: original=fail converted=pass

### Reward-distribution parity (statistical layer)
- samples: 2
- max abs delta: 0.0000
- tolerance: 0.0200

### Ask
Parity could not be closed for this conversion. The translation must
reproduce the original's behavior on identical inputs (including any
reward-hackability it has). This draft has NOT been filed — review it,
iterate on the converter, and open it manually if you need support.
```

The draft is **never filed automatically** — it is printed for a human to
review and open if they need support. Pass `--issue-out PATH` to write it to a
file instead of stdout:

```
$ bench adopt verify webarena-lite --benchmarks-dir /tmp/router-docs/benchmarks --issue-out /tmp/router-docs/divergence.md
Verdict: parity-divergent
  ...
Issue draft written to /tmp/router-docs/divergence.md
```

### The `--roundtrip-task` structural hook

By default `verify` scores the recorded `parity_experiment.json` at the
benchmark level. Pass `--roundtrip-task <task-dir>` to also run the structural
round-trip conformance check on one concrete task tree (it reuses the existing
split-layout round-trip parity utility). It is opt-in because that harness needs a
concrete task directory, which the benchmark-level verdict does not require.

`verify` exits non-zero for `parity-divergent` and `insufficient-evidence`, and
errors if the benchmark was never adopted:

```
$ bench adopt verify nonexistent-bench --benchmarks-dir /tmp/router-docs/benchmarks
benchmark not adopted: /tmp/router-docs/benchmarks/nonexistent-bench — run
`bench adopt init nonexistent-bench` first
```

## From adoption to evaluation

Once `verify` reports `parity-confirmed`, the benchmark is a normal BenchFlow
benchmark: run its tasks with `bench eval create` (see
[Running benchmarks](./running-benchmarks.md)), using the job config the
scaffold generated. The router's job ends at `parity-confirmed`; evaluation
takes it from there.
