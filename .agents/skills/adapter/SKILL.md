---
name: adapter
description: Adopt, convert, verify, and publish upstream benchmarks as BenchFlow benchmarks. Use when asked to port a benchmark, create a BenchFlow adapter, write or review benchmarks/<name>/benchflow.py, run parity, route a benchmark through L1/L2/L3, or decide whether a benchmark should run natively, be translated, or run as-is.
---

# BenchFlow Adapter Adoption

BenchFlow's adapter rule is simple: do not convert a benchmark unless conversion is the right layer.

A benchmark enters through the adoption router and must resolve into one of three layers:

1. **Layer 1 — native / supported framework**

   * Use the ecosystem's own supported surface.
   * Do not rewrite the benchmark.
   * Correctness is inherited from the original framework or runner.
   * Examples: Harbor-style task directories, hosted PrimeIntellect / Verifiers environments, or other first-class inbound sources supported by the current BenchFlow version.

2. **Layer 2 — adopt / translate, then prove parity**

   * Use when the source is a variant of a known format or an unknown benchmark with a reusable task structure.
   * Convert into BenchFlow task packages.
   * Prove the conversion with deterministic side-by-side parity and reward-distribution parity.
   * The conversion must reproduce the original benchmark's behavior on identical inputs. Do not improve, sanitize, or "fix" the benchmark unless the task explicitly asks for a separate hardening pass.

3. **Layer 3 — bespoke / run as-is**

   * Use when the benchmark has a one-off runner, custom agent loop, or non-portable harness.
   * Keep the original harness unchanged.
   * Interface only with its output and map the result into BenchFlow's scored-trajectory contract.

Arguments passed: `$ARGUMENTS`

---

## Operating rule

Never start by writing converter code. First classify the source.

For every benchmark, produce a short routing note:

```text
Source:
Observed format:
Original runner:
Original verifier/scorer:
Task unit:
Oracle availability:
Proposed layer: L1 | L2 | L3
Reason:
Parity evidence required:
```

Only proceed to adapter implementation after this routing note is clear.

---

## Dispatch

### `classify <source>`

Inspect the source repository, dataset, or task directory.

Check:

* Does it already contain a supported signature file such as `task.toml`?
* Is it a hosted environment with its own native runner?
* Is the benchmark a structured collection of tasks that can be converted?
* Is there an original verifier, scorer, judge prompt, or evaluation CLI?
* Are oracle solutions available?
* Does the benchmark require stateful services, snapshots, multiple actors, or a custom loop?

Return one of:

```text
L1 native: run through existing inbound/hosted support.
L2 adopted: create benchmarks/<name>/ converter + parity gate.
L3 as-is: wrap/import original harness output, do not translate task logic.
```

### `init <name>`

Use the canonical adoption command:

```bash
bench eval adopt init <name>
```

Do not use deprecated `bench agent create|run|verify` unless the installed BenchFlow version only supports the legacy alias.

The scaffold should contain:

```text
benchmarks/<name>/
├── benchflow.py
├── main.py
├── parity_test.py
├── parity_experiment.json
├── benchmark.yaml
├── run_<name>.py
├── <name>.yaml
└── README.md
```

Treat `benchflow.py` as the source of truth for conversion.

### `convert <source> --name <name>`

Use the router first:

```bash
bench eval adopt convert <source> --name <name> --dry-run
```

Review the generated Codex command and prompt. Then run the live conversion only when credentials and workspace state are correct:

```bash
bench eval adopt convert <source> --name <name>
```

The conversion must implement:

```text
benchmarks/<name>/benchflow.py
benchmarks/<name>/main.py
benchmarks/<name>/parity_test.py
benchmarks/<name>/parity_experiment.json
benchmarks/<name>/benchmark.yaml
benchmarks/<name>/<name>.yaml
benchmarks/<name>/README.md
```

For the generated task package, follow the contract used by the current scaffold. In older or compatibility paths, the generated task directory is Harbor-style:

```text
<task-id>/
├── task.toml
├── instruction.md
├── environment/
│   └── Dockerfile
├── tests/
│   └── test.sh
└── solution/
    └── solve.sh        # optional
```

In native `task.md` paths, the task package is:

```text
<task-id>/
├── task.md
├── environment/
│   └── Dockerfile
├── verifier/
│   └── test.sh
└── oracle/
    └── solve.sh        # optional
```

Use the format expected by the installed BenchFlow version. When converting from legacy split layout to native `task.md`, validate with:

```bash
bench tasks migrate <task-dir> --remove-legacy
bench tasks check <task-dir>
```

### Converter requirements

The converter must support:

```bash
python benchmarks/<name>/main.py \
  --source-dir <source> \
  --output-dir <tasks-output> \
  --limit <n> \
  --overwrite \
  --task-ids <id1,id2,...>
```

Implementation rules:

* Sanitize task IDs into stable lowercase slugs.
* Preserve the upstream task identity in metadata.
* Keep generated task names stable across reruns.
* Put all agent-visible instructions in `instruction.md` or `task.md`.
* Never expose hidden tests, answer keys, judge prompts, or verifier-only files to the agent.
* Copy only task-needed assets into the environment.
* Put evaluation logic under `tests/` or `verifier/`, depending on the task format.
* `test.sh` must write a numeric reward to `/logs/verifier/reward.txt`.
* If partial credit exists upstream, preserve the upstream scoring formula.
* If the upstream benchmark has known reward-hacking behavior, reproduce it for parity; do not silently harden it during conversion.

### `verify <name>`

Run the parity gate:

```bash
bench eval adopt verify <name>
```

Use rerun mode when available and appropriate:

```bash
bench eval adopt verify <name> --rerun
```

The gate has two layers:

1. **Deterministic side-by-side parity**

   * Original verifier and converted verifier run on identical outputs.
   * Per-criterion verdicts must agree.
   * Record `original_verdict`, `adapted_verdict`, and `agreement`.

2. **Reward-distribution parity**

   * Legacy and converted rewards are compared across representative samples.
   * Reward deltas must be within tolerance, normally `0.02`.
   * Every divergence must be triaged.

The parity file should use the scoreable object shape:

```json
{
  "experiment": "side-by-side-parity",
  "benchmark": "<name>",
  "status": "recorded",
  "judge_model": "",
  "conversion_parity": {
    "tasks": [
      {
        "task_id": "example-001",
        "n_criteria": 1,
        "criteria_results": [
          {
            "criterion_id": "C-001",
            "original_verdict": "pass",
            "adapted_verdict": "pass",
            "agreement": true
          }
        ]
      }
    ]
  },
  "reward_distribution_parity": {
    "samples": [
      {
        "task_id": "example-001",
        "legacy_reward": 1.0,
        "converted_reward": 1.0
      }
    ]
  }
}
```

Do not claim parity from a JSON file that has no scoreable comparisons. A missing, half-recorded, or unknown schema is not a pass.

### `run <name>`

After parity is confirmed, run the converted benchmark like a normal BenchFlow benchmark:

```bash
bench eval run --config benchmarks/<name>/<name>.yaml
```

For local debugging, run a small subset first:

```bash
python benchmarks/<name>/main.py \
  --source-dir <source> \
  --output-dir /tmp/<name>-tasks \
  --limit 3 \
  --overwrite

bench eval run \
  --tasks-dir /tmp/<name>-tasks \
  --agent gemini \
  --model gemini-2.5-flash \
  --sandbox docker
```

Use `bench eval run` in adapter docs and examples. `bench eval create` is a
deprecated alias and should only appear when documenting legacy compatibility.

### `audit <name|jobs-dir>`

Before publishing, audit both the adapter and the run artifacts.

Check:

* The converter is deterministic.
* The generated tasks pass structural validation.
* The verifier writes `/logs/verifier/reward.txt`.
* Oracle solutions pass when available.
* Empty or trivial submissions fail unless the original benchmark also accepts them.
* Agent-visible files do not include hidden tests, gold answers, reward files, or judge-only prompts.
* `parity_experiment.json` is scoreable by `bench eval adopt verify`.
* `benchmark.yaml` accurately describes task count, categories, conversion method, verifier method, reward type, and parity evidence.
* The job config points to the correct task directory.
* The final run emits the standard BenchFlow trajectory contract.

---

## Failure path

If parity fails:

1. Do not publish.
2. Reproduce the divergence on the smallest task/sample.
3. Decide whether the divergence is:

   * converter bug,
   * verifier path mismatch,
   * environment mismatch,
   * upstream nondeterminism,
   * model/judge nondeterminism,
   * unsupported benchmark behavior.
4. Fix and rerun parity.
5. If parity still cannot close, emit a tracked issue draft for human review.

Issue draft template:

````markdown
## Benchmark adoption parity: <name>

Verdict: parity-divergent

### Source
- Upstream:
- Converted path:
- BenchFlow commit:
- Source commit/ref:

### Divergence
- Task:
- Criterion:
- Original verdict/reward:
- Converted verdict/reward:
- Delta:

### Reproduction
```bash
<minimal command>
````

### Suspected cause

<converter bug | verifier mismatch | environment mismatch | nondeterminism | unsupported behavior>

### Requested human action

<what maintainer/partner should decide>

````

Never hide a divergence in aggregate metrics.

---

## Definition of done

A BenchFlow adapter is done only when all are true:

```text
[ ] Routing note says L1, L2, or L3 and explains why.
[ ] benchmarks/<name>/ scaffold exists.
[ ] benchflow.py deterministically converts source tasks.
[ ] main.py delegates to the converter.
[ ] Generated tasks are structurally valid.
[ ] Verifier writes numeric reward.txt.
[ ] Oracle solutions pass when available.
[ ] parity_test.py implements structural, eval, and side-by-side checks.
[ ] parity_experiment.json contains scoreable comparisons.
[ ] bench eval adopt verify <name> returns parity-confirmed.
[ ] benchmark.yaml and README document the conversion honestly.
[ ] <name>.yaml can run the converted tasks.
[ ] Published evidence includes parity results and run trajectories.
````

The final claim should be:

```text
Parity confirmed within recorded evidence; divergences triaged; no conversion defect found.
```

Do not claim:

```text
100% correct
zero defects
better than the original
hardened against reward hacking
```

unless those properties were separately tested and documented.
