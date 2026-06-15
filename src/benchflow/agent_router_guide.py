"""The benchmark-conversion guide embedded into the ``bench eval adopt`` prompt.

This replaces the former ``benchmarks/CONVERT.md`` doc: the guide now lives in
code as prompt content so it stays versioned with the adoption router and the
scaffold it describes. Keep it in sync with :mod:`agent_router_scaffold` and the
reference benchmark ``benchmarks/programbench/``.
"""

from __future__ import annotations

CONVERSION_GUIDE = """\
# Benchmark conversion guide

You are converting an upstream benchmark into a BenchFlow benchmark under
`benchmarks/<name>/`. The deliverable is a *converter* that turns each source
instance into a BenchFlow *task package*, plus the *evidence* that the
conversion is faithful (a parity gate reproducing the original's scoring).

The single `bench eval adopt` command runs this pipeline:
- `bench eval adopt <source>` scaffolded `benchmarks/<name>/` (if missing) and is
  driving this conversion: implement the converter, generate tasks, record parity.
- `bench eval adopt <name> --verify`: the parity gate scores
  `parity_experiment.json` and emits a verdict.

## 1. What a BenchFlow task package is

The converter emits one task directory per source instance. The native format
is a single **`task.md`** — config front-matter plus the agent-facing
instruction in one document — alongside sidecar dirs. The legacy **split layout**
(`task.toml` + `instruction.md`, with `tests/` and `solution/`) remains a
compatibility form and is what the `programbench` reference converter emits;
generate whichever maps more cleanly from the source. Either way a task carries:

- **config** — `name`, `[metadata]`, `[agent]` (`timeout_sec`, `user`,
  `network_mode`), `[verifier]` (`timeout_sec`, `environment`), and resource
  limits. In `task.md` this is the front-matter; in the split layout it is
  `task.toml`.
- **instruction** — exactly what the agent must do, in the task's own terms; the
  agent sees only this plus the environment. The `task.md` body, or
  `instruction.md` in the split layout.
- `environment/Dockerfile` — the sandbox image. Pin a base that reproduces the
  source's execution environment and pre-install everything the agent needs, so
  the build happens before the agent runs (and is therefore trusted).
- **verifier** — the reward check that runs in the sandbox after the agent
  finishes and writes a single reward to `reward.txt` (a float in `[0, 1]`;
  richer verifiers may also emit `reward.json` / `ctrf.json`). Native name
  `verifier/`; the split layout uses `tests/test.sh`. It is trusted code shipped
  with the task — it scores the agent; it is not the agent. Copy the source's
  test manifests/graders in verbatim where possible so scoring cannot drift.
- **oracle** (optional but recommended) — a reference that *passes* the verifier,
  used to prove the task is solvable and to calibrate the reward. Native name
  `oracle/`; the split layout uses `solution/`. Keep the real answer out of the
  agent-visible workspace.

Faithfulness rule: the BenchFlow task must score the agent the **same way the
source benchmark does** — same tests, same anti-cheat, same reward formula.
Never "improve", sanitize, or make a task easier than the original (including
any reward-hackability the original has — parity reproduces, it does not fix).

## 2. Implement the converter (`benchflow.py`)

`benchflow.py` reads the source benchmark and writes task dirs under
`--output-dir`. Two entry points are already stubbed by `init`:
- `convert(source_instance, output_dir, *, overwrite=False) -> Path` — one
  instance -> one task dir.
- `convert_all(source_dir, output_dir, *, overwrite=False, limit=None,
  task_ids=None) -> list[Path]` — iterate the source set.

For each instance, map the source's fields onto the task package (split-layout
filenames below — the form `programbench` emits; for a native `task.md` the
instruction and config collapse into the one document's body + front-matter):
- prompt / spec / docs -> `instruction.md` (or the `task.md` body)
- environment / image / setup -> `environment/Dockerfile`
- test harness / grader -> `tests/test.sh` (+ any scripts it needs, in `tests/`)
- reference / gold solution -> `solution/`
- id / language / difficulty / timeouts / resources -> `task.toml` `[metadata]`
  and limits (or the `task.md` front-matter)

Keep the converter deterministic and re-runnable (honor `--overwrite`). See
`benchmarks/programbench/benchflow.py` for a complete worked converter and
`benchmarks/programbench/README.md` for a field-by-field source->BenchFlow
mapping.

## 3. Describe the benchmark (`benchmark.yaml`)

Fill in `benchmark.yaml`: `name`, `description`, `url` / `paper` / `repo`,
`author`, `converted_by: BenchFlow`, the task `count` / `languages` / `splits`,
and the `conversion` block (the `script` / `generator` and their flags). Every
`benchmarks/<name>/` ships this descriptor; job configs (how to *run* it) live
in separate YAML files.

## 4. Prove parity (`parity_test.py` + `parity_experiment.json`)

A faithful conversion REPRODUCES the original's verdicts on identical inputs.
`parity_test.py` runs the comparison and records evidence into
`parity_experiment.json`. Cover the layers that apply:
- **Structural parity** — every generated task has the required files + metadata.
- **Eval parity** — run the verifier on a known-good (oracle) and a known-bad
  output; rewards land where the source says they should.
- **Side-by-side / pipeline parity** — for a sample of tasks, run the original
  scoring and the BenchFlow verifier on the *same* artifact and record both
  outcomes: per-criterion `original_verdict` / `adapted_verdict` pairs, or
  `tests_passed` / `tests_total` / `reward` with a `parity` marker (e.g.
  `exact_match`). See programbench's `parity_experiment.json` for the shape.

`bench eval adopt <name> --verify` scores `parity_experiment.json`: every compared
criterion's converted verdict must match the original's, and every
legacy-vs-converted reward delta must sit within tolerance (default `0.02`,
`--tolerance`). It emits `parity-confirmed`, `parity-divergent`, or
`insufficient-evidence` (a layer with no data does not block; no data at all ->
insufficient). Divergences are rendered into a draft issue for a human — never
auto-filed, never hidden.

## 5. Definition of done

- `benchmarks/<name>/` has `benchflow.py`, `main.py`, `parity_test.py`,
  `run_<name>.py`, `<name>.yaml`, `benchmark.yaml`, `parity_experiment.json`,
  and `README.md`.
- `python -m benchmarks.<name>.main --output-dir ...` generates valid task
  packages (`bench tasks check <task>` passes on a sample).
- `parity_experiment.json` records real evidence and
  `bench eval adopt <name> --verify` reports `parity-confirmed` (or a triaged,
  explained divergence).
- `README.md` documents the source->BenchFlow mapping (see programbench's README
  for the bar).
- Open a pull request adding `benchmarks/<name>/`.
"""
