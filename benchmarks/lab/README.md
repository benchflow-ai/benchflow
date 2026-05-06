# LAB → BenchFlow adapter

Translates [Harvey AI's Legal Agent Bench (LAB)](https://github.com/harveyai/harvey-labs)
into the BenchFlow task format so that any ACP agent can be evaluated against
LAB's 1,251 legal-work tasks under BenchFlow's standard sandbox + verifier
pipeline.

LAB ships its own Python harness with 6 tools (bash/read/write/edit/glob/grep),
podman sandbox, all-pass rubric scoring, and an LLM judge. This adapter keeps
the rubric semantics intact and replaces the surrounding harness with
BenchFlow's: same instructions, same documents, same pass-or-fail criteria,
generated as a `task.toml` + `instruction.md` + `environment/Dockerfile` +
`tests/test.sh` package per task.

## Layout

```
benchmarks/lab/
├── benchflow.py                  # CLI: translate / list / check
├── adapter/
│   ├── __init__.py
│   └── translate.py              # core translation
├── lab.yaml                      # benchflow run config (Gemini 3.1 flash lite)
├── parity_experiment.json        # parity validation results (per harbor recipe)
├── scripts/
│   ├── parity_subset.txt         # 8-task sanity-check subset
│   └── run_parity.py             # one-shot parity runner
└── README.md
```

## Quickstart

```bash
# 1. Materialise BenchFlow tasks from a fresh harvey-labs clone.
python benchmarks/lab/benchflow.py translate --output-dir /tmp/lab-tasks
# Or just a subset:
python benchmarks/lab/benchflow.py translate \
    --output-dir /tmp/lab-tasks \
    --task-list benchmarks/lab/scripts/parity_subset.txt

# 2. Run benchflow over a single generated task (Docker backend).
GEMINI_API_KEY=$KEY bench run \
    /tmp/lab-tasks/corporate-ma__analyze-cim-deal-teaser/ \
    --agent gemini --model gemini-3.1-flash-lite-preview --backend docker

# 3. Run a sweep across the subset.
GEMINI_API_KEY=$KEY bench run /tmp/lab-tasks/ \
    --config benchmarks/lab/lab.yaml
```

## What gets translated

For each LAB task at `tasks/<area>/<slug>[/<scenario>]/`:

| LAB source | BenchFlow target |
| --- | --- |
| `task.json[title, work_type, tags]` | `task.toml [metadata]` |
| `task.json[instructions]` (or `instructions.md`) | `instruction.md` (with workspace preamble) |
| `task.json[criteria]` (the rubric) | `tests/criteria.json` (read by the judge) |
| `documents/` | `environment/documents/` (COPYed read-only into image) |
| LAB's `evaluation/scoring.py` (all-pass) | `tests/test.sh` + `tests/rubric_judge.py` (all-pass, Gemini) |

The verifier writes `1.0` or `0.0` to `/logs/verifier/reward.txt` exactly as
LAB's scoring writes `score = 1.0 if all_pass else 0.0`.

The agent prompt is `instruction.md` = workspace preamble + the unmodified
LAB instructions. No skill manuals or system-prompt scaffolding are added,
so the parity surface is just BenchFlow's `instruction.md` vs. LAB's
preamble + skills bundle. (See "Parity caveats" below for what this means
in practice.)

## Why a separate `rubric_judge.py` per task?

BenchFlow's verifier contract is: `tests/test.sh` runs inside the verifier
container and writes `/logs/verifier/reward.txt`. To run an LLM judge from
inside that container, the judge code (and its rubric) has to be on the
container filesystem before `bench run` starts. The translator therefore
copies a self-contained `rubric_judge.py` into every generated task's
`tests/` directory; it ships no shared adapter library, only the
`google-genai` SDK already pinned in the Dockerfile.

The judge's defaults are the same as the parity runner's: model =
`$LAB_JUDGE_MODEL` (default `gemini-3.1-flash-lite-preview`), temperature
= 0.0, response forced to JSON via `response_mime_type`, prompt template
identical across the two scoring paths.

## Parity validation (Harbor recipe)

This adapter follows the Harbor parity playbook:

1. **Sanity check on 5–10 tasks (both sides).** Done — see `parity_experiment.json`. `scripts/parity_subset.txt` lists 8 LAB tasks chosen for diversity of work_type and document complexity.
2. **One full run (both sides).** Wired but not executed in this PR — see "Compute budget" below.
3. **Three runs (both sides).** Wired but not executed.

Reporting format follows the Harbor convention: `mean ± sample SEM` across
runs, with the matching criterion that **the two side ranges must overlap**:

```
max(lab_runs) >= min(bench_runs) AND max(bench_runs) >= min(lab_runs)
```

### Pre-run checklist (held identical across both arms)

| Item | Setting |
| --- | --- |
| LAB git ref | `harveyai/harvey-labs@main` (sha pinned in `parity_experiment.json`) |
| BenchFlow ref | `benchflow-ai/benchflow@feature/lab-adapter` |
| Agent | one-shot Gemini call (sanity arm) → `gemini` ACP agent (full arm) |
| Agent model | `gemini-3.1-flash-lite-preview` |
| Judge model | `gemini-3.1-flash-lite-preview` |
| Judge prompt | identical across arms — see `gemini_judge` in `scripts/run_parity.py` |
| Temperature | 0.0 (agent + judge) |
| Verifier semantics | all-pass: `reward = 1.0` iff every criterion verdict is `pass` |

### Sanity arm — observed results

Run on `harveyai/harvey-labs@7daf1ac`, BenchFlow `feature/lab-adapter`, one
`gemini-3.1-flash-lite-preview` call per task (temperature 0) for both
generation and judging. 1 run × 8 tasks × 520 criteria total.

| Metric | Value |
| --- | --- |
| All-pass reward agreement | **8/8** tasks (both arms = 0.0) |
| Per-criterion verdict agreement | **510/520** (98.1%) |
| Range overlap (harbor matching criterion) | ✓ |
| Wall clock (one full pass, both arms) | 191 s |

The 10 disagreements are Gemini temperature-0 non-determinism on borderline
criteria, distributed as 1–5 flips on 3/8 tasks. Full per-task numbers
live in `parity_experiment.json`.

### Sanity-check parity arm

`scripts/run_parity.py` exercises end-to-end translation fidelity without
needing podman/Docker permissions on the host. For each task:

1. Reads source documents with the same extractors LAB and the BenchFlow
   verifier use (pandoc, pdfplumber, pandas, markitdown).
2. Sends `instructions + documents` to Gemini once, parses the reply into
   the declared deliverables.
3. Scores the produced output against the rubric **two ways** —
   - LAB-native: rubric loaded directly from `tasks/.../task.json`,
     passed through the same Gemini judge.
   - BenchFlow: invokes the translated task's `tests/rubric_judge.py`
     subprocess, identical to what the BenchFlow verifier container will
     run.
4. Compares per-criterion verdicts and the all-pass reward.

This isolates *translation* parity (instructions / documents / rubric
mapping / deliverable matching) from *agent* parity. The full agentic
arm (the ACP gemini agent vs. LAB's `harness.run`) re-uses the same I/O
contract — see `scripts/run_parity.py:main` for where to swap in
`bench run` and `python -m harness.run`.

### Debug playbook

Per Harbor's recommendation, when sanity arm scores diverge:

1. Resolve infra errors first (judge timeouts, API throttling).
2. Inspect agent output — both arms see the same files; a discrepancy
   here points at the `_resolve_deliverables` fuzzy matcher.
3. Per-criterion overlap analysis — `summary.json` has a `per_task` block
   with both arms' verdicts; diff them.
4. Distinguish randomness from systematic error — re-run with the same
   seed; Gemini at temperature 0 is reproducible enough to make
   single-criterion flips obvious.
5. Lock configuration once stable, then scale.

## Compute budget

Running every step of the harbor recipe is a real money-and-time spend:

| Arm | Cost driver | Per-task time | Per-task cost |
| --- | --- | --- | --- |
| Sanity (one-shot) | 1 generation + N judge calls | ~30 s | <$0.01 |
| Full (ACP agent) | 20–50 turn agent loop | 5–15 min | $0.05–$0.20 |

For the full corpus (1,251 tasks) with three runs both sides, that's
~7,500 ACP-agent runs. Run that on Daytona/Modal, not on a single host.
The `lab.yaml` config is parameterised so the same file drives the
sanity, full, and three-run sweeps via `--runs`.

## Parity caveats

- **Agent surface differs.** LAB's harness exposes 6 hand-written tools;
  the BenchFlow `gemini` ACP agent uses the gemini CLI's native tool
  surface. Score parity is therefore *framework parity given the same
  agent capability*, not a guarantee of identical traces.
- **Skill manuals are dropped.** LAB ships `harness/skills/{docx,xlsx,pptx}`
  manuals that get loaded into its system prompt. The translated
  BenchFlow tasks expose the same tools (pandoc / openpyxl / python-pptx
  in the Dockerfile) but don't auto-mount the manuals — agents that need
  them can be passed via `--skills-dir` at run time.
- **No oracle path.** LAB tasks are open-ended drafting; there is no
  reference solution. `solution/solve.sh` is an empty stub that exits 0
  so `bench run --agent oracle` doesn't crash.
- **Judge-side variance.** The all-pass scoring rule means a single
  flipped verdict on any criterion drives a task from 1.0 → 0.0.
  Per-criterion verdict comparison (in `summary.json`) is the
  fine-grained signal; treat the all-pass reward as a coarse summary.
