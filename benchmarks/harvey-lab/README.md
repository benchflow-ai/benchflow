# Harvey LAB Benchmark

## Overview

[Harvey LAB](https://github.com/harveyai/harvey-labs) is an open-source benchmark for
evaluating agents on real legal work, maintained by Harvey AI. It contains **1,251 tasks
across 24 practice areas** (corporate M&A, real estate, IP, tax, antitrust, litigation,
ERISA, etc.). Each task gives the agent a synthetic data room of legal documents (contracts,
financial statements, disclosure schedules, board minutes, regulatory correspondence) and
asks for one or more deliverable files — typically a memo, table, or analysis — that are
graded against a structured rubric of pass/fail criteria by an LLM judge.

Tasks span four work types: `analyze` (490), `draft` (444), `review` (293), `research` (24).
Scoring is **per-criterion proportional**: reward = (criteria passed) / (total criteria).

## Features

- Auto-discovers all 1,251 tasks from the `tasks/` directory
- Mounts each task's `documents/` data room into the container at `/app/documents/`
- Verifier: Claude-based LLM judge, one call per rubric criterion, proportional scoring
- Docker base image pinned by digest for reproducibility
- Task selection: `--limit`, `--task-ids`, `--split parity` / `--split xlsx` / `--split <practice-area>`
- Harvey LAB harness ported as BenchFlow ACP agent (`harvey-lab-harness`) for true parity

## Generated Task Structure

Legacy split output is emitted only on request with `--task-format legacy`:

```
corporate-ma-review-data-room-red-flag-review/
├── task.toml                           # tags, work type, declared artifacts, timeouts
├── instruction.md                      # agent instructions (from upstream task.json)
├── environment/
│   ├── Dockerfile                      # digest-pinned python:3.13-slim + doc toolchain
│   ├── documents/                      # synthetic data room (copied per-task)
│   └── rubric.json                     # per-criterion rubric from upstream task.json
└── tests/
    ├── test.sh                         # runs evaluate.py, writes reward to /logs/verifier/
    └── evaluate.py                     # Claude-based LLM judge
```

Native `task.md` output is the default:

```text
corporate-ma-review-data-room-red-flag-review/
├── task.md
├── environment/
│   ├── Dockerfile
│   └── documents/
├── verifier/
│   ├── test.sh
│   ├── evaluate.py
│   ├── verifier.md
│   └── rubrics/
│       ├── rubric.json
│       ├── verifier.md
│       └── context.md
└── oracle/
    └── README.md
```

No `solution/` or `oracle/solve.sh` is generated: Harvey LAB ships no gold
deliverables. Native packages record that explicitly in `oracle/README.md`,
while the verifier package keeps the rubric outside the agent-visible image.

## Usage

### Generate tasks

```bash
cd benchmarks/harvey-lab

# All 1,251 tasks (default: --split main)
python benchflow.py --output-dir /tmp/harvey-lab-tasks --harvey-root /path/to/harvey-labs

# Legacy split packages (task.toml + tests/ + solution/)
python benchflow.py --output-dir /tmp/harvey-lab-legacy --harvey-root /path/to/harvey-labs \
    --task-format legacy

# Parity slice (first 50 tasks alphabetically)
python benchflow.py --output-dir /tmp/parity --harvey-root /path/to/harvey-labs --split parity

# XLSX slice (first 25 tasks with .xlsx deliverables)
python benchflow.py --output-dir /tmp/xlsx --harvey-root /path/to/harvey-labs --split xlsx

# Practice area filter
python benchflow.py --output-dir /tmp/re --harvey-root /path/to/harvey-labs --split real-estate

# Specific tasks
python benchflow.py --output-dir /tmp/out --harvey-root /path/to/harvey-labs \
    --task-ids "corporate-ma/analyze-cim-deal-teaser/scenario-01"
```

### Run benchmarks

```bash
# Via BenchFlow Job (downloads + converts + runs)
python benchmarks/harvey-lab/run_harvey_lab.py

# Harvey LAB harness smoke test: runs Harvey LAB's own agent loop via ACP.
# Requires GEMINI_API_KEY for the agent and ANTHROPIC_API_KEY for the verifier.
uv run bench eval create \
  --source-repo benchflow-ai/benchmarks \
  --source-path datasets/harvey-lab/tasks/corporate-ma-analyze-cim-deal-teaser-scenario-01 \
  --agent harvey-lab-harness \
  --model gemini-3.1-flash-lite-preview \
  --sandbox docker \
  --concurrency 1 \
  --jobs-dir jobs/smoke-test/harvey-harness

# Or with YAML config (uses Gemini as BenchFlow agent)
python -c "import asyncio; from benchflow.evaluation import Evaluation; asyncio.run(Evaluation.from_yaml('benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml').run())"

# Parity mode: uses the original Harvey LAB harness as the agent
python -c "import asyncio; from benchflow.evaluation import Evaluation; asyncio.run(Evaluation.from_yaml('benchmarks/harvey-lab/harvey-lab-harness-parity.yaml').run())"
```

### Run parity tests

```bash
# Structural parity (full — all 1,251 tasks)
python benchmarks/harvey-lab/parity_test.py --mode full

# Native task.md structural parity
python benchmarks/harvey-lab/parity_test.py --mode subset --task-format task-md

# Prompt-level side-by-side (requires Gemini API key)
GEMINI_API_KEY=... python benchmarks/harvey-lab/parity_test.py --mode side-by-side
```

## Comparison with Original Benchmark (Parity)

### End-to-end parity (3 runs × 100 tasks)

Both sides ran Harvey LAB's own harness (agent loop + 6 tools + system prompt) via
DirectSandbox ACP shim with `gemini-3.1-flash-lite-preview`. 100 tasks sampled with
stratified selection across all 24 practice areas. Ran on Modal (concurrency=20).

| Agent | Model | Metric | Runs | Dataset Size | Original (mean ± SEM) | BenchFlow (mean ± SEM) |
|-------|-------|--------|------|--------------|-----------------------|------------------------|
| harvey-lab-harness | gemini-3.1-flash-lite-preview | mean_per_criterion_pass_rate | 3 | 100 (8%) | **23.0 ± 0.5** | **22.2 ± 0.5** |

Per-run breakdown:

| Run | Original | BenchFlow | Delta |
|-------|----------|-----------|-------|
| 1 | 22.0% | 22.7% | +0.6% |
| 2 | 23.2% | 22.6% | -0.6% |
| 3 | 23.7% | 21.2% | -2.6% |
| **Aggregate** | **23.0%** | **22.2%** | **-0.8%** |

14,799 criteria evaluated. No systematic conversion bias — all 12 consistently-disagreeing
tasks attributed to model non-determinism. Raw run data on
[HuggingFace](https://huggingface.co/datasets/benchflow/benchmarks/tree/main/benchmarks/harvey-lab/benchflow_parity).

### Prompt-level parity (side-by-side judge agreement)

Ran the original Harvey LAB `rubric_criterion.txt` prompt and the converted BenchFlow
`string.Template` prompt through the same Gemini judge on identical synthetic output.

| Judge Model | Criteria Compared | Agreement |
|-------------|-------------------|-----------|
| gemini-3.1-flash-lite-preview | 25 (5 tasks × 5 practice areas) | **100%** |

Links:
- Original benchmark repo: https://github.com/harveyai/harvey-labs
- Benchmark PR: https://github.com/benchflow-ai/benchflow/pull/239
- Dataset PR: https://github.com/benchflow-ai/benchmarks/pull/1
- HuggingFace parity data: https://huggingface.co/datasets/benchflow/benchmarks

## Notes & Caveats

#### Instruction
Original `task.json.instructions` with minimal edits: deliverable section appended with
`output/` path convention, workspace layout hint appended. Uses `string.Template.safe_substitute()`
to avoid crashes on legal text with `{` or `}` characters.

#### Container image
Digest-pinned `python:3.13-slim` with document toolchain: `pandoc`, `pdfplumber`, `openpyxl`,
`python-docx`, `python-pptx`, `markitdown`, `pandas`, `anthropic`.

#### Judge
Claude-based LLM judge (`evaluate.py`, model: `claude-sonnet-4-6`), one call per rubric
criterion. Each criterion specifies deliverable files to grade against.
Reward = (criteria passed) / (total criteria). Same per-criterion individual grading and
default judge model as Harvey LAB's original `evaluation/judge.py`.

> **Note:** The parity data below was recorded with the previous `gemini-3.1-flash-lite-preview`
> judge. Those results validated prompt equivalence (same prompts → same verdicts), not
> judge-model equivalence. The judge was switched to `claude-sonnet-4-6` in PR #264 to match
> the original Harvey LAB benchmark's default.

#### Task naming
`harvey-lab/<practice-area>-<task-slug>[-scenario-NN]`. All lowercase, non-alphanumeric
replaced with hyphens, collapsed.

#### Agent timeout
Scales with criteria count: `max(1800, criteria_count * 30)` seconds.

#### No static oracle
Harvey LAB ships no gold deliverables. Legacy output has no `solution/`
folder. Native output uses `oracle/README.md` as evidence that the
human-authored rubric, not a static answer file, is the benchmark ground truth.

#### Harvey LAB harness agent
Ported as `harvey-lab-harness` (alias: `harvey-lab`) in BenchFlow's agent registry.
The ACP shim wraps Harvey LAB's native agent loop — same 6 tools (`bash`, `read`, `write`,
`edit`, `glob`, `grep`), same system prompt, same model adapters. `DirectSandbox` replaces
Podman with direct filesystem ops (BenchFlow's Docker provides equivalent sandboxing).

## Troubleshooting

- **Verifier returns 0 reward for all criteria.** Check that the agent produced output files
  in `/app/output/`. The verifier checks `/app/output` first, falls back to `/app`.
- **`parse-doc` not found in harness agent.** The Harvey LAB harness's `read` tool calls
  `parse-doc` for .docx/.xlsx/.pdf parsing. Ensure it's installed in PATH when running
  the ACP shim outside Docker.
- **Rate limit errors from Claude judge.** Lower concurrency or retry. The verifier calls
  Claude once per criterion (~60 calls per task).

## Citation

```bibtex
@misc{harveyai-lab-2025,
  title  = {Harvey LAB: An open-source benchmark for evaluating agents on real legal work},
  author = {Harvey AI},
  year   = {2025},
  howpublished = {\url{https://github.com/harveyai/harvey-labs}}
}
```

## Statistics

- **24** practice areas
- **1,251** tasks
- **4** work types: analyze (490), draft (444), review (293), research (24)
- **~60** criteria per task (range: 23–194)
- **Parity cost:** ~$50 (3 runs × 100 tasks on Gemini Flash Lite)
