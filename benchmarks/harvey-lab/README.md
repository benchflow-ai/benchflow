# Harvey LAB

[Harvey LAB (Legal Agent Benchmark)](https://github.com/harveyai/harvey-labs) in BenchFlow format — 1,251 legal tasks across 24 practice areas.

## Overview

Harvey LAB is an open-source benchmark for evaluating agents on real legal work. Tasks span M&A, insurance, IP, tax, real estate, and more. Each task provides documents and rubric criteria graded by an LLM judge (all-pass scoring).

This benchmark converts Harvey LAB tasks into BenchFlow format, preserving:
- **Instructions** → `instruction.md`
- **Documents** → baked into the Docker environment
- **Rubric criteria** → LLM-as-judge verifier (`tests/evaluate.py` using Gemini)
- **Metadata** (practice area, work type, tags) → `task.toml` metadata

## Directory Structure

```
benchmarks/harvey-lab/
├── benchflow.py                     # converter: Harvey LAB task.json → BenchFlow task format
├── parity_test.py                   # structural, eval, and side-by-side parity tests
├── run_harvey_lab.py                # runner: download + convert + run via Job
├── harvey-lab-gemini-flash-lite.yaml # BenchFlow-native YAML config
├── parity_experiment.json           # side-by-side parity results (prompt-level)
├── parity_final_report.md           # 3-trial end-to-end parity report
├── harvey-lab-harness-parity.yaml   # parity config using Harvey LAB's own harness agent
├── benchmark.yaml                   # standard benchmark descriptor
└── README.md
```

### BenchFlow Benchmark Convention

| File | Purpose |
|---|---|
| `benchflow.py` | Converter CLI: `--output-dir`, `--limit`, `--overwrite`, `--task-ids` |
| `run_<name>.py` | Runner: downloads raw tasks via `ensure_tasks()`, converts, runs via `Job` |
| `<name>.yaml` | BenchFlow-native YAML config (`tasks_dir`, `agent`, `model`, `environment`) |
| `parity_test.py` | Validates structural, eval, and side-by-side parity |
| `parity_experiment.json` | Records side-by-side parity results |
| `benchmark.yaml` | Standard benchmark descriptor (metadata, verification, parity) |

## Task Mapping

| Harvey LAB | BenchFlow |
|---|---|
| `task.json` (title, instructions, criteria) | `task.toml` + `instruction.md` |
| `documents/` (input docs) | `environment/documents/` (COPY'd in Dockerfile) |
| LLM judge with rubric criteria | `tests/evaluate.py` (Gemini-based judge) |
| No oracle solutions provided | No `solution/` directory |

## Usage

### Generate tasks

```bash
# All 1,251 tasks
python benchmarks/harvey-lab/benchflow.py \
    --output-dir /tmp/harvey-lab-tasks \
    --harvey-root /path/to/harvey-labs

# Subset
python benchmarks/harvey-lab/benchflow.py \
    --output-dir /tmp/harvey-lab-tasks \
    --harvey-root /path/to/harvey-labs \
    --limit 10

# Specific tasks
python benchmarks/harvey-lab/benchflow.py \
    --output-dir /tmp/harvey-lab-tasks \
    --harvey-root /path/to/harvey-labs \
    --task-ids "corporate-ma/analyze-cim-deal-teaser/scenario-01"
```

### Run parity tests

```bash
# Structural parity (subset — 5 tasks)
python benchmarks/harvey-lab/parity_test.py --mode subset

# Structural parity (full — all 1,251 tasks)
python benchmarks/harvey-lab/parity_test.py --mode full

# Eval pipeline end-to-end (requires Gemini API key)
GEMINI_API_KEY=... python benchmarks/harvey-lab/parity_test.py \
    --mode eval-parity --gemini-api-key $GEMINI_API_KEY

# Side-by-side parity (original vs adapted prompt, same judge)
GEMINI_API_KEY=... python benchmarks/harvey-lab/parity_test.py \
    --mode side-by-side --gemini-api-key $GEMINI_API_KEY
```

### Run benchmarks

```bash
# Via BenchFlow Job (downloads + converts + runs)
python benchmarks/harvey-lab/run_harvey_lab.py

# Or with YAML config (uses Gemini as BenchFlow agent)
python -c "import asyncio; from benchflow.job import Job; asyncio.run(Job.from_yaml('benchmarks/harvey-lab/harvey-lab-gemini-flash-lite.yaml').run())"

# Parity mode: uses the original Harvey LAB harness as the agent
# (same tools, same system prompt, same agent loop — for true apples-to-apples comparison)
python -c "import asyncio; from benchflow.job import Job; asyncio.run(Job.from_yaml('benchmarks/harvey-lab/harvey-lab-harness-parity.yaml').run())"
```

## Parity Results

### Step-by-step validation

| Step | Test | Result |
|---|---|---|
| 1 | Understand original benchmark | Harvey LAB: 1,251 tasks, 24 practice areas, LLM-judge evaluation |
| 2 | Converter code complete | `benchflow.py` with `--output-dir`, `--limit`, `--overwrite`, `--task-ids` |
| 3 | Oracle verification | N/A — Harvey LAB has no oracle solutions; cheap agent pass validates solvability |
| 4 | Plan parity & implement agents | Gemini 3.1 Flash Lite used as both agent model and judge |
| 5 | **Side-by-side parity** | **25/25 criteria agree (100%)** across 5 practice areas |
| 6 | Record parity results | `parity_experiment.json` |
| 7 | Upload results | Included in PR |
| 8 | Register dataset | `harvey-lab` registered in `task_download.py` |
| 9 | Document & submit | This README + `benchmark.yaml` |

### Side-by-side parity details

Ran the original Harvey LAB `rubric_criterion.txt` prompt template and the converted BenchFlow `string.Template` prompt through the same Gemini 3.1 Flash Lite judge on identical synthetic agent output:

| Task | Practice Area | Criteria Tested | Agreement |
|---|---|---|---|
| analyze-cim-deal-teaser | Corporate M&A | 5/5 | 100% |
| compare-reinsurance-treaty | Insurance | 5/5 | 100% |
| draft-construction-contract | Real Estate | 5/5 | 100% |
| review-enterprise-saas | IP | 5/5 | 100% |
| draft-workplace-policy | Employment | 5/5 | 100% |
| **Total** | | **25/25** | **100%** |

## Comparison with Original Benchmark (Parity)

### Prompt-level parity (side-by-side judge agreement)

Full results are recorded in [`parity_experiment.json`](parity_experiment.json).

| Judge Model | Metric | Dataset Size | Parity Size | Criteria Compared | Agreement |
|---|---|---|---|---|---|
| gemini-3.1-flash-lite-preview | side-by-side agreement | 1,251 | 5 tasks (5 practice areas) | 25 | **100%** |

### End-to-end parity (3 trials × 100 tasks)

Ran Harvey LAB's own harness (agent loop + 6 tools + system prompt) via DirectSandbox on both original and BenchFlow-converted tasks with `gemini-3.1-flash-lite-preview`. Full report in [`parity_final_report.md`](parity_final_report.md). Raw trial data on [HuggingFace](https://huggingface.co/datasets/benchflow/benchmarks/tree/main/benchmarks/harvey-lab/benchflow_parity).

| Trial | Original | BenchFlow | Delta |
|---|---|---|---|
| 1 | 22.0% | 22.7% | +0.6% |
| 2 | 23.2% | 22.6% | -0.6% |
| 3 | 23.7% | 21.2% | -2.6% |
| **Aggregate** | **23.0%** | **22.2%** | **-0.8%** |

14,799 criteria evaluated. No systematic conversion bias — all disagreements from model non-determinism.

Links:
- Original benchmark repo: https://github.com/harveyai/harvey-labs
- Parity experiments (HF): https://huggingface.co/datasets/benchflow/benchmarks

## Evaluation

The verifier uses Gemini as an LLM-as-judge. For each task criterion:
1. Reads the agent's deliverable files (.docx, .xlsx, .pdf, .md, etc.)
2. Formats a judge prompt via `string.Template.safe_substitute()` (safe against injection)
3. Gets a PASS/FAIL verdict from Gemini
4. Reward = (criteria passed) / (total criteria)

Set `GEMINI_API_KEY` in your environment or in `task.toml`'s `[verifier.env]`.

## Statistics

- **24** practice areas
- **1,251** tasks
- **4** work types: analyze (490), draft (444), review (293), research (24)
- **~60** criteria per task (range: 23–194)
