# Harvey LAB Adapter

BenchFlow adapter for [Harvey LAB (Legal Agent Benchmark)](https://github.com/harveyai/harvey-labs) — 1,251 legal tasks across 24 practice areas.

## Overview

Harvey LAB is an open-source benchmark for evaluating agents on real legal work. Tasks span M&A, insurance, IP, tax, real estate, and more. Each task provides documents and rubric criteria graded by an LLM judge (all-pass scoring).

This adapter translates Harvey LAB tasks into BenchFlow format, preserving:
- **Instructions** → `instruction.md`
- **Documents** → baked into the Docker environment
- **Rubric criteria** → LLM-as-judge verifier (`tests/evaluate.py` using Gemini)
- **Metadata** (practice area, work type, tags) → `task.toml` metadata

## Task Structure

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
```

### Run parity tests

```bash
# Structural parity (subset)
python benchmarks/harvey-lab/parity_test.py --mode subset

# Structural parity (full)
python benchmarks/harvey-lab/parity_test.py --mode full

# Evaluation parity (LLM judge — requires Gemini API key)
GEMINI_API_KEY=... python benchmarks/harvey-lab/parity_test.py \
    --mode eval-parity --gemini-api-key $GEMINI_API_KEY
```

### Run benchmarks

```bash
# Via BenchFlow CLI
bench run datasets/harvey-lab --agent gemini --backend docker

# Via script
python benchmarks/harvey-lab/run_harvey_lab.py
```

## Evaluation

The verifier uses Gemini as an LLM-as-judge. For each task criterion:
1. Reads the agent's deliverable files (.docx, .xlsx, .pdf, .md, etc.)
2. Formats a judge prompt with the criterion and deliverable content
3. Gets a PASS/FAIL verdict from Gemini
4. Reward = (criteria passed) / (total criteria)

Set `GEMINI_API_KEY` in your environment or in `task.toml`'s `[verifier.env]`.

## Statistics

- **24** practice areas
- **1,251** tasks
- **4** work types: analyze (490), draft (444), review (293), research (24)
- **~60** criteria per task (range: 23–194)
