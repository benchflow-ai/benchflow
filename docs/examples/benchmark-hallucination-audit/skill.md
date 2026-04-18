---
name: benchmark-hallucination-audit
description: >
  Exhaustive multi-source verification of claims in benchmark comparison tables.
  Probes every row and every cell against paper, appendix, GitHub, HuggingFace, and project websites.
  Uses layered subagent parallelization across multiple rounds.
trigger: >
  Use when the user asks to verify, probe, fact-check, or audit claims in a benchmark paper's
  comparison table, survey table, or related-work table. Also use when asked to check for
  hallucinations in academic papers, especially tables comparing multiple benchmarks.
---

# Benchmark Hallucination Audit

## Philosophy

Benchmark comparison tables are the most hallucination-prone part of any survey or evaluation paper. Authors fill in 50-100+ rows of boolean claims about other benchmarks, often from memory or abstracts alone. The result: ~10-15% cell-level error rate, systematically biased toward making the authors' benchmark look more unique than it is.

**Core principle**: Never trust a single source. A paper's abstract may contradict its appendix, GitHub README, or HuggingFace dataset card. Each source has different information density:

| Source | What it reveals | What it hides |
|--------|----------------|---------------|
| Abstract | High-level claims | Evaluation details, data formats |
| Paper body/appendix | Methodology, task examples, eval paradigms | Leaderboard status, community evolution |
| GitHub | README, code structure, leaderboard, versioning, issue activity | Claims the paper makes but code doesn't support |
| HuggingFace | Dataset cards, modalities, actual data samples, version history | Often absent for newer benchmarks |
| Project website | Leaderboard, submission mechanism, domain breakdown | May be aspirational vs actual |

## Process: Layered multi-round subagent audit

### Round 0: Extract the table (1 agent, ~10 min)
1. Fetch the paper PDF via WebFetch or read local copy
2. Convert to text (pdftotext -layout)
3. Extract the full table: every row, every column, every cell value
4. Extract the reference list (map each row to its citation/arXiv ID)
5. Identify column definitions from the table caption

### Round 1: Existence check (4 parallel agents, ~20 min)
- Verify every cited paper/benchmark actually exists
- Check arXiv IDs resolve to real papers
- Verify author names match
- Flag any benchmark names that don't match the cited paper's title
- Output: per-row verdict [VERIFIED / UNVERIFIED / NAME-MISMATCH / HALLUCINATION]

### Round 2: Abstract-level cell scan (4 parallel agents, ~30 min)
- For each row, fetch the arXiv abstract
- Check each ✓ cell against the abstract (is the claim supported?)
- Check each ✗ cell against the abstract (does the abstract contradict the ✗?)
- Output: per-cell flags [OK / OVERCLAIM / MISSING-✓ / UNVERIFIABLE]
- **Warning**: This round will produce many false positives. It's a starting point, not final.

### Round 3: Deep 4-source audit (6 parallel agents × 5 rows each, ~60 min per batch)
This is the core audit. For each row:

1. **Fetch the paper** (full text if possible, not just abstract)
   - Read methodology, evaluation sections, limitations
   - Check appendix for task examples, domain breakdown, eval details

2. **Check GitHub**
   - README: does it confirm/contradict paper claims?
   - Leaderboard: does it exist? Is it live? Submission mechanism?
   - Code: eval scripts reveal actual paradigms used
   - Versioning: releases, growing task pools, active development

3. **Check HuggingFace**
   - Dataset card: modalities, size, format, domain tags
   - Actual data samples: are inputs really multi-modal?
   - Version history: was it updated?
   - Leaderboard spaces: do they exist?

4. **Check project website** (if exists)
   - Leaderboard status
   - Domain enumeration
   - "About" / FAQ pages

Per cell, produce a structured verdict:
```
CELL: VERDICT (paper: QUOTE | github: QUOTE | HF: QUOTE | web: QUOTE) → FINAL ✓/✗
```

### Round 4: Correction round (after all deep audits complete)
- Compare Round 2 (aggressive) findings with Round 3 (deep) findings
- Overturn any aggressive claims not supported by deep evidence
- This round typically REDUCES the error count by 30-50% — the aggressive pass overcounts
- Update the visualization

### Round 5: Self-audit (1 agent)
- Verify the paper's OWN row in the table (they always mark themselves ✓ on everything)
- Check internal numerical consistency (do tables agree with each other?)
- Check for structural issues (conflict of interest, undisclosed relationships)

## Subagent dispatch strategy

**Batch size**: 5-6 rows per subagent for deep audit (Round 3). Each row needs ~6-8 WebFetch calls across 4 sources.

**Parallelization**: Launch 6 agents simultaneously. Each gets:
- Clear column definitions (copy-paste the exact definitions from the paper)
- Exact rows with current cell values
- Specific suspicions to investigate per row
- Budget (~40 WebFetch calls)
- Required output format

**Priority order**: Audit the most-contested rows first:
1. Rows where the paper claims uniqueness (e.g., "Production" column — usually only 2-3 benchmarks get ✓)
2. Rows with the most ✓ marks (overclaim candidates)
3. Rows from the same research group as the authors
4. Rows for very recent (2026) papers (highest hallucination risk)
5. Rows with all ✗ marks (under-credit candidates)

## Common error patterns to watch for

1. **Cross-Domain inflation**: Authors conflate "task types" (OS/DB/Web/Game) with "professional domains" (medicine/law/finance). Most agent benchmarks fail this under strict reading.

2. **Dynamic aspiration vs mechanism**: "We plan to update" ≠ "designed for continuous evolution." Check for: live leaderboard, versioning, submission mechanism, actual version history. Annual competitions (AIME, ARC Prize) do qualify.

3. **Expert Val scope**: "Expert annotators" who validate data quality ≠ "domain experts co-developing evaluation criteria." Graduate students ≠ industry professionals.

4. **Multi-Modal confusion**: Agent interface modality (terminal vs GUI) ≠ data input modality (PDFs/spreadsheets/images). A terminal-based benchmark whose tasks process PDFs IS multi-modal.

5. **Production generosity**: "Tasks inspired by real workflows" ≠ "tasks sourced from real commercial deployments." Bug bounties with real payouts DO qualify. Kaggle competitions do NOT (offline reproductions).

6. **Diverse Eval conflation**: Multiple metrics within one paradigm (accuracy + F1 + precision) ≠ multiple paradigms (execution + rubric + LLM-judge). Count paradigms, not metrics.

7. **Self-grading bias**: The paper's own row invariably gets all ✓ marks. The Dynamic claim is usually the weakest — check for actual mechanism vs aspiration.

## Output: Interactive HTML visualization

Generate a single-file HTML with:
- Left panel: full table with color-coded cells (green=correct, red=overclaim, yellow=missing, gray=unverifiable)
- Right panel: click any cell for proof trail from all sources
- Tally section with error counts
- Structural concerns section (conflicts of interest, internal inconsistencies)

## Resource estimation

For a table with N benchmark rows:
- Round 0-1: ~15 min, minimal token usage
- Round 2: ~30 min, ~4 agents
- Round 3: ~N/5 batches × ~60 min each, ~N/5 × 6 agents
- Round 4-5: ~30 min, ~2 agents
- Total for 94-row table: ~5 hours on Claude Code Max (Opus 4.6, 1M context)
- WebFetch calls: ~8 per row = ~750 total

## Example prompt for Round 3 subagent

```
EXHAUSTIVE 4-SOURCE CELL AUDIT. For each of 5 benchmarks, verify ALL 7
Table-1 cells using paper + GitHub + HuggingFace. Per cell: quote evidence.

Strict definitions:
- Production: real commercial deployments with paying customers
- Multi-Modal: PDFs/spreadsheets/images/video/audio inputs
- Underspec.: deliberately preserves ambiguity
- Diverse Eval: 3+ distinct paradigms
- Expert Val.: co-developed with domain experts
- Dynamic: designed for continuous evolution
- Cross-Domain: 3+ distinct PROFESSIONAL domains

ROWS:
1. **BenchmarkName (Author Year)** — arXiv:XXXX.XXXXX. Currently [✗,✓,✗,✗,✗,✗,✗].
   - Paper: WebFetch https://arxiv.org/abs/XXXX.XXXXX
   - GitHub: WebSearch "BenchmarkName github"
   - HuggingFace: WebSearch "benchmarkname huggingface"

OUTPUT per benchmark:
## BENCHMARK
Sources: [paper ✓/✗ | github ✓/✗ | HF ✓/✗]
Prod: VERDICT (evidence) → FINAL ✓/✗
MM:   VERDICT → FINAL
...all 7 cells...
```
